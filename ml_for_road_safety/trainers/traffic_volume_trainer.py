import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from logger import Logger
import os

class VolumeRegressionTrainer:

    def __init__(self, model, predictor, dataset, optimizer, evaluator,
                 train_years, valid_years, test_years,
                 epochs, batch_size, eval_steps, device,
                 log_metrics = ['MAE', 'MSE'],
                 use_time_series = False, input_time_steps = 12
                 ): 
        self.model = model
        self.predictor = predictor
        self.dataset = dataset
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.train_years = train_years
        self.valid_years = valid_years
        self.test_years = test_years

        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.device = device

        self.loggers = {
            key: Logger(runs=1) for key in log_metrics
        }

        # self.save_steps = save_steps
        # self.checkpoint_dir = checkpoint_dir
        # if not os.path.exists(self.checkpoint_dir):
        #     os.makedirs(self.checkpoint_dir)

        self.use_time_series = use_time_series
        self.input_time_steps = input_time_steps

    def train_on_year_data(self, year): 
        yearly_data = self.dataset.load_yearly_data(year)

        if self.use_time_series:
            list_x = [yearly_data['data'].x]
            for i in range(self.input_time_steps-1):
                list_x.append(yearly_data['data'].x.clone())
            inputs = torch.stack(list_x, dim=0).unsqueeze(0)
        else:
            inputs = yearly_data['data'].x

        new_data = yearly_data['data']
        pos_edges, pos_edge_weights = \
            yearly_data['traffic_volume_edges'], yearly_data['traffic_volume_weights']
        
        if pos_edges is None or pos_edges.size(0) < 10:
            return 0, 0
        
        self.model.train()
        self.predictor.train()

        # encoding
        new_data = new_data.to(self.device); inputs = inputs.to(self.device)
        h = self.model(inputs, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr
        if len(h.size()) == 4:
            h = h.squeeze(0)[-1, :, :]
        if len(h.size()) == 3:
            h = h[-1, :, :]

        # predicting
        pos_edge_weights = pos_edge_weights.to(self.device)
        pos_train_edge = pos_edges.to(self.device)
        total_loss = total_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size, shuffle=True):
            self.optimizer.zero_grad()
            # positive edges
            edge = pos_train_edge[perm].t()
            pos_out = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            
            labels = pos_edge_weights.view(-1, 1).to(self.device)
            # print(pos_out)
            # print(labels)
            loss = self.evaluator.criterion(pos_out, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples
        
        return total_loss, total_examples

    @torch.no_grad()
    def test_on_year_data(self, year):
        yearly_data = self.dataset.load_yearly_data(year)

        if self.use_time_series:
            list_x = [yearly_data['data'].x]
            for i in range(self.input_time_steps-1):
                list_x.append(yearly_data['data'].x.clone())
            inputs = torch.stack(list_x, dim=0).unsqueeze(0)
        else:
            inputs = yearly_data['data'].x

        new_data = yearly_data['data']
        pos_edges, pos_edge_weights = yearly_data['traffic_volume_edges'], yearly_data['traffic_volume_weights']
        
        if pos_edges is None or pos_edges.size(0) < 10:
            return {}, 0

        print(f"Eval on {year} data")
        print(f"Number of edges with valid traffic volume: {pos_edges.size(0)}")

        self.model.eval()
        self.predictor.eval()

        # encoding
        new_data = new_data.to(self.device); inputs = inputs.to(self.device)
        h = self.model(inputs, new_data.edge_index, new_data.edge_attr)
        edge_attr = new_data.edge_attr
        if len(h.size()) == 4:
            h = h.squeeze(0)[-1, :, :]
        if len(h.size()) == 3:
            h = h[-1, :, :]

        # predicting
        pos_edge = pos_edges.to(self.device)
        pos_preds = []
        for perm in DataLoader(range(pos_edge.size(0)), self.batch_size):
            edge = pos_edge[perm].t()
            preds = self.predictor(h[edge[0]], h[edge[1]]) \
                if edge_attr is None else \
                self.predictor(h[edge[0]], h[edge[1]], edge_attr[perm])
            pos_preds += [preds.squeeze().cpu()] 
        pos_preds = torch.cat(pos_preds, dim=0)

        # Eval ROC-AUC
        results = self.evaluator.eval(pos_preds, pos_edge_weights)

        return results, pos_edges.size(0)


    def train_epoch(self):
        total_loss = total_examples = 0
        for year in self.train_years:
            loss, samples = self.train_on_year_data(year)
            total_loss += loss
            total_examples += samples
        return total_loss/total_examples
    

    def train(self):
        train_log = {}
        for epoch in range(1, 1 + self.epochs):
            loss = self.train_epoch()

            if epoch % self.eval_steps == 0:
                results = self.test()
                for key, result in results.items():
                    self.loggers[key].add_result(run=0, result=result)
            
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_hits:.4f}, '
                          f'Valid: {valid_hits:.4f}, '
                          f'Test: {test_hits:.4f}')
                print('---')

                # if epoch % self.save_steps == 0:
                #     torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))

        for key in self.loggers.keys():
            print(key)
            mode = 'min' if (key == 'Loss' or key == "MAE" or key == "MSE") else 'max'
            train, valid, test = self.loggers[key].print_statistics(run=0, mode=mode)
            train_log[f"Train_{key}"] = train
            train_log[f"Valid_{key}"] = valid
            train_log[f"Test_{key}"] = test
            
        return train_log

    def test(self):
        train_results = {}; train_size = 0
        for year in self.train_years:
            month_results, month_sample_size = self.test_on_year_data(year)
            for key, value in month_results.items():
                if key not in train_results:
                    train_results[key] = 0
                train_results[key] += value * month_sample_size
            train_size += month_sample_size

        for key, value in train_results.items():
            train_results[key] = value / train_size

        val_results = {}; val_size = 0
        for year in self.valid_years:
            month_results, month_sample_size = self.test_on_year_data(year)
            for key, value in month_results.items():
                if key not in val_results:
                    val_results[key] = 0
                val_results[key] += value * month_sample_size
            val_size += month_sample_size
        
        for key, value in val_results.items():
            val_results[key] = value / val_size

        test_results = {}; test_size = 0
        for year in self.test_years:
            month_results, month_sample_size = self.test_on_year_data(year)
            for key, value in month_results.items():
                if key not in test_results:
                    test_results[key] = 0
                test_results[key] += value * month_sample_size
            test_size += month_sample_size
            
        for key, value in test_results.items():
            test_results[key] = value / test_size

        results = {}
        for key in train_results.keys():
            if key not in val_results:
                test_results[key] = 0
            if key not in test_results:
                test_results[key] = 0
            results[key] = (train_results[key], val_results[key], test_results[key])
        return results
    