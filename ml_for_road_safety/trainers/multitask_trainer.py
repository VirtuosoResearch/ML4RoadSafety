from trainers.trainer import Trainer
from trainers.regression_trainer import AccidentRegressionTrainer
from trainers.traffic_volume_trainer import VolumeRegressionTrainer

from data_loaders import load_monthly_data, load_yearly_data
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from evaluators import eval_mae, eval_hits, eval_rocauc
from logger import Logger
from torch_geometric.loader import NeighborLoader
import os

state_to_train_years = {
"DE": [2009, 2010, 2011, 2012],
"IA": [2013, 2014, 2015, 2016],
"IL": [2012, 2013, 2014],
"MA": [2002, 2003, 2004, 2005, 2006, 2007, 2008],
"MD": [2015, 2016, 2017],
"MN": [2015, 2016, 2017],
"MT": [2016, 2017],
"NV": [2016, 2017],
}

state_to_valid_years = {
"DE": [2013, 2014, 2015, 2016, 2017],
"IA": [2017, 2018, 2019],
"IL": [2015, 2016, 2017],
"MA": [2009, 2010, 2011, 2012, 2013, 2014, 2015],
"MD": [2018, 2019],
"MN": [2018, 2019],
"MT": [2018],
"NV": [2018],
}

state_to_test_years = {
"DE": [2018, 2019, 2020, 2021, 2022],
"IA": [2020, 2021, 2022],
"IL": [2018, 2019, 2020, 2021],
"MA": [2016, 2017, 2018, 2019, 2020, 2021, 2022],
"MD": [2020, 2021, 2022],
"MN": [2020, 2021, 2022],
"MT": [2019, 2020],
"NV": [2019, 2020],
}

class MultitaskTrainer:

    def __init__(self, model, optimizer, data_dir,
                 epochs, batch_size, eval_steps, device, 
                 save_steps, checkpoint_dir,
                 use_dynamic_node_features=False, 
                 use_dynamic_edge_features=False, 
                 num_negative_edges=10000, 
                 node_feature_mean=None, node_feature_std=None, edge_feature_mean=None, edge_feature_std=None, 
                 tasks={}, task_to_datas={}, task_to_predictors={}):
        self.model = model
        self.tasks = tasks
        self.task_to_datas = task_to_datas
        self.task_to_predictors = task_to_predictors
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.task_to_trainers = {}
        for task_name in tasks:
            print(task_name)
            state_name, data_type, task_type = task_name.split("_")
            data = task_to_datas[task_name]
            predictor = task_to_predictors[task_name]

            task_train_years = state_to_train_years[state_name]
            task_valid_years = state_to_valid_years[state_name]
            task_test_years = state_to_test_years[state_name]
            if data_type == "accident":
                if task_type == "classification":
                    task_trainer = Trainer(model, predictor, data, optimizer,
                                    data_dir=data_dir, state_name=state_name,
                                    train_years = task_train_years,
                                    valid_years = task_valid_years,
                                    test_years = task_test_years,
                                    epochs=epochs,
                                    batch_size = batch_size,
                                    eval_steps = eval_steps,
                                    device = device,
                                    use_dynamic_node_features=use_dynamic_node_features,
                                    use_dynamic_edge_features=use_dynamic_edge_features,
                                    log_metrics=['ROC-AUC', 'F1', 'AP', 'Hits@100'],
                                    num_negative_edges=num_negative_edges,
                                    node_feature_mean=node_feature_mean, node_feature_std=node_feature_std,
                                    edge_feature_mean=edge_feature_mean, edge_feature_std=edge_feature_std)
                elif task_type == "regression":
                    task_trainer = AccidentRegressionTrainer(model, predictor, data, optimizer,
                            data_dir=data_dir, state_name=state_name,
                            train_years = task_train_years,
                            valid_years = task_valid_years,
                            test_years = task_test_years,
                            epochs=epochs,
                            batch_size = batch_size,
                            eval_steps=eval_steps,
                            device = device,
                            use_dynamic_node_features=use_dynamic_node_features,
                            use_dynamic_edge_features=use_dynamic_edge_features,
                            log_metrics=['MAE', 'MSE'],
                            node_feature_mean=node_feature_mean, node_feature_std=node_feature_std,
                            edge_feature_mean=edge_feature_mean, edge_feature_std=edge_feature_std,)
            else:
                task_trainer = VolumeRegressionTrainer(model, predictor, data, optimizer,
                            data_dir=data_dir, state_name=state_name,
                            train_years = task_train_years,
                            valid_years = task_valid_years,
                            test_years = task_test_years,
                            epochs=epochs,
                            batch_size = batch_size,
                            eval_steps=eval_steps,
                            device = device,
                            use_dynamic_node_features=use_dynamic_node_features,
                            use_dynamic_edge_features=use_dynamic_edge_features,
                            log_metrics=['MAE', 'MSE'],
                            node_feature_mean=node_feature_mean, node_feature_std=node_feature_std,
                            edge_feature_mean=edge_feature_mean, edge_feature_std=edge_feature_std,)    
                
            self.task_to_trainers[task_name] = task_trainer

    def train(self):
        train_log = {}
        for epoch in range(1, 1 + self.epochs):
            task_list = self.tasks[:]
            np.random.shuffle(task_list)

            for task_name in task_list:
                task_trainer = self.task_to_trainers[task_name]
                task_loss = task_trainer.train_epoch()

            if epoch % self.eval_steps == 0:
                for task_name in self.tasks:
                    task_trainer = self.task_to_trainers[task_name]

                    results = task_trainer.test()
                    for key, result in results.items():
                        task_trainer.loggers[key].add_result(run=0, result=result)
                
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(task_name)
                        print(key)
                        print(f'Epoch: {epoch:02d}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    print('---')
            
            if epoch % self.save_steps == 0:
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth'))

        for task_name in self.tasks:
            task_trainer = self.task_to_trainers[task_name]
            for key in task_trainer.loggers.keys():
                print(task_name)
                print(key)
                mode = 'min' if (key == 'Loss' or key == "MAE" or key == "MSE") else 'max'
                train, valid, test = task_trainer.loggers[key].print_statistics(run=0, mode=mode)
                train_log[f"Train_{task_name}_{key}"] = train
                train_log[f"Valid_{task_name}_{key}"] = valid
                train_log[f"Test_{task_name}_{key}"] = test
        return train_log