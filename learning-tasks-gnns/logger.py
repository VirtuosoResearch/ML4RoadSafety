import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Valid: {result[:, 1].max():.2f}')
            print(f'Test: {result[argmax, 2]:.2f}')
            return result[:, 0].max(), result[:, 1].max(), result[argmax, 2]
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train, valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Test: {r.mean():.2f} ± {r.std():.2f}')
