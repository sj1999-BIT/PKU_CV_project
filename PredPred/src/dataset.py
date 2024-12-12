import torch


class Dataset:
    def __init__(self, path, device):
        self.xs, self.ys, self.obj_names, self.pred_names = torch.load(path, map_location=device, weights_only=True)
        assert len(self.xs) == len(self.ys)
        assert len(self.xs) != 0

    def input_size(self):
        return len(self.xs[0])

    def output_size(self):
        return len(self.ys[0])

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])
        
    def __len__(self):
        return len(self.xs)
