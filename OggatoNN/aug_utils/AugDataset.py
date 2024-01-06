import torch


class AugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.x = dataset[:, 0]
        self.x = self.x.float()

        self.y = torch.from_numpy(dataset[:, 1])
        self.y = self.y.long()

        self.n_samples = dataset.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]
