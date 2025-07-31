import torch
from torch.utils.data import Dataset

class MatrixDataset(Dataset):
    def __init__(self, args):
        assert args.g_mode in ('sum', 'mean', 'prod', 'max')
        # Args parameters
        self.n_samples = args.n_samples
        self.d = args.d
        self.N = args.N
        self.low, self.high = args.low, args.high
        self.g_mode = args.g_mode
        self.variable_n = args.variable_n

        if self.variable_n:
            self.Nmin, self.Nmax = args.Nmin, args.Nmax

        # Define a generator to create random matrices
        self.rng  = torch.Generator().manual_seed(0 if args.seed is None else args.seed)

        if self.n_samples == 1 and not self.variable_n:
            X = torch.randint(self.low, self.high+1, (self.N, self.d), generator=self.rng).float()
            shared = self.compute_shared(X)
            Y = X**2 + shared

            # Reuse this data sample every call
            self._cached_X = X
            self._cached_Y = Y

    def __len__(self):
        return self.n_samples
    
    def compute_shared(self, X):
        if self.g_mode == 'sum':
            return X.sum(dim=0)
        elif self.g_mode == 'mean':
            return X.mean(dim=0)
        elif self.g_mode == 'prod':
            return X.prod(dim=0)
        elif self.g_mode == 'max':
            return X.max(dim=0).values

    def __getitem__(self, idx):
        if self.n_samples == 1 and not self.variable_n:
            return self._cached_X, self._cached_Y
        
        if self.variable_n:
            N = torch.randint(self.Nmin, self.Nmax + 1, (1,), generator=self.rng).item()
        else:
            N = self.N

        X = torch.randint(self.low, self.high+1, (N, self.d), generator=self.rng).float()

        shared = self.compute_shared(X)
        
        Y = X**2 + shared
        return X, Y