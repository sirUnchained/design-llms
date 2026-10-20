import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler):
    """
    ## A deterministic, resumable replacement for shuffle=True.

    Given a fixed `seed` and `epoch`, this always generates the *same* full permutation of dataset indices - so the shuffle order for any
    given epoch is fully reproducible - and can start yielding from partway through that permutation via `start_index`. That's what lets
    use resume an interrupted epoch without re-feeding batches the model already trained on.

    Args:
        data_source: Dataset to sample from (only `len()` is used).
        seed (int): Base seed, shared across all epochs of a run.
        epoch (int): Current epoch number. Mixed into the seed so every epoch gets a distinct-but-reproducible shuffle order.
        start_index (int): How many samples into this epoch's permutation
                            to start from. 0 for a fresh epoch; on resume, set to however many batches * batch_size were already consumed.
    """

    def __init__(self, data_source, seed: int, epoch: int, start_index: int = 0):
        self.num_samples = len(data_source)
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.num_samples, generator=g).tolist()
        return iter(perm[self.start_index :])

    def __len__(self):
        return self.num_samples - self.start_index
