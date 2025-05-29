import torch


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noisy_tensor = tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return torch.clamp(noisy_tensor, 0., 1.)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class AddRandomBackground(object):
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def __call__(self, tensor):
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            return tensor
        foreground_mask = tensor > self.threshold
        random_background = torch.rand_like(tensor, device=tensor.device)
        output_tensor = torch.where(foreground_mask, tensor, random_background)
        return output_tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(threshold={self.threshold})'
