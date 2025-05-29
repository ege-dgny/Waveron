import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
from .custom_transforms import AddGaussianNoise, AddRandomBackground


def get_dataset(dataset_name, data_root='./data', target_img_size=(64, 64),
                batch_size=64, noise_std=0.2, bg_rand_threshold=0.1):
    input_channels = 1
    num_classes = None
    normalize_grayscale = transforms.Normalize((0.5,), (0.5,))
    mnist_mean, mnist_std = (0.1307,), (0.3081,)

    if dataset_name.lower() in ['mnist', 'mnist_noisy', 'mnist_bg_rand', 'mnist_noisy_bg_rand']:
        img_h, img_w = 28, 28
        transform_list = [
            transforms.Resize((img_h, img_w)),
            transforms.ToTensor(),
        ]
        # Apply noise/bg_rand transforms *before* normalization if they expect [0,1] range
        if dataset_name.lower() == 'mnist_noisy':
            print(f"Applying Gaussian noise with std: {noise_std}")
            transform_list.append(AddGaussianNoise(std=noise_std))
        elif dataset_name.lower() == 'mnist_bg_rand':
            print(f"Applying random background with threshold: {bg_rand_threshold}")
            transform_list.append(AddRandomBackground(threshold=bg_rand_threshold))
        elif dataset_name.lower() == 'mnist_noisy_bg_rand':
            print(f"Applying Gaussian noise (std: {noise_std}) then random background (thresh: {bg_rand_threshold})")
            transform_list.append(AddGaussianNoise(std=noise_std))
            transform_list.append(AddRandomBackground(threshold=bg_rand_threshold))

        transform_list.append(transforms.Normalize(mnist_mean, mnist_std))

        train_dataset = torchvision.datasets.MNIST(root=data_root, train=True,
                                                   transform=transforms.Compose(transform_list), download=True)
        test_dataset = torchvision.datasets.MNIST(root=data_root, train=False,
                                                  transform=transforms.Compose(transform_list), download=True)
        num_classes = 10
        input_channels = 1

    elif dataset_name.lower() == 'usps':
        img_h, img_w = target_img_size
        transform_list = [transforms.Resize((img_h, img_w)), transforms.ToTensor(), normalize_grayscale]
        train_dataset = torchvision.datasets.USPS(root=data_root, train=True,
                                                  transform=transforms.Compose(transform_list), download=True)
        test_dataset = torchvision.datasets.USPS(root=data_root, train=False,
                                                 transform=transforms.Compose(transform_list), download=True)
        num_classes = 10
        input_channels = 1

    elif dataset_name.lower() == 'caltech101':
        img_h, img_w = target_img_size
        transform_list = [transforms.Resize((img_h, img_w)), transforms.Grayscale(num_output_channels=1),
                          transforms.ToTensor(), normalize_grayscale]
        try:
            full_dataset = torchvision.datasets.Caltech101(root=data_root, transform=transforms.Compose(transform_list),
                                                           download=True)
            num_classes = len(full_dataset.categories)
            train_size = int(0.8 * len(full_dataset));
            test_size = len(full_dataset) - train_size
            generator = torch.Generator().manual_seed(42)
            train_indices, test_indices = torch.utils.data.random_split(range(len(full_dataset)),
                                                                        [train_size, test_size], generator=generator)
            train_dataset = Subset(full_dataset, train_indices);
            test_dataset = Subset(full_dataset, test_indices)
        except Exception as e:
            print(f"Error loading Caltech101: {e}. Check download/path.");
            return None, None, 0, (0, 0), 0
        input_channels = 1

    elif dataset_name.lower() in ['curet', 'brodatz']:
        img_h, img_w = target_img_size
        dataset_path = os.path.join(data_root, dataset_name.lower())
        if not os.path.isdir(dataset_path):
            print(f"Dataset dir not found: {dataset_path}. Organize in ImageFolder format.");
            return None, None, 0, (0, 0), 0
        transform_list = [transforms.Resize((img_h, img_w)), transforms.Grayscale(num_output_channels=1),
                          transforms.ToTensor(), normalize_grayscale]
        train_path = os.path.join(dataset_path, 'train');
        test_path = os.path.join(dataset_path, 'test')
        if not os.path.isdir(train_path) or not os.path.isdir(test_path):
            print(f"Train/test subdirs not found in {dataset_path}.");
            return None, None, 0, (0, 0), 0
        train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transforms.Compose(transform_list))
        test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transforms.Compose(transform_list))
        num_classes = len(train_dataset.classes)
        input_channels = 1
        if num_classes == 0: print(f"No classes found in {train_path}."); return None, None, 0, (0, 0), 0
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Add to choices if new.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, num_classes, (img_h, img_w), input_channels