import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
import flwr as fl
from typing import List
from pathlib import Path
import tempfile
import shutil
import os


def load_weights(filename: str):
    """Load Parameters object from a local file."""
    # Load the .npz file
    with np.load(filename) as data:
        # Convert the loaded arrays back to a list of NDArrays
        ndarrays = [data[arr] for arr in data.files]

    # Convert NDArrays back to Parameters
    return ndarrays


def save_weights(weights, filename: str):

    ndarrays = weights

    # Save the NDArrays to a .npz file
    np.savez(filename, *ndarrays)


def get_weights(model: torch.nn.ModuleList) -> List[fl.common.NDArrays]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class WeightManager:
    def __init__(self, base_path):
        """Initialize weight manager with base path for storage"""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._temp_dir = None

    @property
    def temp_dir(self):
        """Lazy initialization of temporary directory"""
        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp()
        return self._temp_dir

    def get_weight_path(self, cid, round_num):
        """Generate consistent path for weight files"""
        return self.base_path / f"model_params_cid_{cid}_round_{round_num}.pt"

    def save_weights(self, model, cid, round_num, use_temp=True):
        """Save model weights efficiently

        Args:
            model: PyTorch model
            cid: Client ID
            round_num: Training round number
            use_temp: Whether to use temporary file for atomic writes
        """
        final_path = self.get_weight_path(cid, round_num)

        if use_temp:
            # Use temporary file for atomic writes
            temp_path = Path(self.temp_dir) / f"temp_{cid}_{round_num}.pt"
            torch.save(model.state_dict(), temp_path)
            shutil.move(str(temp_path), str(final_path))
        else:
            torch.save(model.state_dict(), final_path)

    def load_weights(self, cid, round_num):
        """Load model weights efficiently

        Args:
            cid: Client ID
            round_num: Training round number

        Returns:
            model_state: Dictionary containing model state
        """
        weight_path = self.get_weight_path(cid, round_num)

        if not weight_path.exists():
            raise FileNotFoundError(
                f"No weights found for client {cid} at round {round_num}"
            )

        return torch.load(weight_path, map_location="cpu")

    def cleanup(self):
        """Clean up temporary directory"""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None

    def __del__(self):
        """Ensure cleanup on deletion"""
        self.cleanup()


def non_iid_train_iid_test(
    class_partitions=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
    data_type="mnist",
    outlier_count=1000,
):
    torch.manual_seed(len(class_partitions))
    # Load the MNIST training dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    if data_type == "mnist":
        train_dataset = MNIST(
            root="./data/", train=True, download=True, transform=transform
        )
    elif data_type == "fmnist":
        train_dataset = FashionMNIST(
            root="./data/", train=True, download=True, transform=transform
        )

    # Define class pairs for each partition
    class_partitions = class_partitions

    # Create a list to store datasets for each partition
    partition_datasets_train = []

    # Iterate over class pairs and create a dataset for each partition
    for class_pair in class_partitions:
        class_filter = lambda label: label in class_pair
        filtered_indices = [
            i for i, (_, label) in enumerate(train_dataset) if class_filter(label)
        ]

        # Use Subset to create a dataset with filtered indices
        partition_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)
        partition_datasets_train.append(partition_dataset)

    if outlier_count > 0:
        # Add 100 samples with all pixel values set to 1 to the first train partition
        all_ones_samples = torch.ones((outlier_count, 1, 28, 28))
        all_ones_labels = torch.full(
            (outlier_count,), 10, dtype=torch.long
        )  # Label 10 for these samples
        all_ones_dataset = CustomDataset(all_ones_samples, all_ones_labels)
        partition_datasets_train[0] = ConcatDataset(
            [partition_datasets_train[0], all_ones_dataset]
        )

    # Load the test dataset
    if data_type == "mnist":
        test_dataset = MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif data_type == "fmnist":
        test_dataset = FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    class_partitions_test = [
        element for sublist in class_partitions for element in sublist
    ]

    partition_datasets_test_ = []

    # Iterate over class pairs and create a dataset for each partition
    filtered_data_size = 0
    for class_pair in class_partitions_test:
        class_filter = class_pair
        filtered_indices = [
            i for i, (_, label) in enumerate(test_dataset) if class_filter == label
        ]

        # Use Subset to create a dataset with filtered indices
        partition_datasets_test_.append(Subset(test_dataset, filtered_indices))
        filtered_data_size += len(filtered_indices)

    # Specify the size of each partition
    partition_sizes = [filtered_data_size // len(class_partitions)] * (
        len(class_partitions) - 1
    ) + [
        filtered_data_size
        - (filtered_data_size // len(class_partitions)) * (len(class_partitions) - 1)
    ]
    concatinated_testset = ConcatDataset(partition_datasets_test_)

    # Use random_split to create 5 datasets with random samples
    partition_datasets_test = torch.utils.data.random_split(
        concatinated_testset, partition_sizes
    )
    return partition_datasets_train, partition_datasets_test


def non_iid_train_iid_test_less_samples(
    data_type="mnist",
    class_partitions=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
    samples_per_class=1000,
    outlier_count=1,
):
    """
    Creates non-IID training datasets and IID test datasets with specified number of samples per class.

    Args:
        data_type (str): Type of dataset ("mnist" or "fmnist")
        class_partitions (list): List of tuples containing class pairs
        samples_per_class (int): Number of samples to use per class
        outlier_count (int): Number of outlier samples to add to first partition

    Returns:
        tuple: (train_datasets, test_datasets)
    """
    # Set random seed for reproducibility
    torch.manual_seed(len(class_partitions))

    # Define transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Load datasets
    dataset_class = MNIST if data_type == "mnist" else FashionMNIST
    train_dataset = dataset_class(
        root="./data/", train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root="./data", train=False, download=True, transform=transform
    )

    # Initialize class indices dictionary
    class_indices_train = {i: [] for i in range(10)}

    # Collect first samples_per_class indices for each class
    for idx, (_, label) in enumerate(train_dataset):
        label_val = label
        if len(class_indices_train[label_val]) < samples_per_class:
            class_indices_train[label_val].append(idx)

    # Create training partitions
    partition_train_datasets = []
    for class_pair in class_partitions:
        # Combine indices for the class pair
        pair_indices = (
            class_indices_train[class_pair[0]] + class_indices_train[class_pair[1]]
        )
        pair_dataset = Subset(train_dataset, pair_indices)
        partition_train_datasets.append(pair_dataset)

    # Add outliers to first partition if specified
    if outlier_count > 0:
        if data_type == "mnist":
            outlier_dataset_class = FashionMNIST
        else:
            outlier_dataset_class = MNIST

        outlier_dataset = outlier_dataset_class(
            root="./data/", train=True, download=True, transform=transform
        )

        outlier_indices = torch.randperm(len(outlier_dataset))[:outlier_count].tolist()
        outlier_samples = torch.stack([outlier_dataset[i][0] for i in outlier_indices])
        outlier_labels = torch.full((outlier_count,), 10, dtype=torch.long)
        outlier_dataset = CustomDataset(outlier_samples, outlier_labels)
        partition_train_datasets[0] = ConcatDataset(
            [partition_train_datasets[0], outlier_dataset]
        )

    # Prepare test datasets
    # Flatten class partitions for test set
    test_classes = [cls for pair in class_partitions for cls in pair]
    test_datasets = []
    total_test_samples = 0

    # Create test datasets for each class
    for class_label in test_classes:
        indices = [
            i for i, (_, label) in enumerate(test_dataset) if label == class_label
        ]
        test_datasets.append(Subset(test_dataset, indices))
        total_test_samples += len(indices)

    # Calculate partition sizes for test set
    base_size = total_test_samples // len(class_partitions)
    partition_sizes = [base_size] * (len(class_partitions) - 1)
    partition_sizes.append(total_test_samples - base_size * (len(class_partitions) - 1))

    # Combine and split test datasets
    combined_test = ConcatDataset(test_datasets)
    partition_datasets_test = torch.utils.data.random_split(
        combined_test, partition_sizes
    )

    return partition_train_datasets, partition_datasets_test,outlier_dataset


def alignment_dataloader(
    samples_per_class=100,
    batch_size=8,
    shuffle=False,
    only_data=False,
    num_classes=10,
    data_type="mnist",
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    # Load the MNIST test dataset
    if data_type == "mnist":
        test_data = MNIST(
            root="./data/",
            train=False,
            download=True,
            transform=transform,
        )
    elif data_type == "fmnist":
        test_data = FashionMNIST(
            root="./data/",
            train=False,
            download=True,
            transform=transform,
        )

    # Create an alignment dataset with 20 samples for each class
    alignment_datasets = []

    for class_label in range(num_classes):
        class_indices = [
            i for i, (img, label) in enumerate(test_data) if label == class_label
        ]
        # print(f"{class_label}:{len(class_indices)}")
        selected_indices = class_indices[:samples_per_class]
        alignment_dataset = Subset(test_data, selected_indices)
        alignment_datasets.append(alignment_dataset)

    # Concatenate the alignment datasets into one
    alignment_dataset = ConcatDataset(alignment_datasets)

    if only_data:
        return alignment_dataset

    # Create a DataLoader for the alignment dataset
    alignment_loader = DataLoader(
        alignment_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return alignment_loader


def set_params_return(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, int(label)  # Convert label to int to match MNIST format

    def __len__(self):
        return len(self.images)


def create_subset_data(dataset, num_samples):
    subset_indices = torch.randperm(len(dataset))[:num_samples].tolist()
    sampled_subset = Subset(dataset, subset_indices)
    return sampled_subset
