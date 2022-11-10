from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader


DATASET_ROOT_DIR = "demos/EMNIST_data"
SPLIT_TYPE = "balanced"


TRANSFORMATIONS = transforms.Compose(
    [
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.2), scale=(0.5, 1)),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.75),
        transforms.ToTensor(),
    ]
)


EMNIST_TRAIN, EMNIST_TEST = (
    EMNIST(root=DATASET_ROOT_DIR, split=SPLIT_TYPE, train=True, download=True, transform=transforms.ToTensor()),
    EMNIST(root=DATASET_ROOT_DIR, split=SPLIT_TYPE, train=False, download=True, transform=transforms.ToTensor()),
)


EMNIST_AUG_TRAIN, EMNIST_AUG_TEST = (
    EMNIST(root=DATASET_ROOT_DIR, split=SPLIT_TYPE, train=True, download=True, transform=TRANSFORMATIONS),
    EMNIST(root=DATASET_ROOT_DIR, split=SPLIT_TYPE, train=False, download=True, transform=TRANSFORMATIONS),
)


def get_dataloader(split: str, augment=True, batch_size=64, shuffle=True) -> DataLoader:
    assert split in ['train', 'test'], 'split should be either "train" or "test"'
    loader = None
    if split == 'train':
        if augment:
            loader = DataLoader(EMNIST_AUG_TRAIN, batch_size=batch_size, shuffle=shuffle)
        else:
            loader = DataLoader(EMNIST_TRAIN, batch_size=batch_size, shuffle=shuffle)
    else:
        if augment:
            loader = DataLoader(EMNIST_AUG_TEST, batch_size=batch_size, shuffle=shuffle)
        else:
            loader = DataLoader(EMNIST_TEST, batch_size=batch_size, shuffle=shuffle)
    return loader
