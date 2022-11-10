from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader, Dataset


DATASET_ROOT_DIR = "data"
SPLIT_TYPE = "balanced"


AUGMENTATIONS = transforms.Compose(
    [
        transforms.RandomAffine(degrees=30, translate=(0.1, 0.2), scale=(0.5, 1)),
        transforms.RandomPerspective(distortion_scale=0.6, p=0.75),
        transforms.ToTensor(),
    ]
)


def get_datasets(
        root: str = DATASET_ROOT_DIR,
        augment: bool = False,
        split: str = SPLIT_TYPE,
        train: bool = True,
        augmentations=AUGMENTATIONS
) -> Dataset:
    """ Helper function to generate EMNIST Dataset instance with given parameters

    :param root: root directory of dataset
    :param augment: if True, uses augmentations
    :param split: dataset split, one of "byclass", "bymerge", "balanced", "letters", "digits" and "mnist".
    :param train: if True, returns training samples, else – testing
    :param augmentations: transforms.Compose of augmentation transformations
    :return: Dataset with given parameters
    """
    if augment:
        dataset = EMNIST(root=root, split=split, train=train, download=True, transform=augmentations)
    else:
        dataset = EMNIST(root=root, split=split, train=train, download=True, transform=transforms.ToTensor())

    return dataset


def get_dataloader(
        root: str = DATASET_ROOT_DIR,
        split: str = SPLIT_TYPE,
        train: bool = True,
        augment: bool = True,
        augmentations=AUGMENTATIONS,
        batch_size: int = 64,
        shuffle: bool = True
) -> DataLoader:
    """ Helper function to generate a DataLoader out of EMNIST Dataset instance with given parameters

    :param root: root directory of dataset
    :param split: dataset split, one of "byclass", "bymerge", "balanced", "letters", "digits" and "mnist".
    :param train: if True, returns training samples, else – testing
    :param augment: if True, uses augmentations
    :param augmentations: transforms.Compose of augmentation transformations
    :param batch_size: int, batch size
    :param shuffle: if True, picks samples from dataset randomly
    :return: DataLoader with given parameters
    """
    dataset = get_datasets(root=root, augment=augment, split=split, train=train, augmentations=augmentations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
