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

AUGMENTATIONS_EASY = transforms.Compose(
    [
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.2), scale=(0.5, 1)),
        transforms.ToTensor(),
    ]
)


def get_datasets(
        root: str = DATASET_ROOT_DIR,
        split: str = SPLIT_TYPE,
        train: bool = True,
        augment: str = None,
) -> Dataset:
    """ Helper function to generate EMNIST Dataset instance with given parameters

    :param root: root directory of dataset
    :param split: dataset split, one of "byclass", "bymerge", "balanced", "letters", "digits" and "mnist".
    :param train: if True, returns training samples, else – testing
    :param augment: type of transformations, one of "easy", "complex" or None (default)
    :return: Dataset with given parameters
    """
    if not augment:
        dataset = EMNIST(root=root, split=split, train=train, download=True, transform=transforms.ToTensor())
    elif augment == "easy":
        dataset = EMNIST(root=root, split=split, train=train, download=True, transform=AUGMENTATIONS_EASY)
    elif augment == "complex":
        dataset = EMNIST(root=root, split=split, train=train, download=True, transform=AUGMENTATIONS)
    else:
        raise NameError("Unknown augmentation type")

    return dataset


def get_dataloader(
        root: str = DATASET_ROOT_DIR,
        split: str = SPLIT_TYPE,
        train: bool = True,
        augment: str = None,
        batch_size: int = 64,
        shuffle: bool = True,
) -> DataLoader:
    """ Helper function to generate a DataLoader out of EMNIST Dataset instance with given parameters

    :param root: root directory of dataset
    :param split: dataset split, one of "byclass", "bymerge", "balanced", "letters", "digits" and "mnist".
    :param train: if True, returns training samples, else – testing
    :param augment: type of transformations, one of "easy", "complex" or None (default)
    :param batch_size: int, batch size
    :param shuffle: if True, picks samples from dataset randomly
    :return: DataLoader with given parameters
    """

    dataset = get_datasets(root=root, split=split, train=train, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader
