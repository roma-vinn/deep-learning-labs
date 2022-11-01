from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader


BATCH_SIZE = 64
DATASET_ROOT_DIR = "./EMNIST_data"
SPLIT_TYPE = "bymerge"


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


# # create dataloader
# train_loader = DataLoader(emnist_train, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(emnist_test, batch_size=BATCH_SIZE, shuffle=True)

