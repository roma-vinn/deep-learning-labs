from tqdm import tqdm
import argparse
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from model import InceptionModelSmall
from dataloaders import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-root', type=str, default='output/')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--augment', action='store_true')

    args = parser.parse_args()

    return args


def train_loop(dataloader, model, loss_fn, optimizer, device):
    iterator = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch_num, (X, y) in iterator:
        X = X.to(device)
        y = y.to(device)

        get_pred = model(X)
        loss = loss_fn(get_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0 and batch_num != 0:
            loss, current = loss.item(), batch_num * len(X)
            iterator.set_description(f"loss: {loss:>7f}")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, true_ = 0, 0

    with torch.no_grad():
        for X, y in tqdm(dataloader, total=len(dataloader)):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            true_ += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    true_ /= size
    print(f"Accuracy: {(100 * true_):.3f}%")


def train(args, run_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Training on device={device}')

    model = InceptionModelSmall(num_classes=47).to(device)

    train_loader = get_dataloader(train=True, augment=args.augment, batch_size=args.batch_size)
    test_loader = get_dataloader(train=False, augment=args.augment, batch_size=args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    for t in range(args.num_epochs):
        print(f"Epoch {t + 1}:")
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loop(test_loader, model, loss_fn, device)
        scheduler.step()
        print()

    model_fn = os.path.join(run_path, "model.pt")
    torch.save(model.state_dict(), model_fn)


def main():
    args = parse_args()
    print(f'Starting training with following args: {args}')

    # create output root directory (if it exists, do nothing)
    os.makedirs(args.output_root, exist_ok=True)
    runs_dir = os.path.join(args.output_root, 'runs')
    os.makedirs(runs_dir, exist_ok=True)  # if it's the first time, make `runs` dir too
    run_name = f'run_{len(os.listdir(runs_dir)) + 1}'  # run_1, run_2, ...
    run_path = os.path.join(runs_dir, run_name)
    os.makedirs(run_path)

    train(args=args, run_path=run_path)


if __name__ == '__main__':
    main()
