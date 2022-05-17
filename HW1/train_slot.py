import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqTagDataset
from utils import Vocab

# Self
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
import torch.optim as optim
from model import TagClassifier
from model import GRUTagClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
torch.manual_seed(0)

def test_accuracy(model, testdataloader):
    acc = 0
    for batch in testdataloader:
        token_batch, lens, tag_batch, id_batch = batch.values()
        current_batch_size = len(token_batch)
        token_batch = torch.tensor(token_batch).to(args.device)
        prediction = model(token_batch)
        evaluation = torch.argmax(prediction, dim = 2)
        for i in range(current_batch_size):
            for j in range(lens[i]):
                if evaluation[i][j] != tag_batch[i][j]:
                    break
            else:
                acc += 1
    return acc

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}

    if args.add_eval:
        data["train"].extend(data["eval"])

    datasets: Dict[str, SeqTagDataset] = {
        split: SeqTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_len = len(datasets["train"])
    eval_len = len(datasets["eval"])
    dataloader: Dict[str, DataLoader] = {
        split: DataLoader(
            dataset = datasets[split], batch_size = args.batch_size, shuffle = True, 
            collate_fn = datasets[split].collate_fn
        )
        for split in SPLITS
    }
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = TagClassifier(
        embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers, 
        dropout=args.dropout, bidirectional=args.bidirectional, num_class=datasets["train"].num_classes
    ).to(args.device)

    # TODO: init optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = args.lr)
    else:
        print("optimizer not defined")
        exit(1)
    loss_fn = nn.CrossEntropyLoss()

    acc, max_acc = 0, 0
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        running_loss = 0
        for batch in dataloader["train"]:
            token_batch, lens, tag_batch, id_batch = batch.values()
            current_batch_size = len(token_batch)
            token_batch = torch.tensor(token_batch).to(args.device)
            # tag_batch = torch.tensor(tag_batch).to(args.device)
            prediction = model(token_batch)
            loss = torch.tensor([0], dtype=torch.float).to(args.device)
            for i in range(current_batch_size):
                tag_list = torch.tensor(tag_batch[i]).to(args.device)
                loss += loss_fn(prediction[i][:lens[i]], tag_list) / lens[i]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # TODO: Evaluation loop - calculate accuracy and save model weights
        if not args.add_eval:
            acc = test_accuracy(model, dataloader["eval"])
            if acc > max_acc:
                max_acc = acc
                if acc / eval_len > 0.7:
                    print(f"[epoch {epoch}] Accuracy: {round(acc / eval_len, 3)} ({acc}/{eval_len})")
                    print("save model at",  args.ckpt_dir / "model.pt")
                    torch.save(model.state_dict(), args.ckpt_dir / "model.pt")
            if epoch % 20 == 0:
                print(f'[epoch {epoch}] loss: {round(running_loss, 3)}', 
                      f'current max accuracy:{round(max_acc / eval_len, 3)} ({max_acc}/{eval_len})')
        elif epoch % 20 == 0:
            print(f'[epoch {epoch}] loss: {round(running_loss, 3)}')

    if args.add_eval:
        acc = test_accuracy(model, dataloader["train"])
        print(f'Accuracy: {round(acc / train_len, 3)} ({acc}/{train_len})')
        print("save model at",  args.ckpt_dir / "model.pt")
        torch.save(model.state_dict(), args.ckpt_dir / "model.pt")

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=28)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    parser.add_argument("--optimizer", type=str, default="SGD")

    parser.add_argument("--add_eval", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
