import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from model import GRUSeqClassifier
from utils import Vocab

from torch.utils.data import DataLoader
import torch.nn as nn

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset

    dataloader = DataLoader(
        dataset = dataset, batch_size = args.batch_size, collate_fn = dataset.collate_fn
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = GRUSeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    prediction_list = []
    id_list = []
    for batch in dataloader:
        text_batch, intent_batch, id_batch = batch.values()
        text_batch = torch.tensor(text_batch).to(args.device)
        prediction = model(text_batch)
        evaluation = torch.argmax(prediction, dim = 1)
        for i in range(len(text_batch)):
            id_list.append(id_batch[i])
            prediction_list.append(dataset.idx2label(evaluation[i].item()))

    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.truncate()
        f.write('id,intent\n')
        for i in range(len(dataset)):
            strline = id_list[i] + ',' + prediction_list[i] + '\n'
            f.write(strline)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
