from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import spacy
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

def remove_stopword(token_list):
    return token_list
    pruned_list = []
    for token in token_list:
        if token not in stopwords:
            pruned_list.append(token)
    return pruned_list

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        texts, intents, ids = [], [], []
        for i in range(len(samples)):
            token_list = samples[i]["text"].split()
            pruned_list = remove_stopword(token_list)
            texts.append(pruned_list)
        encode_texts = self.vocab.encode_batch(batch_tokens=texts, to_len = None)
        if "intent" in samples[0].keys():
            for i in range(len(samples)):
                intents.append(self.label_mapping[samples[i]["intent"]])
        for i in range(len(samples)):
            ids.append(samples[i]["id"])
        ret_dict = {"texts": encode_texts, "intents": intents, "ids": ids}
        return ret_dict

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        tokens, lens, tags, ids = [], [], [], []
        sample_len = len(samples)
        for i in range(sample_len):
            tokens.append(samples[i]["tokens"])
            lens.append(len(samples[i]["tokens"]))
        encode_tokens = self.vocab.encode_batch(batch_tokens=tokens, to_len = None)
        if "tags" in samples[0].keys():
            for i in range(sample_len):
                taglist = []
                for j in range(len(samples[i]["tags"])):
                    taglist.append(self.label_mapping[samples[i]["tags"][j]])
                tags.append(taglist)
        for i in range(sample_len):
            ids.append(samples[i]["id"])
        ret_dict = {"tokens": encode_tokens, "lens": lens, "tags": tags, "ids": ids}
        return ret_dict

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

