from collections import defaultdict
from itertools import islice
import os
import re

import torch
from torch.utils.data import Dataset


class KompasTempo:
    def __init__(self, data_dir=os.path.join(os.getenv('DATA_DIR'), 'input'),
                 which='train', max_sentences=None, remove_duplicate_spaces=False,
                 remove_empty_lines=True):
        self.data_dir = data_dir
        self.which = which
        self.max_sentences = max_sentences
        self.remove_duplicate_spaces = remove_duplicate_spaces
        self.remove_empty_lines = remove_empty_lines

        self._dataset_file = os.path.join(self.data_dir, f'{which}.txt')
        self._prog = re.compile(r'  +')  # two or more spaces

    def _get_iterator(self):
        with open(self._dataset_file) as f:
            for line in f:
                line = line.strip()
                if self.remove_empty_lines and not line:
                    continue  # skip empty line
                if self.remove_duplicate_spaces:
                    line = self._prog.sub(' ', line)
                yield line

    def __iter__(self):
        return islice(self._get_iterator(), self.max_sentences)


class KompasTempoDataset(Dataset):
    def __init__(self, iterator, min_count=5):
        self.iterator = iterator
        self.min_count = min_count

        self._build_vocab()
        self._convert_data_to_ids()
        self._convert_data_to_onehot_encoding()

    def _build_vocab(self):
        self.char2id, self.id2char, self._freq = {}, {}, defaultdict(int)
        charss = []
        for line in self.iterator:
            charss.append(self._to_chars(line))
            for c in charss[-1]:
                self._freq[c] += 1

        self._data = []
        for chars in charss:
            # Set rare chars (fewer than `min_count` occurrences) to <UNK>
            self._data.append(['<UNK>' if self._freq[c] < self.min_count else c
                               for c in chars])
            # Map chars to ids and vice versa
            for c in self._data[-1]:
                if c not in self.char2id:
                    cnt = len(self.char2id)
                    self.char2id[c] = cnt
                    self.id2char[cnt] = c

    def _to_chars(self, line):
        res = ['<s>']
        res.extend(line)
        res.append('</s>')
        return res

    def _convert_data_to_ids(self):
        self._data = [[self.char2id[c] for c in chars] for chars in self._data]

    def _convert_data_to_onehot_encoding(self):
        self._onehot_data = [self._onehot_encode(cids) for cids in self._data]

    def _onehot_encode(self, ids):
        ids = torch.LongTensor(ids).view(-1, 1)
        onehot = torch.zeros(ids.size(0), len(self.vocab))
        return onehot.scatter_(1, ids, 1)

    @property
    def vocab(self):
        return set(self.char2id.keys())

    def __getitem__(self, index):
        return self._onehot_data[index]

    def __len__(self):
        return len(self._onehot_data)


def collate_batch(batch):
    # Sort descending by sequence lengths
    batch = sorted(batch, key=lambda b: b.size(0), reverse=True)
    seq_lens = [b.size(0) - 1 for b in batch]
    max_seq_len = max(seq_lens)
    dim = batch[0].size(1)

    inputs, targets = [], []
    for b, seq_len in zip(batch, seq_lens):
        # Pad inputs and targets with zeros
        padded_inputs = torch.zeros(max_seq_len, dim)
        padded_targets = torch.zeros(max_seq_len, dim)
        padded_inputs[:seq_len] = b[:-1]
        padded_targets[:seq_len] = b[1:]
        inputs.append(padded_inputs)
        targets.append(padded_targets)

    return torch.stack(inputs, 1), torch.stack(targets, 1), seq_lens
