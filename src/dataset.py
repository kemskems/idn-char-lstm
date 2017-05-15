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


class CharLanguageModelDataset(Dataset):
    def __init__(self, iterator, min_count=5, char2id=None):
        self.iterator = iterator
        self.min_count = min_count
        if char2id is None:
            self.char2id = None
        elif isinstance(char2id, dict):
            self.char2id = char2id
            self.id2char = {v: k for k, v in char2id.items()}
        else:
            self.char2id, self.id2char = char2id

        self._tokenize_chars()
        if self.char2id is None:
            self._build_vocab()
        self._convert_data_to_ids()

    def _tokenize_chars(self):
        self._data = []
        for line in self.iterator:
            self._data.append(self._to_chars(line))

    def _build_vocab(self):
        self.char2id, self.id2char, self._freq = {}, {}, defaultdict(int)

        for chars in self._data:
            for ch in chars:
                self._freq[ch] += 1

        # Always have UNK token
        self.char2id['<UNK>'] = 0
        self.id2char[0] = '<UNK>'

        for ch, cnt in self._freq.items():
            ch = '<UNK>' if cnt < self.min_count else ch
            if ch not in self.char2id:
                size = len(self.char2id)
                self.char2id[ch] = size
                self.id2char[size] = ch

    def _convert_data_to_ids(self):
        unk_id = self.char2id['<UNK>']
        self._data = [
            torch.LongTensor([self.char2id.get(ch, unk_id) for ch in chars])
            for chars in self._data
        ]

    @staticmethod
    def _to_chars(line):
        res = ['<s>']
        res.extend(line)
        res.append('</s>')
        return res

    @property
    def vocab(self):
        return set(self.char2id.keys())

    def cuda(self):
        self._data = [cids.cuda() for cids in self._data]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def collate_batch(vocab_size, batch):
    # Sort descending by sequence lengths
    batch = sorted(batch, key=lambda b: b.size(0), reverse=True)
    seq_lens = [b.size(0) - 1 for b in batch]
    max_seq_len = max(seq_lens)

    inputs, targets = [], []
    for b, seq_len in zip(batch, seq_lens):
        batch_inputs, batch_targets = b[:-1], b[1:]
        # Convert inputs to onehot encoding
        onehot = b.new(seq_len, vocab_size).float().zero_()
        onehot.scatter_(1, batch_inputs.view(-1, 1), 1)
        # Pad inputs and targets
        padded_inputs = b.new(max_seq_len, vocab_size).float().fill_(-1)
        padded_targets = b.new(max_seq_len).fill_(-1)
        padded_inputs[:seq_len] = onehot
        padded_targets[:seq_len] = batch_targets
        inputs.append(padded_inputs)
        targets.append(padded_targets)

    return torch.stack(inputs, 1), torch.stack(targets, 1), seq_lens
