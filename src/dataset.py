from itertools import islice
import os
import re

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
    def __init__(self, iterator):
        self.iterator = iterator

    def build_vocab(self):
        self._vocab = {}
        self._data = []
        for line in self.iterator:
            self._data = self._to_chars(line)
            for c in self._data:
                self._vocab.add(c)

    def _to_chars(self, line):
        res = ['<s>']
        res.extend(line)
        res.append('</s>')
        return res

    def __getitem__(self, index):
        chars = self._data[index]
        return chars[:-1], chars[1:]

    def __len__(self):
        return len(self._data)
