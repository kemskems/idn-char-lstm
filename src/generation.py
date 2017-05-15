import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from src.dataset import KompasTempo, CharLanguageModelDataset
from src.models import CharLSTM


def generate(model, dataset, start_ch=None, max_length=200):
    if model.vocab_size != len(dataset.vocab):
        raise ValueError(
            f"Model's vocab size does not equal dataset's"
            " ({model.vocab_size} != {len(dataset.vocab_size)})")
    if start_ch is not None and start_ch not in dataset.vocab:
        raise ValueError(f"Character '{start_ch}' is not in the vocabulary")

    if start_ch is None:
        start_ch = dataset.START_TOKEN

    char2id = dataset.char2id
    id2char = dataset.id2char
    vocab_size = len(char2id)
    weight = next(model.parameters()).data
    onehot = weight.new(1, vocab_size).zero_()
    res = [start_ch]
    states = model.init_states(1)
    while len(res) <= max_length and res[-1] != dataset.END_TOKEN:
        onehot = onehot.zero_()
        onehot.scatter_(1, weight.new(1, 1).long().fill_(char2id[res[-1]]), 1)
        inputs = Variable(onehot.unsqueeze(1), volatile=True)
        outputs, states = model(inputs, states)
        probs = F.softmax(outputs.view(1, -1)).data.squeeze()
        next_ch = id2char[torch.multinomial(probs, 1)[0]]
        res.append(next_ch)

    res = ''.join([ch for ch in res[1:-1]
                   if ch not in [dataset.UNK_TOKEN, dataset.START_TOKEN]])
    return f'{dataset.START_TOKEN}{res}{dataset.END_TOKEN}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate text from a model trained on training corpus')
    parser.add_argument('--load-from', required=True, help='path to saved model state')
    parser.add_argument('--min-count', type=int, default=5,
                        help='threshold count for rare words')
    parser.add_argument('--hidden-size', type=int, default=50,
                        help='number of hidden units in the model')
    parser.add_argument('--num-layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--start-char', help='starting character')
    parser.add_argument('--max-length', type=int, default=200,
                        help='max length of the generated text')
    args = parser.parse_args()

    source = KompasTempo(which='train')
    dataset = CharLanguageModelDataset(source, args.min_count)
    model = CharLSTM(len(dataset.vocab), args.hidden_size, num_layers=args.num_layers)
    model.load_state_dict(torch.load(args.load_from))
    print(generate(model, dataset, start_ch=args.start_char, max_length=args.max_length))
