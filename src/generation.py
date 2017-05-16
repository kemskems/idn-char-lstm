import argparse

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from src.dataset import KompasTempo, CharLanguageModelDataset
from src.models import CharLSTM


def generate(model, dataset, prime_text=None, max_length=200):
    if prime_text is None:
        prime_text = [dataset.START_TOKEN]

    if model.vocab_size != len(dataset.vocab):
        raise ValueError(
            f"Model's vocab size does not equal dataset's"
            " ({model.vocab_size} != {len(dataset.vocab_size)})")
    for ch in prime_text:
        if ch not in dataset.vocab:
            raise ValueError(f"Character '{ch}' is not in the vocabulary")

    char2id = dataset.char2id
    id2char = dataset.id2char
    vocab_size = len(char2id)
    weight = next(model.parameters()).data
    res = []

    # Priming the model
    ptids = [char2id[ch] for ch in prime_text]
    onehot = weight.new(len(prime_text), vocab_size).zero_()
    onehot.scatter_(1, weight.new(ptids).long().view(-1, 1), 1)
    inputs = Variable(onehot.unsqueeze(1), volatile=True)
    states = model.init_states(1)
    outputs, states = model(inputs, states)
    probs = F.softmax(outputs[-1]).data.squeeze()
    next_ch = id2char[torch.multinomial(probs, 1)[0]]
    res.append(next_ch)

    onehot = weight.new(1, vocab_size)
    while len(res) < max_length and res[-1] != dataset.END_TOKEN:
        onehot = onehot.zero_()
        onehot.scatter_(1, weight.new(1, 1).long().fill_(char2id[res[-1]]), 1)
        inputs = Variable(onehot.unsqueeze(1), volatile=True)
        outputs, states = model(inputs, states)
        probs = F.softmax(outputs.view(1, -1)).data.squeeze()
        next_ch = id2char[torch.multinomial(probs, 1)[0]]
        res.append(next_ch)
    if res[-1] == dataset.END_TOKEN:
        del res[-1]

    res = [ch for ch in res if ch not in [dataset.UNK_TOKEN, dataset.START_TOKEN]]
    return prime_text + res


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
