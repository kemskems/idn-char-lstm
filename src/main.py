import argparse
from functools import partial
import math
import sys
import time

import torch

from src.dataset import KompasTempo, CharLanguageModelDataset, collate_batch
from src.models import CharLSTM
from src.train import train, evaluate, cross_entropy_loss
from src.utils import augment_parser, dump_args, load_args


def main(train_loader, valid_loader, model, criterion, optimizer, num_epochs=20, grad_clip=5.,
         log_interval=100, eval_interval=2, tol=1.0e-4, save_to=None):
    print('Evaluating epoch 0...')
    init_loss, init_ppl = evaluate(train_loader, model, criterion)
    print(f'Epoch 0: loss={init_loss:.4f} ppl={init_ppl:.4f}')

    print(f'Training on {len(train_loader)} batches...')
    best_ppl = math.inf
    start_time = time.time()
    for e in range(num_epochs):
        train(train_loader, model, criterion, optimizer, log_interval=log_interval,
              epoch=e+1, grad_clip=grad_clip)
        if (e + 1) % eval_interval == 0:
            print('Evaluating on validation data...')
            val_loss, val_ppl = evaluate(valid_loader, model, criterion)
            print(f'val_loss={val_loss:.4f} val_ppl={val_ppl:.4f}')
            if val_ppl < best_ppl - tol:
                print(f'Found new best ppl (last was {best_ppl:.4f})')
                best_ppl = val_ppl
                if save_to is not None:
                    torch.save(model.state_dict(), save_to)
                    print(f'Model parameters are saved to {save_to}')
    end_time = time.time()
    print(f'Done training in {end_time-start_time:.2f}s')

    return best_ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train character language model with LSTM')
    parser.add_argument('--train-max-sents', type=int, help='max sentences for training')
    parser.add_argument('--valid-max-sents', type=int, help='max sentences for validation')
    parser.add_argument('--min-count', type=int, default=5,
                        help='threshold count for rare words')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--hidden-size', type=int, default=50,
                        help='number of hidden units in the model')
    parser.add_argument('--num-layers', type=int, default=1, help='number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--grad-clip', type=float, default=5.,
                        help='gradients will be clipped at this value')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='log will be printed every this interval of batch')
    parser.add_argument('--eval-interval', type=int, default=2,
                        help='validation loss will be computed every this interval of epoch')
    parser.add_argument('--tol', type=float, default=1.0e-4, help='floating point tolerance')
    parser.add_argument('--save-to', help='path to which the model state will be saved')
    parser.add_argument('--cuda', action='store_true', help='use GPU if available')
    parser.add_argument('--load-from', help='path to model state file to be loaded')
    augment_parser(parser)
    args = parser.parse_args()

    dump_args(args, excludes=['num_epochs', 'log_interval', 'eval_interval', 'save_to',
                              'cuda', 'load_from'])
    load_args(args)

    print('Loading dataset...')
    start_time = time.time()
    train_source = KompasTempo(which='train', max_sentences=args.train_max_sents)
    valid_source = KompasTempo(which='valid', max_sentences=args.valid_max_sents)
    train_dataset = CharLanguageModelDataset(train_source, min_count=args.min_count)
    valid_dataset = CharLanguageModelDataset(valid_source, char2id=(train_dataset.char2id,
                                                                    train_dataset.id2char))
    vocab_size = len(train_dataset.vocab)
    print(f'Done loading dataset in {time.time()-start_time:.2f}s')
    print(f'Vocab size: {vocab_size}')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=partial(collate_batch, vocab_size))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        collate_fn=partial(collate_batch, vocab_size))
    model = CharLSTM(vocab_size, args.hidden_size, num_layers=args.num_layers,
                     dropout=args.dropout)
    if args.load_from is not None:
        model.load_state_dict(torch.load(args.load_from))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.cuda and torch.cuda.is_available():
        print('CUDA is enabled and available, GPU will be used', file=sys.stderr)
        torch.backends.cudnn.benchmark = True
        train_dataset.cuda()
        valid_dataset.cuda()
        model.cuda()

    try:
        best_ppl = main(train_loader, valid_loader, model, cross_entropy_loss, optimizer,
                        num_epochs=args.num_epochs, grad_clip=args.grad_clip,
                        log_interval=args.log_interval, eval_interval=args.eval_interval,
                        tol=args.tol, save_to=args.save_to)
    except KeyboardInterrupt:
        print('Training is stopped early. Exiting...')
    else:
        print(f'Best ppl (validation) is {best_ppl:.4f}')
