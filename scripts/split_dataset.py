#!/usr/bin/env python

import argparse
import glob
import os
import random


def main(corpus_dir, train=0.7, valid=0.2):
    if train < 0:
        raise ValueError('Proportion of training set should be positive')
    if valid < 0:
        raise ValueError('Proportion of validation set should be positive')
    if train + valid > 1:
        raise ValueError(
            'Sum of proportion of training and validation set should not exceed 1')

    lines = list(read_kompas(args.corpus_dir))
    lines.extend(list(read_tempo(args.corpus_dir)))
    random.shuffle(lines)

    n_train = int(train * len(lines))
    n_valid = int(valid * len(lines))

    train_set = lines[:n_train]
    valid_set = lines[n_train:n_train+n_valid]
    test_set = lines[n_train+n_valid:]

    return train_set, valid_set, test_set


def read_kompas(corpus_dir):
    for filename in glob.glob(os.path.join(corpus_dir, 'kompas', 'txt', '**', '*.txt')):
        with open(filename) as f:
            for line in f:
                yield line.strip()


def read_tempo(corpus_dir):
    for filename in glob.glob(os.path.join(corpus_dir, 'tempo', 'txt', '**', '*.txt')):
        with open(filename) as f:
            for line in f:
                yield line.strip()


def write_to_file(path, lines):
    with open(path, 'w') as f:
        print('\n'.join(lines), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset for char LSTM project')
    parser.add_argument('--corpus-dir', default=os.getenv('CORPUS_DIR'),
                        help='Path to corpus directory')
    parser.add_argument('--train', type=float, default=0.7, help='Proportion of training set')
    parser.add_argument('--valid', type=float, default=0.2,
                        help='Proportion of validation set')
    parser.add_argument('--output-dir', default=os.getcwd(), help='Path to output directory')
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    train_set, valid_set, test_set = main(args.corpus_dir, train=args.train, valid=args.valid)

    write_to_file(os.path.join(args.output_dir, 'train.txt'), train_set)
    write_to_file(os.path.join(args.output_dir, 'valid.txt'), valid_set)
    write_to_file(os.path.join(args.output_dir, 'test.txt'), test_set)
