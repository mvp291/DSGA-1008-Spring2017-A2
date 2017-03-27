import os
import torch
import pickle
from collections import Counter

class Dictionary(object):
    def __init__(self, vocabulary_size=10000):
        self.word_freq = Counter()
        self.word2idx = {}
        self.idx2word = []
        self.vocabulary_size = vocabulary_size

    def add_word(self, word):
        if word != '<unk>':
            self.word_freq[word] += 1
        return self.word_freq[word]

    def update_match(self):
        self.idx2word = [i[0] for i in self.word_freq.most_common(self.vocabulary_size - 1)]
        # Add word for all unkown words, idx for unknown is always 0
        self.idx2word = ['<unk>'] + self.idx2word
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocabulary_size=None, dict_path=None):
        if not dict_path:
            self.dictionary = Dictionary()
        else:
            self.load_dictionary(dict_path)

        if vocabulary_size:
            # If vocabulary size is given, update word idx match according to
            # given vocabulary size
            self.dictionary.vocabulary_size = vocabulary_size
            self.dictionary.update_match()

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary if training file and not dictionary is provided
        reset_dict = 'train' in path and len(self.dictionary.word_freq) == 0

        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)

                if reset_dict:
                    for word in words:
                        self.dictionary.add_word(word)

        if reset_dict:
            # Update indexed assigmed to each word according to vocabulary size
            # Use most common self.vocabulary_size words
            self.dictionary.update_match()

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    # if word is unkown, assign index 0 (for unkown words)
                    ids[token] = self.dictionary.word2idx.get(word, 0)
                    token += 1

        return ids

    def save_dictionary(self, path):
        """Save train dictionary to path"""
        f = open(path, 'w')
        pickle.dump(self.dictionary, f)
        f.close()

    def load_dictionary(self, path):
        """Save train dictionary to path"""
        f = open(path, 'r')
        self.dictionary = pickle.load(f)
        f.close()

