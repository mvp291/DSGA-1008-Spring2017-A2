import numpy as np
import argparse
import time
import math
import urllib2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import pickle as pkl
import data

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--save-dict', type=str, default='word_dict.pkl',
                    help='path to save the training word dictonary')
parser.add_argument('--load-dict', type=str, default=None,
                    help='path to load the training word dictonary from')
parser.add_argument('--vocab-size', type=int, default=10000,
                    help='Number of most frequent words to keep in dictionary')
parser.add_argument('--randomize-input', action='store_true',
                     help='Randomize dataset for each epoch')
parser.add_argument('--embedding-noise', action='store_true',
                     help='Use noise in the output of the embedding matrix')
parser.add_argument('--retrain', action='store_true',
                    help='Retrain the model instead of downloading it.')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)


##############################
# Initialization utils
##############################

# W(weight_ih) se inicializa con norm, U(weight_hh) se inicializa con orto
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u


def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=True, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

        for weight_attr in self.rnn.state_dict().keys():
            if 'bias' in weight_attr:
                getattr(self.rnn, weight_attr).data.fill_(0)
            else:
                getattr(self.rnn, weight_attr).data.uniform_(-initrange, initrange)

        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))

        emb_noise = emb.data * 0.
        emb_noise.normal_(mean=0, std=0.1)
        emb_noise = Variable(emb_noise)
        if args.cuda:
            emb_noise = emb_noise.cuda()
        if args.embedding_noise:
            emb = emb + emb_noise

        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1),
                                           output.size(2)))
        return decoded.view(output.size(0), output.size(1),
                            decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


###############################################################################
# Load data
###############################################################################

print('Loading data')
if args.load_dict:
    corpus = data.Corpus(args.data, args.vocab_size, "{}/{}".format(args.data, args.load_dict))
else:
    corpus = data.Corpus(args.data, args.vocab_size)

if args.save_dict:
    corpus.save_dictionary("{}/{}".format(args.data, args.save_dict))

print('Using vocabulary size of {}'.format(len(corpus.dictionary)))

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def evaluate(data_source):
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)

    iteration_order = range(0, train_data.size(0) - 1, args.bptt)
    if args.randomize_input:
        iteration_order = np.random.permutation(iteration_order)

    for batch, i in enumerate(iteration_order):
        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:2.6f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    return cur_loss, math.exp(cur_loss)



###############################################################################
# Build the model
###############################################################################



ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()

if args.retrain:
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    if args.cuda:
        model = model.cuda()

    # Loop over epochs.
    lr = args.lr
    prev_val_loss = None
    optimizer = optim.SGD(model.parameters(), lr)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_perp = train()
        val_loss = evaluate(val_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Anneal the learning rate.
        if prev_val_loss and val_loss > prev_val_loss:
            lr /= 4.
            optimizer = optim.SGD(model.parameters(), lr)
        prev_val_loss = val_loss

else:
    print('Downloading pretrained model...')
    response = urllib2.urlopen('https://s3.amazonaws.com/emrbucket-fnd212/DL_HW/LSTM_1500.pt')
    M = response.read()
    f = open('./best_network', 'wb')
    f.write(M)
    f.close()

    with open('./best_network', 'rb') as f:
        model = torch.load(f)
    if args.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

# Run on test data and save the model.
print('Running evaluation on test set')
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
if args.save != '':
    with open(args.save, 'wb') as f:
        torch.save(model, f)
