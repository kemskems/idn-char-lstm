import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=1, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bias=True,
                            dropout=dropout, bidirectional=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, init_states, seq_lens=None):
        if seq_lens is None:
            outputs, states = self.lstm(inputs, init_states)
        else:
            packed = pack_padded_sequence(inputs, seq_lens)
            outputs, states = self.lstm(packed, init_states)
            outputs = pad_packed_sequence(outputs)

        unsoftmax = self.linear(outputs.view(-1, outputs.size(2)))
        return unsoftmax.view(-1, outputs.size(1), unsoftmax.size(1)), states

    def init_weights(self, init_range=0.1):
        weight_attrs = [attr for attr in dir(self.lstm) if attr.startswith('weight_')]
        bias_attrs = [attr for attr in dir(self.lstm) if attr.startswith('bias_')]
        for attr in weight_attrs:
            getattr(self.lstm, attr).data.uniform_(-init_range, init_range)
        for attr in bias_attrs:
            getattr(self.lstm, attr).data.zero_()
        init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.zero_()

    def init_states(self, batch_size):
        # To ensure that the states have the same type as the weights, the states
        # should be created from the weights tensor, NOT from `torch.zeros`
        weights = next(self.parameters()).data
        h0 = Variable(weights.new(self.num_layers, batch_size, self.hidden_size).zero_())
        c0 = Variable(weights.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return (h0, c0)
