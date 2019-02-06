import torch.nn as nn
import torch
from RNNCell import DecoderRNNCell

class Decoder(nn.Module):

    def __init__(self, emb_size, hid_size, output_size):
        super(Decoder, self).__init__()
        rnn = DecoderRNNCell(emb_size, hid_size, output_size)
        output = []

    def forward(self, hidden_state, targetsentence):
        for x in targetsentence:
            hidden_state, y = rnn(x, hidden_state)
            output.append(y)
        return output
