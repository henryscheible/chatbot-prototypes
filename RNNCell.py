import torch.nn as nn
import torch

class EncoderRNNCell(nn.Module):
    
    def __init__(self, emb_size, hid_size):
        super(EncoderRNNCell, self).__init__()
        w = nn.Parameter(torch.rand(emb_size, hid_size))
        u = nn.Parameter(torch.rand(hid_size, hid_size))

    def forward(self, x, h):
        h_f = nn.functional.sigmoid(w*x + u*h)
        return h_f

class DecoderRNNCell(nn.Module):

    def __init__(self, emb_size, hid_size, output_size):
        super(DecoderRNNCell, self).__init__()
        w = nn.Parameter(torch.rand(emb_size, hid_size))
        u = nn.Parameter(torch.rand(hid_size, hid_size))
        v = nn.Parameter(torch.rand(hid_size, output_size))

    def forward(self, x, h):
        h_f = nn.functional.sigmoid(w*x+u*h)
        y = nn.functional.softmax(nn.functional.sigmoid(v*h_f))
        return h_f, y 