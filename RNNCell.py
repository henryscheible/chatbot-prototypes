import torch.nn as nn
import torch

class EncoderRNNCell(nn.Module):
    
    def __init__(self, emb_size, hid_size, output_size):
        super(EncoderRNNCell, self).__init__()
        w = nn.Parameter(torch.rand(emb_size, hid_size))
        u = nn.Parameter(torch.rand(hid_size, hid_size))

    def forward(self, x, h):
        h_f = nn.functional.sigmoid(w*x + u*h)
        return h_fv = nn.Parameter(torch.rand(hid_size, output_size))