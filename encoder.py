import torch.nn as nn
import torch
import RNNCell

class Encoder(nn.Module):
    
    def __init__(self, emb_size, hid_size):
        super(self, nn.Module).__init__()
        rnn = EncoderRNNCell(emb_size, hid_size)
        
    def forward(self, sentence):
        hidden_state = torch.rand(hid_size)
        for word in sentence:
            hidden_state = rnn(word, hidden_state)
        return hidden_state
