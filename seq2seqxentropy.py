import torch.nn as nn
import torch 
from encoder import Encoder
from decoder import Decoder 
from RNNCell import DecoderRNNCell, EncoderRNNCell

class Seq2SeqXentropy():
    
    def __init__(self, emb_size, hid_size, output_size):
        super(Seq2SeqXentropy, self).__init__()
        encoder = Encoder(emb_size, hid_size)
        decoder = Decoder(emb_size, hid_size, output_size)

    def forward(self, input_sentence, target_sentence):
        hidden_state = encoder(input_sentence)
        output_sentence = decoder(hidden_state, target_sentence)
        return output_sentence
