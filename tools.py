import torch.nn as nn
import torch 
from encoder import Encoder
from decoder import Decoder 
from RNNCell import DecoderRNNCell, EncoderRNNCell
from os import path
from os import open

def load_movie_data(data_path):
    lines = load_movie_lines(path.join(data_path, "movie_lines.txt"))
    conversations = load_movie_conversations(path.join(data_path, "movie_conversations.txt"))
    dataset = []
    for conversation in conversations:
        dataset = dataset+[conversation[i:i+2] for i in range(len(conversation)-1)]

def load_movie_lines(movie_lines_path):
    movie_lines_file = open(movie_lines_path,"r")
    movie_lines_raw = movie_lines_file.readlines()
    movie_lines_formatted = {}
    for line in movie_lines_raw:
        line_contents = line.split(" +++$+++ ")
        movie_lines_formatted{line_contents[0]} = line_contents[4]
    return movie_lines_formatted

def load_movie_conversations(movie_conversations_path):
    movie_conversations_file = open(movie_conversations_path,"r")
    movie_conversations_raw = movie_conversations_file.readlines()
    movie_conversations_formatted = []
    for conversation in movie_lines_raw:
        line_contents = line.split(" +++$+++ ")
        movie_conversations_formatted.append(line_contents[3])
    
class Embedder():


    def __init__(emb_path):
        pass

    def __call__(data):
        
        return embeddings
