import torch.nn as nn
import torch 
from encoder import Encoder
from decoder import Decoder 
from RNNCell import DecoderRNNCell, EncoderRNNCell
from os import path

def load_movie_data(data_path):
    lines = load_movie_lines(path.join(data_path, "movie_lines.txt"))
    conversations = load_movie_conversations(path.join(data_path, "movie_conversations.txt"))
    dataset = []
    for conversation in conversations:
        dataset = dataset+[conversation[i:i+2] for i in range(len(conversation)-1)]
    return dataset

def load_movie_lines(movie_lines_path):
    movie_lines_file = open(movie_lines_path,"r", encoding="utf-8",errors="replace")
    movie_lines_raw = movie_lines_file.readlines()
    movie_lines_formatted = {}
    for line in movie_lines_raw:
        line_contents = line.split(" +++$+++ ")
        movie_lines_formatted[line_contents[0]] = line_contents[4]
    return movie_lines_formatted

def load_movie_conversations(movie_conversations_path):
    movie_conversations_file = open(movie_conversations_path,"r", encoding="utf-8",errors="replace")
    movie_conversations_raw = movie_conversations_file.readlines()
    movie_conversations_formatted = []
    for conversation in movie_conversations_raw:
        line_contents = conversation.split(" +++$+++ ")
        movie_conversations_formatted.append(line_contents[3])
    return movie_conversations_formatted
    
class Embedder():

    def __init__(self, emb_path):
        self.emb_path = emb_path
        #emb_file = open(emb_path, "r", encoding="utf-8")
        #emb_lines = emb_file.readlines()
        #emb_lines_list = [emb_line.split(" ") for emb_line in emb_lines]
        pass

    def __call__(self, data):
        with open(self.emb_path, "r", encoding="utf-8") as emb_file:
            for line_raw in emb_file: 
                line = line_raw.split(" ")
                if line[0] == data:
                    return [float(x) for x in line[1:]]