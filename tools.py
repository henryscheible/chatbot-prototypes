import torch.nn as nn
import torch 
from encoder import Encoder
from decoder import Decoder 
from RNNCell import DecoderRNNCell, EncoderRNNCell
from os import path
import ast
import sys

def load_movie_data(data_path):
    lines = load_movie_lines(path.join(data_path, "movie_lines.txt"))
    conversations = load_movie_conversations(path.join(data_path, "movie_conversations.txt"))
    dataset = []
    print("formatting data...")
    for conversation in conversations:
        print("parsing conversation "+str(conversations.index(conversation))+"/"+str(len(conversations)))
        dataset = dataset+[conversation[i:i+2] for i in range(len(conversation)-1)]
    print("substituting lines...")
    subdataset = []
    for datavector in dataset:
        print("substituting "+str(dataset.index(datavector))+"/"+str(len(dataset)))
        subdataset.append([lines[datavector[0]],lines[datavector[1]]])
    return subdataset

def load_movie_lines(movie_lines_path):
    print("Loading movie lines...")
    with open(movie_lines_path,"r", encoding="utf-8",errors="replace") as movie_lines_file:
        movie_lines_raw = movie_lines_file.readlines()
        print("formatting movie lines...")
        movie_lines_formatted = {}
        for line in movie_lines_raw:
            line_contents = line.split(" +++$+++ ")
            movie_lines_formatted[line_contents[0]] = line_contents[4]
    return movie_lines_formatted

def load_movie_conversations(movie_conversations_path):
    print("loading movie conversations...")
    with open(movie_conversations_path,"r", encoding="utf-8",errors="replace") as movie_conversations_file:
        movie_conversations_raw = movie_conversations_file.readlines()
        movie_conversations_formatted = []
        print("formatting movie conversations...")
        for conversation in movie_conversations_raw:
            line_contents = conversation.split(" +++$+++ ")
            lines = ast.literal_eval(line_contents[3])
            movie_conversations_formatted.append(lines)
    return movie_conversations_formatted
    
def progressBar(title, value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\r"+title+": [{0}] {1}%".format(arrow + spaces, round(percent * 100,2)))
        sys.stdout.flush()

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