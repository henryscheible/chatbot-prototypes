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
        embedder = Embedder("/data/glove/glove.6B.300d.txt","/data/glove/glove.840B.300d.txt")
        movie_lines_formatted = {}
        for line in movie_lines_raw:
            line_contents = line.split(" +++$+++ ")
            print("\rembedding movie lines: "+str(movie_lines_raw.index(line))+"/"+str(len(movie_lines_raw)))
            print(line_contents, encoding="utf-8")
            newwords =[]
            for word in line_contents[4].split(" "):
                newword = word.replace("\n","")
                newword = newword.replace("?","")
                newword = newword.replace("!","")
                newword = newword.replace("'","")
                newword = newword.replace(" ","")
                print("\n"+newword+"\n")
                newwords.append(newword)
            newwords = [word for word in newwords if word!=""]
            try:
                embeddings = embedder(newwords)
                movie_lines_formatted[line_contents[0]] = embeddings
            except:
                print("movie line has no embedding...skipping...")
    return movie_lines_formatted

def load_movie_conversations(movie_conversations_path):
    print("loading movie conversations...")
    with open(movie_conversations_path,"r", encoding="utf-8",errors="replace") as movie_conversations_file:
        movie_conversations_raw = movie_conversations_file.readlines(10000)
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

    def __init__(self, emb_path1, emb_path2):
        self.emb_path1 = emb_path1
        self.emb_path2 = emb_path2
        self.emb_file_1 = open(emb_path1,"r",encoding="utf-8")
        self.emb_file_2 = open(emb_path2,"r",encoding="utf-8")
        print("loading emb1")
        self.lines_1 = {}
        for i,line in enumerate(self.emb_file_1):
            sys.stdout.write("\rload: "+str(i))
            line_list = line.split(" ")
            self.lines_1[line_list[0]]=i
        print("loading emb2")
        self.lines_2 = {}
        for i,line in enumerate(self.emb_file_2):
            sys.stdout.write("\rload: "+str(i))
            line_list = line.split(" ")
            self.lines_1[line_list[0]]=i

    def __call__(self, databatch):
        poslistmutable = []
        for data in databatch:
            if data in self.lines_1.keys():
                poslistmutable.append([1,self.lines_1[data]])
                print(str([1,self.lines_1[data]]))
            elif data in self.lines_2.keys():
                poslistmutable.append([2,self.lines_2[data]])
                print(str([2,self.lines_2[data]]))
            else:
                raise Exception(data + " has no embedding...")
        returnlist = [None]*len(poslistmutable)
        poslist = tuple(poslistmutable)
        # indexstart = {pos:0 for pos in poslist}
        indexstart = {}
        for i,line in enumerate(self.emb_file_1):
            sys.stdout.write("\rpopulating emb1: "+str(i))
            for pos in [pos for pos in poslist if pos[0]==1]:
                if i==pos:
                    returnlist[poslist.index(pos),indexstart[pos]]
                    indexstart[pos] == poslist.index(pos)+1
        
        for i,line in enumerate(self.emb_file_2):
            sys.stdout.write("\rpopulating emb2: "+str(i))
            for pos in [pos for pos in poslist if pos[0]==2]:
                if i==pos:
                    returnlist[poslist.index(pos),indexstart[pos]]
                    indexstart[pos] == poslist.index(pos)+1

        return returnlist
                
    def deep(self, data):
        scan = 0
        for line_raw in self.emb_file_2: 
            scan+=1
            sys.stdout.write("\r deep scan: "+str(scan))
            sys.stdout.flush()
            line = line_raw.split(" ")
            if line[0] == data:
                return [float(x) for x in line[1:]]