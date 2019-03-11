from trainingsession import TrainingSession
from seq2seqxentropy import Seq2SeqXentropy
import tools
import pickle

print("loading data...")
print("Serialized Data Path:")
datapath = input()
data = pickle.load(open(datapath, "r+b"))
print("Initializing Embedder...")
embedder = tools.Embedder("/data/glove/glove.840B.300d.txt")
data_emb=[]
for datavalue in data:
    print("emb_stat:"+str(data.index(datavalue)))
    tools.progressBar("Embedding Data: ", data.index(datavalue),len(data))
    words0 = datavalue[0].split(" ")
    words1 = datavalue[1].split(" ")
    words0parsed = []
    words1parsed = []
    for word in words0:
        print("parsing 0")
        words0parsed.append(embedder(word))
    for word in words1:
        print("parsing 1")
        words1parsed.append(embedder(word))
    data_emb.append([words0parsed,words1parsed])
print("Data embedding complete")
print("Final Data Path: ")
finaldatapath = input()
pickle.dump(data_emb,open(finaldatapath, "r+b"))