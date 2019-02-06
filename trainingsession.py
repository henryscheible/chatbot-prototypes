import torch.nn as nn
import torch
from threading import Thread

class TrainingSession():

    def __init__(self, trainingmodel, emb_size, hid_size, output_size):
        self.trainingmodel = trainingmodel
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.output_size = output_size
        self.trainthread = Thread(target=trainloop())

    def start():
        trainthread.start()

    def trainloop():
