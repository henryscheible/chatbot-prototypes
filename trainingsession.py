import torch.nn as nn
import torch
from threading import Thread

class TrainingSession():

    def __init__(self, trainingmodel, loss_function, optimizer, emb_size, hid_size, output_size):
        self.trainingmodel = trainingmodel
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.output_size = output_size
        self.model = trainingmodel(emb_size, hid_size, output_size)

    def train(self, data):
        for batch_samples, batch_lables in iterate_batches(data):
            batch_samples_t = torch.Tensor(batch_samples)
            batch_lables_t = torch.Tensor(batch_lables)
            output_t = model(batch_samples_t)
            loss_t = loss_function(output_t, batch_lables_t)
            optimizer.step(loss_t)
            optimizer.zero_grad()