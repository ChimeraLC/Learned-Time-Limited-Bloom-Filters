import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from preprocessing import Preprocessing
from model import UsernameClassifier

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class UsernameDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
class Runner:
    def __init__(self, args):
        self.__init_data__(args)

        # Get config arguments
        self.args = args
        self.batch_size = args.batch_size

        # Create model
        self.model = UsernameClassifier(args)

    def __init_data__(self, args):
        # Load and process data
        self.preprocessing = Preprocessing(args)
        self.preprocessing.load_data()

        # Get username data and classifications
        self.x_train = self.preprocessing.label_sequences(self.preprocessing.x_train)
        self.x_test = self.preprocessing.label_sequences(self.preprocessing.x_test)

        self.y_train = self.preprocessing.y_train.values
        self.y_test = self.preprocessing.y_test.values

    def train(self):
        # Try to load model data
        if os.path.isfile(self.args.model_save_path) and not self.args.retrain_model:
            print("Loaded model from file")
            self.model.load_state_dict(torch.load(self.args.model_save_path))
            self.model.eval()
        else:
            print("Training model")
            # Create datasets
            train_data = UsernameDataset(self.x_train, self.y_train)
            test_data = UsernameDataset(self.x_test, self.y_test)
            self.train_loader = DataLoader(train_data, batch_size = self.args.batch_size)
            self.test_loader = DataLoader(test_data, batch_size = self.args.batch_size)
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            for epoch in range(self.args.train_epochs):
                        
                        predictions = []
                        
                        self.model.train()
                        
                        for x_batch, y_batch in self.train_loader:
                            
                            x = x_batch.type(torch.LongTensor)
                            y = y_batch.type(torch.FloatTensor)
                            
                            y_pred = self.model(x)
                            
                            loss = F.binary_cross_entropy(y_pred, y)
                            
                            model_optim.zero_grad()
                            
                            loss.backward()
                            
                            model_optim.step()
                            
                            predictions += list(y_pred.squeeze().detach().numpy())
                        
                        test_predictions = self.predict()
                        
                        train_accuary = self.accuracy(self.y_train, predictions)
                        test_accuracy = self.accuracy(self.y_test, test_predictions)
                        
                        print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))

                        #TODO: early stopping
            torch.save(self.model.state_dict(), self.args.model_save_path)
    
    # Makes predictions over the dataset
    def predict(self):
        predictions = []
        with torch.no_grad():
            for x_batch, _ in self.test_loader:
                x = x_batch.type(torch.LongTensor)
                
                y_pred = self.model(x)
                predictions += list(y_pred.detach().numpy())
                
        return predictions
            
    # Gets the accuracy of a set of predictions
    def accuracy(self, actual, predicted):
        correct = 0
        
        for true, pred in zip(actual, predicted):
            if pred.round() == true:
                 correct += 1
                
        return correct / len(actual)

    # Given a username, returns the prediction
    def classify(self, username):
         self.model.eval()
         # Convert username to labels
         labelled = self.preprocessing.string_to_label(username)
         # Convert to tensor
         input = torch.IntTensor(labelled).unsqueeze(0)
         return self.model(input).round().tolist()[0][0]        #todo: clean this up
    
    # Given a list of username, returns the predictions
    def longClassify(self, usernames):
        inputs = self.preprocessing.label_sequences(usernames)
        predictions = []
        self.model.eval()
        train_data = UsernameDataset(inputs, inputs)
        train_loader = DataLoader(train_data, batch_size = self.args.batch_size)
        with torch.no_grad():
            for x_batch, _ in train_loader:
                x = x_batch.type(torch.LongTensor)
                
                y_pred = self.model(x)
                predictions += list(y_pred.detach().numpy())
        return predictions


    # Returns the overall size of the model
    def get_size(self):
        size = 0
        for param in self.model.parameters():
            size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            size += buffer.nelemnt() * buffer.elemnt_size()
        return size
