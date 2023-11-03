from embedding import StringConverter
from rnn import LSTM
import torch
import torch.nn as nn
import torch.optim as optim
#https://www.usna.edu/Users/cs/nchamber/data/twitternames/
# Binary classification accuracy function
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

#data = pd.read_csv("user-ct-test-collection-01.txt", sep="\t")
#urllist = data.ClickURL.dropna().unique()

# Get string converter
converter = StringConverter()

# Define hyperparameters
device = torch.device("cuda")
batch_size = 64
char_size = converter.num_chars()
embedding_dim = 100
hidden_dim = 64
output_dim = 2
num_layers = 2

# Create model
model = LSTM(char_size, embedding_dim, hidden_dim, output_dim, num_layers)
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-4)
criterion = nn.BCELoss()
criterion = criterion.to(device)

