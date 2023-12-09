import torch
import torch.nn as nn
import torch.nn.functional as F

# LSTM Classification Model
class UsernameClassifierLSTM(nn.ModuleList):

	def __init__(self, args):
		super(UsernameClassifierLSTM, self).__init__()
		
		self.batch_size = args.batch_size
		self.hidden_dim = args.hidden_dim
		self.LSTM_layers = args.lstm_layers
		self.linear_features = args.linear_features
		self.input_size = len(args.chars) + 1
		
        # Dropout layer
		self.dropout = nn.Dropout(0.5)
		
        # Embedding layer
		self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx = 0)
		
        # LSTM layer
		self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers, batch_first=True)
		
        # Linear layers
		self.linear1 = nn.Linear(in_features=self.hidden_dim, out_features=self.linear_features)
		self.linear2 = nn.Linear(self.linear_features, 1)
		
	def forward(self, x):
		h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
		c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
		
        # Apply layers
		out = self.embedding(x)
		out, _ = self.lstm(out, (h,c))
		out = self.dropout(out)
		out = torch.relu_(self.linear1(out[:,-1,:]))
		out = self.dropout(out)
		out = torch.sigmoid(self.linear2(out))
		return out