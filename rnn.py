import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, char_size, embedding_dim, hidden_dim, output_dim,
                 num_layers):
        super(LSTM, self).__init__()
        
        # Embedding layer to convert to vectors
        self.embedding = nn.Embedding(char_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers = num_layers)

        # Dense prediction layer
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

        # Sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, text_lengths):
        embeddings = self.embedding(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeddings, 
                    text_lengths.cpu(),batch_first=True)

        _, (hidden_state, _) = self.lstm(packed_embedded)

        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)

        linear_outputs = self.linear(hidden)

        outputs = self.sigmoid(linear_outputs)

        return outputs