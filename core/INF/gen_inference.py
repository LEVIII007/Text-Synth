import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

seq_length = 64
def generate_square_subsequent_mask(sz):

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads):
        super(TextGen, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=seq_length, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    # Positional encoding is required. Else the model does not learn.
    def forward(self, x):
        emb = self.emb(x)
        
        # Generate input sequence mask with shape (SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)
        return out
    
import pickle

with open('core/INF/tokens/gpt_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

device = torch.device('cpu')
model = TextGen(
    vocab_size = len(tokenizer),
    embed_dim=100,
    num_layers=2,
    num_heads=2,
).to(device)
criterion = nn.CrossEntropyLoss()

model.load_state_dict(torch.load('core/INF/models/gen_model.pth', map_location=device))


SEQUENCE_LENGTH = 64

# Helper function to convert text to token ids and pad if necessary
def return_int_vector(text):
    tokens = tokenizer.encode(text)
#     if len(tokens) > SEQUENCE_LENGTH:
    tokens = tokens[-SEQUENCE_LENGTH:]
#     else:
#         tokens = [pad_token_id] * (SEQUENCE_LENGTH - len(tokens)) + tokens
    input_seq = torch.LongTensor(tokens).unsqueeze(0)
    return input_seq

# Helper function for greedy sampling
def sample_next(predictions):
    """
    Greedy sampling.
    """
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    next_token = torch.argmax(probabilities)
    return int(next_token.cpu())

# Text generation function
def text_generator(sentence, generate_length):
    model.eval()
    sample = sentence
    for _ in range(generate_length):
        int_vector = return_int_vector(sample)
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions)
        sample += tokenizer.decode([next_token])
    print(sample)
    print(len(sample))
    print('\n')
    return sample