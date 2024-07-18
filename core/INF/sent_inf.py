# Making a lightweight config class that allows for struct like attribute accessing
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np


DEVICE = torch.device('cpu')
import pickle

with open('core/INF/tokens/tokenizer_bert.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

# Setting parameters for our model
config = {# We get the vocabulary size used by our tokenizer
          'vocab_size': 30522,
          
          # We will use 128 dimensional token embeddings initially
          'embedding_dimensions': 128,
        
          # We're only going to use a maximum of 100 tokens per input sequence
          'max_tokens': 100,

          # Number of attention heads to be used
          'num_attention_heads': 8,

          # Dropout on feed-forward network
          'hidden_dropout_prob': 0.3,

          # Number of neurons in the intermediate hidden layer (quadruple the number of emb dims)
          'intermediate_size': 128 * 4,

          # How many encoder blocks to use in our architecture
          'num_encoder_layers': 2,

          # Device
          'device': DEVICE,
    
}
# Wrapping our config dict with the lightweight class
config = Config(config)



class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Create an embedding layer, with ~32,000 possible embeddings, each having 128 dimensions
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dimensions)

    def forward(self, tokenized_sentence):
        return self.token_embedding(tokenized_sentence)
    


class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        pe = torch.zeros(config.max_tokens, config.embedding_dimensions)
        position = torch.arange(0, config.max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = 1 / (10000 ** (torch.arange(0, config.embedding_dimensions, 2).float() / config.embedding_dimensions))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.pe = self.pe.to(config.device)
        
    def forward(self, x):
        return x + self.pe[:, 0]
    


def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k) 
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)
    
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        
    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state), 
                                                    self.k(hidden_state), 
                                                    self.v(hidden_state)) 
        return attn_outputs

class MultiHeadAttention(nn.Module): 
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embedding_dimensions
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1) 
        x = self.output_linear(x)
        return x
    

class FeedForward(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.embedding_dimensions, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.embedding_dimensions)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        x = self.linear_1(x) 
        x = self.gelu(x)
        x = self.linear_2(x) 
        x = self.dropout(x) 
        return x




class PostLNEncoder(nn.Module):
    "The original architecture used in Attention Is All You Need"
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimensions)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimensions)
        self.feed_forward = FeedForward(config)
        
    def forward(self, x):
        # Layer normalization + skip connections over the self-attention block
        x = self.layer_norm1(x + self.attention(x))
        # Layer norm + skip connections over the FFN
        x = self.layer_norm2(x + self.feed_forward(x))
        return x
        
class Encoder(nn.Module):
    "The improved pre-LN architecture"
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimensions)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimensions)
        self.feed_forward = FeedForward(config)
        
    def forward(self, x):
        # First perform layer normalization
        hidden_state = self.layer_norm1(x)
        # Then apply attention + skip connection
        x = x + self.attention(hidden_state)
        
        # Apply layer normalization before inputting to the FFN
        hidden_state = self.layer_norm2(x)
        # Apply FNN + skip connection
        x = x + self.feed_forward(hidden_state)
        return x    

class ClassifierHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(config.max_tokens * config.embedding_dimensions, 2 * config.embedding_dimensions)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2 * config.embedding_dimensions, 1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return torch.sigmoid(x) # Sigmoid activation for binary classification
    
# Initialize a classifier head
classifier = ClassifierHead(config).to(config.device)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = TokenEmbedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.encoders = nn.ModuleList([Encoder(config) for _ in range(config.num_encoder_layers)])
        self.classifier_head = ClassifierHead(config)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for encoder in self.encoders:
            x = encoder(x)
        return self.classifier_head(x)
    

    
# Instantiate a full model
model = Transformer(config).to(config.device)
# Initialize the model
model = Transformer(config).to(config.device)

# Load the state dict
state_dict = torch.load('core/INF/models/sentiment_model.pth', map_location=config.device)

# Load the weights into the model
model.load_state_dict(state_dict)

loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



def sentiment(sentence):
    tokenized_text = {}
    max_length = config.max_tokens
    tokenized = tokenizer.encode_plus(sentence, padding='max_length', truncation=True, max_length=max_length)
    prep = tokenized['input_ids']
    # print(len(prep))
    # print(type(prep))
    prep = torch.LongTensor(prep)
    prep = prep.to(config.device)
    prep = prep.unsqueeze(0)
    print(prep.shape)
    output = model(prep)
    print(output.item())
    if(output.item()*100 < 0.4):
        return "Negative"
    elif(output.item()*100 > 0.6):
        return "Positive"
    else:
        return "Neutral"
    
# print(sentiment())

