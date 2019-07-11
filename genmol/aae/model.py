import torch

import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


from data import *


emb_dim = 30
hidden_dim = 64
latent_dim = 4
disc_input = 64
disc_output = 84
batch_size = 50


class encoder(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_dim, latent_dim):
        super(encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.vocab = vocab

        self.embeddings_layer = nn.Embedding(len(vocab), emb_dim, padding_idx=c2i['<pad>'])

        self.rnn = nn.LSTM(emb_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
        nn.Drop = nn.Dropout(p=0.25)

    def forward(self, x, lengths):
        batch_size = x.shape[0]

        x = self.embeddings_layer(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        output, (_, x) = self.rnn(x)

        x = x.permute(1, 2, 0).view(batch_size, -1)
        x = self.fc(x)
        state = self.relu(x)
        return state


class decoder(nn.Module):
    def __init__(self, vocab, emb_dim, latent_dim, hidden_dim):
        super(decoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.vocab = vocab

        self.latent = nn.Linear(latent_dim, hidden_dim)
        self.embeddings_layer = nn.Embedding(len(vocab), emb_dim, padding_idx=c2i['<pad>'])
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(vocab))

    def forward(self, x, lengths, state, is_latent_state=False):
        if is_latent_state:
            c0 = self.latent(state)

            c0 = c0.unsqueeze(0)
            h0 = torch.zeros_like(c0)

            state = (h0, c0)

        x = self.embeddings_layer(x)

        x = pack_padded_sequence(x, lengths, batch_first=True)

        x, state = self.rnn(x, state)

        x, lengths = pad_packed_sequence(x, batch_first=True)
        x = self.fc(x)

        return x, lengths, state


class Discriminator(nn.Module):
    def __init__(self, latent_dim, disc_input, disc_output):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim
        self.disc_input = disc_input
        self.disc_output = disc_output

        self.lin1 = nn.Linear(latent_dim, disc_input)
        self.lin2 = nn.Linear(disc_input, disc_output)
        self.lin3 = nn.Linear(disc_output, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        x = self.sig(x)
        return x
class AAE(nn.Module):
    def __init__(self):
        super(AAE,self).__init__()
        self.encoder = encoder(vocab,emb_dim,hidden_dim,latent_dim)
        self.decoder = decoder(vocab,emb_dim,latent_dim,hidden_dim)
        self.discriminator = Discriminator(latent_dim,disc_input,disc_output)
