import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from model import *

def pretrain(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.001)
    model.zero_grad()
    for epoch in range(4):
        if optimizer is None:
            model.train()
        else:
            model.eval()
        for i, (encoder_inputs, decoder_inputs, decoder_targets) in enumerate(train_loader):
            encoder_inputs = (data.to(device) for data in encoder_inputs)
            decoder_inputs = (data.to(device) for data in decoder_inputs)
            decoder_targets = (data.to(device) for data in decoder_targets)

            latent_code = model.encoder(*encoder_inputs)
            decoder_output, decoder_output_lengths, states = model.decoder(*decoder_inputs, latent_code,
                                                                           is_latent_state=True)

            decoder_outputs = torch.cat([t[:l] for t, l in zip(decoder_output, decoder_output_lengths)], dim=0)
            decoder_targets = torch.cat([t[:l] for t, l in zip(*decoder_targets)], dim=0)
            loss = criterion(decoder_outputs, decoder_targets)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def train(model, train_loader):
    criterion = {"enc": nn.CrossEntropyLoss(), "gen": lambda t: -torch.mean(F.logsigmoid(t)),"disc": nn.BCEWithLogitsLoss()}

    optimizers = {'auto': torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=0.001),
        'gen': torch.optim.Adam(model.encoder.parameters(), lr=0.001),
        'disc': torch.optim.Adam(model.discriminator.parameters(), lr=0.001)}

    model.zero_grad()
    for epoch in range(10):
        if optimizers is None:
            model.train()
        else:
            model.eval()

        for i, (encoder_inputs, decoder_inputs, decoder_targets) in enumerate(train_loader):
            encoder_inputs = (data.to(device) for data in encoder_inputs)
            decoder_inputs = (data.to(device) for data in decoder_inputs)
            decoder_targets = (data.to(device) for data in decoder_targets)

            latent_code = model.encoder(*encoder_inputs)
            decoder_output, decoder_output_lengths, states = model.decoder(*decoder_inputs, latent_code,
                                                                           is_latent_state=True)
            discriminator_output = model.discriminator(latent_code)

            decoder_outputs = torch.cat([t[:l] for t, l in zip(decoder_output, decoder_output_lengths)], dim=0)
            decoder_targets = torch.cat([t[:l] for t, l in zip(*decoder_targets)], dim=0)

            autoencoder_loss = criterion["enc"](decoder_outputs, decoder_targets)
            generation_loss = criterion["gen"](discriminator_output)

            if i % 2 == 0:
                discriminator_input = torch.randn(batch_size, latent_dim)
                discriminator_output = model.discriminator(discriminator_input)
                discriminator_targets = torch.ones(batch_size, 1)
            else:
                discriminator_targets = torch.zeros(batch_size, 1)
            discriminator_loss = criterion["disc"](discriminator_output, discriminator_targets)

            if optimizers is not None:
                optimizers["auto"].zero_grad()
                autoencoder_loss.backward(retain_graph=True)
                optimizers["auto"].step()

                optimizers["gen"].zero_grad()
                autoencoder_loss.backward(retain_graph=True)
                optimizers["gen"].step()

                optimizers["disc"].zero_grad()
                autoencoder_loss.backward(retain_graph=True)
                optimizers["disc"].step()

def fit(model,train_data):
    train_loader = get_dataloader(model, train_data, collate_fn=None, shuffle=True)
    pretrain(model,train_loader)
    train(model,train_loader)

def get_collate_device(model):
    return device
def get_dataloader(model, data, collate_fn=None, shuffle=True):
    if collate_fn is None:
        collate_fn = get_collate_fn(model)
    return DataLoader(data, batch_size= batch_size,shuffle=shuffle,collate_fn=collate_fn)


def get_collate_fn(model):
    device = get_collate_device(model)

    def collate(data):
        data.sort(key=lambda x: len(x), reverse=True)

        tensors = [string2tensor(string, device=device) for string in data]
        lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long, device=device)

        encoder_inputs = pad_sequence(tensors, batch_first=True, padding_value=c2i["<pad>"])
        encoder_input_lengths = lengths - 2

        decoder_inputs = pad_sequence([t[:-1] for t in tensors], batch_first=True, padding_value=c2i["<pad>"])
        decoder_input_lengths = lengths - 1
        decoder_targets = pad_sequence([t[1:] for t in tensors], batch_first=True, padding_value=c2i["<pad>"])
        decoder_target_lengths = lengths - 1
        return (encoder_inputs, encoder_input_lengths), (decoder_inputs, decoder_input_lengths), (decoder_targets, decoder_target_lengths)

    return collate