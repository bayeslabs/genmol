import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.optim import Adam
import random

from Data import *

n_batch = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
discriminator_pretrain_epochs = 50
discriminator_epochs = 10
generator_pretrain_epochs = 50
max_length = 100
save_frequency = 25
generator_updates = 1
discriminator_updates = 1
n_samples = 64
n_rollouts = 16
pg_iters = 10

class PolicyGradientLoss(nn.Module):
    def forward(self, outputs, targets, rewards, lengths):
        log_probs = F.log_softmax(outputs, dim=2)
        items = torch.gather(
            log_probs, 2, targets.unsqueeze(2)
        ) * rewards.unsqueeze(2)
        loss = -sum(
            [t[:l].sum() for t, l in zip(items, lengths)]
        ) / lengths.sum().float()
        return loss


def generator_collate_fn(model):
    def collate(data):
        data.sort(key=len, reverse=True)
        tensors = [model.string2tensor(string)
                   for string in data]

        prevs = pad_sequence(
            [t[:-1] for t in tensors],
            batch_first=True, padding_value=c2i['<pad>']
        )
        nexts = pad_sequence(
            [t[1:] for t in tensors],
            batch_first=True, padding_value=c2i['<pad>']
        )
        lens = torch.tensor(
            [len(t) - 1 for t in tensors],
            dtype=torch.long, device=device)
        return prevs, nexts, lens

    return collate


def get_dataloader(training_data, collate_fn):
    return DataLoader(training_data, batch_size=n_batch,
                      shuffle=True, num_workers=8, collate_fn=collate_fn, worker_init_fn=None)


def _pretrain_generator_epoch(model, tqdm_data, criterion, optimizer):
    model.discriminator.eval()
    if optimizer is None:
        model.eval()
    else:
        model.train()

    postfix = {'loss': 0, 'running_loss': 0}

    for i, batch in enumerate(tqdm_data):
        (prevs, nexts, lens) = (data.to(device) for data in batch)
        outputs, _, _, = model.generator_forward(prevs, lens)

        loss = criterion(outputs.view(-1, outputs.shape[-1]),
                         nexts.view(-1))

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        postfix['loss'] = loss.item()
        postfix['running_loss'] += (
                                           loss.item() - postfix['running_loss']
                                   ) / (i + 1)
        tqdm_data.set_postfix(postfix)

    postfix['mode'] = ('Pretrain: eval generator'
                       if optimizer is None
                       else 'Pretrain: train generator')
    return postfix


def _pretrain_generator(model, train_loader):
    generator = model.generator
    criterion = nn.CrossEntropyLoss(ignore_index=c2i['<pad>'])
    optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-4)

    model.zero_grad()
    for epoch in range(generator_pretrain_epochs):
        tqdm_data = tqdm(train_loader, desc='Generator training (epoch #{})'.format(epoch))
        postfix = _pretrain_generator_epoch(model, tqdm_data, criterion, optimizer)
        if epoch % save_frequency == 0:
            generator = generator.to('cpu')
            torch.save(generator.state_dict(), 'model.csv'[:-4] +
                       '_generator_{0:03d}.csv'.format(epoch))
        generator = generator.to(device)


def discriminator_collate_fn(model):
    def collate(data):
        data.sort(key=len, reverse=True)
        tensors = [model.string2tensor(string) for string in data]
        inputs = pad_sequence(tensors, batch_first=True, padding_value=c2i['<pad>'])

        return inputs

    return collate


def _pretrain_discriminator_epoch(model, tqdm_data,
                                  criterion, optimizer=None):
    model.eval()
    if optimizer is None:
        model.eval()
    else:
        model.train()

    postfix = {'loss': 0,
               'running_loss': 0}
    for i, inputs_from_data in enumerate(tqdm_data):
        inputs_from_data = inputs_from_data.to(device)
        inputs_from_model, _ = model.sample_tensor(n_batch, 100)

        targets = torch.zeros(n_batch, 1, device=device)
        outputs = model.discriminator_forward(inputs_from_model)
        loss = criterion(outputs, targets) / 2

        targets = torch.ones(inputs_from_data.shape[0], 1, device=device)
        outputs = model.discriminator_forward(inputs_from_data)
        loss += criterion(outputs, targets) / 2

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        postfix['loss'] = loss.item()
        postfix['running_loss'] += (loss.item() -
                                    postfix['running_loss']) / (i + 1)
        tqdm_data.set_postfix(postfix)

    postfix['mode'] = ('Pretrain: eval discriminator'
                       if optimizer is None
                       else 'Pretrain: train discriminator')
    return postfix


def _pretrain_discriminator(model, train_loader):
    discriminator = model.discriminator
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.discriminator.parameters(),
                                 lr=1e-4)

    model.zero_grad()
    for epoch in range(discriminator_pretrain_epochs):
        tqdm_data = tqdm(
            train_loader,
            desc='Discriminator training (epoch #{})'.format(epoch)
        )
        postfix = _pretrain_discriminator_epoch(
            model, tqdm_data, criterion, optimizer
        )
        if epoch % save_frequency == 0:
            discriminator = discriminator.to('cpu')
            torch.save(discriminator.state_dict(), 'model.csv'[:-4] + '_discriminator_{0:03d}.csv'.format(epoch))
            discriminator = discriminator.to(device)


def _policy_gradient_iter(model, train_loader, criterion, optimizer, iter_, ref_smiles, ref_mols):
    smooth = 0.1

    # Generator
    gen_postfix = {'generator_loss': 0,
                   'smoothed_reward': 0}

    gen_tqdm = tqdm(range(generator_updates),
                    desc='PG generator training (iter #{})'.format(iter_))
    for _ in gen_tqdm:
        model.eval()
        sequences, rewards, lengths = model.rollout(ref_smiles, ref_mols, n_samples=n_samples,
                                                    n_rollouts=n_rollouts, max_len=max_length)
        model.train()

        lengths, indices = torch.sort(lengths, descending=True)
        sequences = sequences[indices, ...]
        rewards = rewards[indices, ...]

        generator_outputs, lengths, _ = model.generator_forward(
            sequences[:, :-1], lengths - 1
        )
        generator_loss = criterion['generator'](
            generator_outputs, sequences[:, 1:], rewards, lengths
        )

        optimizer['generator'].zero_grad()
        generator_loss.backward()
        nn.utils.clip_grad_value_(model.generator.parameters(), clip_value=5)
        optimizer['generator'].step()

        gen_postfix['generator_loss'] += (
                                                 generator_loss.item() -
                                                 gen_postfix['generator_loss']
                                         ) * smooth
        mean_episode_reward = torch.cat(
            [t[:l] for t, l in zip(rewards, lengths)]
        ).mean().item()
        gen_postfix['smoothed_reward'] += (
                                                  mean_episode_reward - gen_postfix['smoothed_reward']
                                          ) * smooth
        gen_tqdm.set_postfix(gen_postfix)

    # Discriminator
    discrim_postfix = {'discrim-r_loss': 0}
    discrim_tqdm = tqdm(
        range(discriminator_updates),
        desc='PG discrim-r training (iter #{})'.format(iter_)
    )
    for _ in discrim_tqdm:
        model.generator.eval()
        n_batches = (
                            len(train_loader) + n_batch - 1
                    ) // n_batch
        sampled_batches = [
            model.sample_tensor(n_batch,
                                max_length=max_length)[0]
            for _ in range(n_batches)
        ]

        for _ in range(discriminator_epochs):
            random.shuffle(sampled_batches)

            for inputs_from_model, inputs_from_data in zip(
                    sampled_batches, train_loader
            ):
                # print(inputs_from_model)
                inputs_from_data = inputs_from_data.to(device)
                print(inputs_from_data)

                discrim_outputs = model.discriminator_forward(
                    inputs_from_model
                )
                discrim_targets = torch.zeros(len(discrim_outputs),
                                              1, device=device)
                discrim_loss = criterion['discriminator'](
                    discrim_outputs, discrim_targets
                ) / 2

                discrim_outputs = model.discriminator.forward(
                    inputs_from_data)
                discrim_targets = torch.ones(
                    len(discrim_outputs), 1, device=device)
                discrim_loss += criterion['discriminator'](
                    discrim_outputs, discrim_targets
                ) / 2
                optimizer['discriminator'].zero_grad()
                discrim_loss.backward()
                optimizer['discriminator'].step()

                discrim_postfix['discrim-r_loss'] += (
                                                             discrim_loss.item() -
                                                             discrim_postfix['discrim-r_loss']
                                                     ) * smooth

        discrim_tqdm.set_postfix(discrim_postfix)

    postfix = {**gen_postfix, **discrim_postfix}
    postfix['mode'] = 'Policy Gradient (iter #{})'.format(iter_)
    return postfix


def _train_policy_gradient(model, pg_train_loader, ref_smiles, ref_mols):
    criterion = {
        'generator': PolicyGradientLoss(),
        'discriminator': nn.BCEWithLogitsLoss(),
    }

    optimizer = {
        'generator': torch.optim.Adam(model.generator.parameters(),
                                      lr=1e-4),
        'discriminator': torch.optim.Adam(
            model.discriminator.parameters(), lr=1e-4)
    }
    ref_smiles = ref_smiles
    ref_mols = ref_mols
    model.zero_grad()
    for iter_ in range(pg_iters):
        postfix = _policy_gradient_iter(model, pg_train_loader, criterion, optimizer, iter_, ref_smiles, ref_mols)


def fit(model, train_data):
    # Generator
    gen_collate_fn = generator_collate_fn(model)
    gen_train_loader = get_dataloader(train_data, gen_collate_fn)
    _pretrain_generator(model, gen_train_loader)

    # Discriminator
    dsc_collate_fn = discriminator_collate_fn(model)
    desc_train_loader = get_dataloader(train_data, dsc_collate_fn)
    _pretrain_discriminator(model, desc_train_loader)

    # Policy Gradient
    if model.metrics_reward is not None:
        (ref_smiles, ref_mols) = model.metrics_reward.get_reference_data(train_data)

        pg_train_loader = desc_train_loader
        _train_policy_gradient(model, pg_train_loader, ref_smiles, ref_mols)

        del ref_smiles
        del ref_mols
        #
        return model
