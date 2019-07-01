
from trainer import *
from vae_model import VAE
from data import *
from samples import *

model = VAE(vocab,vector).to(device)
fit(model, train_data)
model.eval()
sample = sample.take_samples(model,n_batch)
print(sample)