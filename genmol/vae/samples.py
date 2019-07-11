# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd
n_samples = 3000
n_jobs = 1
max_len = 100

class sample():
  def take_samples(model,n_batch):
    n = n_samples
    samples = []
    with tqdm(total=n_samples, desc='Generating samples') as T:
      while n > 0:
        current_samples = model.sample(min(n, n_batch), max_len)
        samples.extend(current_samples)
        n -= len(current_samples)
        T.update(len(current_samples))
    samples = pd.DataFrame(samples, columns=['SMILES'])
    return samples