from model import *
from tqdm import tqdm
from data import *
def sample(model,n_batch, max_len=100):
    with torch.no_grad():
        samples = []
        lengths = torch.zeros(n_batch, dtype=torch.long, device=device)
        state = sample_latent(n_batch)
        prevs = torch.empty(n_batch, 1, dtype=torch.long, device=device).fill_(c2i["<bos>"])
        one_lens = torch.ones(n_batch, dtype=torch.long, device=device)
        is_end = torch.zeros(n_batch, dtype=torch.uint8, device=device)
        for i in range(max_len):
            logits, _, state = model.decoder(prevs, one_lens, state, i == 0)
            currents = torch.argmax(logits, dim=-1)
            is_end[currents.view(-1) == c2i["<eos>"]] = 1
            if is_end.sum() == max_len:
                break

            currents[is_end, :] = c2i["<pad>"]
            samples.append(currents)
            lengths[~is_end] += 1
            prevs = currents
    if len(samples):
        samples = torch.cat(samples, dim=-1)
        samples = [tensor2string(t[:l]) for t, l in zip(samples, lengths)]
    else:
        samples = ['' for _ in range(n_batch)]
    return samples


def get_samples(model):
    samples = []
    n = 300
    max_len = 100
    with tqdm(total=300, desc='Generating samples') as T:
        while n > 0:
            current_samples = sample(model,min(n, batch_size), max_len)
            samples.extend(current_samples)
            n -= len(current_samples)
            T.update(len(current_samples))
    print(samples)

def tensor2string(tensor):
    ids = tensor.tolist()
    string = ids2string(ids, rem_bos=True, rem_eos=True)
    return string
def sample_latent(n):
    return torch.randn(n,latent_dim)