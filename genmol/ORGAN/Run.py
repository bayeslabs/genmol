import tqdm as tqdm
from Metrics_Reward import *
from Data import *
from Trainer import fit
from Model import ORGAN


def sampler(model):
    n_samples = 100
    samples = []
    with tqdm(total=n_samples, desc='Generating Samples')as T:
        while n_samples > 0:
            current_samples = model.sample(min(n_samples, 64), max_length=100)
            samples.extend(current_samples)
            n_samples -= len(current_samples)
            T.update(len(current_samples))

    return samples


def evaluate(test, samples, test_scaffolds=None, ptest=None, ptest_scaffolds=None):
    gen = samples
    metrics = get_all_metrics(test, gen, k=[1000, 1000], n_jobs=1,
                              device=device,
                              test_scaffolds=test_scaffolds,
                              ptest=ptest, ptest_scaffolds=ptest_scaffolds)
    for name, value in metrics.items():
        print('{}, {}'.format(name, value))


model = ORGAN()
fit(model, train_data)
samples = sampler(model)
evaluate(test_data, samples, test_scaffold)
