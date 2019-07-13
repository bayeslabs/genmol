import warnings
import numpy as np
from scipy.spatial.distance import cosine as cos_distance
from fcd_torch import FCD as FCDMetric
from fcd_torch import calculate_frechet_distance
from Reward_Metrics import *
from rdkit import rdBase
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_all_metrics(test, gen, k=None, n_jobs=1, device=device,
                    batch_size=512, test_scaffolds=None,
                    ptest=None, ptest_scaffolds=None,
                    pool=None, gpu=None):
    """
    Computes all available metrics between test (scaffold test)
    and generated sets of SMILES.
    Parameters:
        test: list of test SMILES
        gen: list of generated SMILES
        k: list with values for unique@k. Will calculate number of
            unique molecules in the first k molecules. Default [1000, 10000]
        n_jobs: number of workers for parallel processing
        device: 'cpu' or 'cuda:n', where n is GPU device number
        batch_size: batch size for FCD metric
        test_scaffolds: list of scaffold test SMILES
            Will compute only on the general test set if not specified
        ptest: dict with precalculated statistics of the test set
        ptest_scaffolds: dict with precalculated statistics
            of the scaffold test set
        pool: optional multiprocessing pool to use for parallelization
        gpu: deprecated, use `device`

    Available metrics:
        * %valid
        * %unique@k
        * Frechet ChemNet Distance (FCD)
        * Fragment similarity (Frag)
        * Scaffold similarity (Scaf)
        * Similarity to nearest neighbour (SNN)
        * Internal diversity (IntDiv)
        * Internal diversity 2: using square root of mean squared
            Tanimoto similarity (IntDiv2)
        * %passes filters (Filters)
        * Distribution difference for logP, SA, QED, NP, weight
    """
    if k is None:
        k = [1000, 10000]
    rdBase
    metrics = {}
    if gpu is not None:
        warnings.warn(
            "parameter `gpu` is deprecated. Use `device`",
            DeprecationWarning
        )
        if gpu == -1:
            device = 'cpu'
        else:
            device = 'cuda:{}'.format(gpu)
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    metrics['valid'] = fraction_valid(gen)
    gen = remove_invalid(gen, canonize=True)
    if not isinstance(k, (list, tuple)):
        k = [k]
    for _k in k:
        metrics['unique@{}'.format(_k)] = fraction_unique(gen, _k)

    if ptest is None:
        ptest = compute_intermediate_statistics(test, n_jobs=n_jobs,
                                                device=device,
                                                batch_size=batch_size,
                                                pool=pool)
    if test_scaffolds is not None and ptest_scaffolds is None:
        ptest_scaffolds = compute_intermediate_statistics(
            test_scaffolds, n_jobs=n_jobs,
            device=device, batch_size=batch_size,
            pool=pool
        )
    mols = mapper(pool)(get_mol, gen)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    metrics['FCD/Test'] = FCDMetric(**kwargs_fcd)(gen=gen, pref=ptest['FCD'])
    metrics['SNN/Test'] = SNNMetric(**kwargs)(gen=mols, pref=ptest['SNN'])
    metrics['Frag/Test'] = FragMetric(**kwargs)(gen=mols, pref=ptest['Frag'])
    metrics['Scaf/Test'] = ScafMetric(**kwargs)(gen=mols, pref=ptest['Scaf'])
    if ptest_scaffolds is not None:
        metrics['FCD/TestSF'] = FCDMetric(**kwargs_fcd)(
            gen=gen, pref=ptest_scaffolds['FCD']
        )
        metrics['SNN/TestSF'] = SNNMetric(**kwargs)(
            gen=mols, pref=ptest_scaffolds['SNN']
        )
        metrics['Frag/TestSF'] = FragMetric(**kwargs)(
            gen=mols, pref=ptest_scaffolds['Frag']
        )
        metrics['Scaf/TestSF'] = ScafMetric(**kwargs)(
            gen=mols, pref=ptest_scaffolds['Scaf']
        )

    metrics['IntDiv'] = internal_diversity(mols, pool, device=device)
    metrics['IntDiv2'] = internal_diversity(mols, pool, device=device, p=2)
    metrics['Filters'] = fraction_passes_filters(mols, pool)

    # Properties
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED), ('NP', NP),
                       ('weight', weight)]:
        metrics[name] = FrechetMetric(func, **kwargs)(gen=mols,
                                                      pref=ptest[name])
    enable_rdkit_log()
    if close_pool:
        pool.terminate()
    return metrics


def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED), ('NP', NP),
                       ('weight', weight)]:
        statistics[name] = FrechetMetric(func, **kwargs).precalc(mols)
    if close_pool:
        pool.terminate()
    return statistics


def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    else:
        return [x for x in mapper(n_jobs)(canonic_smiles, gen) if x is not None]


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs, fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref['fps'], pgen['fps'],
                                    device=self.device)


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])


class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])


class FrechetMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {'mu': np.mean(values), 'var': np.var(values)}

    def metric(self, pref, pgen):
        return calculate_frechet_distance(
            pref['mu'], pref['var'], pgen['mu'], pgen['var']
        )


class MetricsReward:
    supported_metrics = ['fcd', 'snn', 'fragments', 'scaffolds',
                         'internal_diversity', 'filters',
                         'logp', 'sa', 'qed', 'np', 'weight']

    @staticmethod
    def _nan2zero(value):
        if value == np.nan:
            return 0

        return value

    def __init__(self, n_ref_subsample, n_rollouts, n_jobs, metrics=[]):
        assert all([m in MetricsReward.supported_metrics for m in metrics])

        self.n_ref_subsample = n_ref_subsample
        self.n_rollouts = n_rollouts
        # TODO: profile this. Pool works too slow.
        n_jobs = n_jobs if False else 1
        self.n_jobs = n_jobs
        self.metrics = metrics

    def get_reference_data(self, data):
        ref_smiles = remove_invalid(data, canonize=True, n_jobs=self.n_jobs)
        ref_mols = mapper(self.n_jobs)(get_mol, ref_smiles)
        return ref_smiles, ref_mols

    def _get_metrics(self, ref, ref_mols, rollout):
        rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
        result = [[0 if m is None else 1] for m in rollout_mols]

        if sum([r[0] for r in result], 0) == 0:
            return result

        rollout = remove_invalid(rollout, canonize=True, n_jobs=self.n_jobs)
        rollout_mols = mapper(self.n_jobs)(get_mol, rollout)
        if len(rollout) < 2:
            return result

        if len(self.metrics):
            for metric_name in self.metrics:
                if metric_name == 'fcd':
                    m = FCDMetric(n_jobs=self.n_jobs)(ref, rollout)
                elif metric_name == 'morgan':
                    m = SNNMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'fragments':
                    m = FragMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'scaffolds':
                    m = ScafMetric(n_jobs=self.n_jobs)(ref_mols, rollout_mols)
                elif metric_name == 'internal_diversity':
                    m = internal_diversity(rollout_mols, n_jobs=self.n_jobs)
                elif metric_name == 'filters':
                    m = fraction_passes_filters(
                        rollout_mols, n_jobs=self.n_jobs
                    )
                elif metric_name == 'logp':
                    m = -FrechetMetric(func=logP, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )
                elif metric_name == 'sa':
                    m = -FrechetMetric(func=SA, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )
                elif metric_name == 'qed':
                    m = -FrechetMetric(func=QED, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )
                elif metric_name == 'np':
                    m = -FrechetMetric(func=NP, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )
                elif metric_name == 'weight':
                    m = -FrechetMetric(func=weight, n_jobs=self.n_jobs)(
                        ref_mols, rollout_mols
                    )

                m = MetricsReward._nan2zero(m)
                for i in range(len(rollout)):
                    result[i].append(m)

        return result

    def __call__(self, gen, ref, ref_mols):

        idxs = random.sample(range(len(ref)), self.n_ref_subsample)
        ref_subsample = [ref[idx] for idx in idxs]
        ref_mols_subsample = [ref_mols[idx] for idx in idxs]

        gen_counter = Counter(gen)
        gen_counts = [gen_counter[g] for g in gen]

        n = len(gen) // self.n_rollouts
        rollouts = [gen[i::n] for i in range(n)]

        metrics_values = [self._get_metrics(
            ref_subsample, ref_mols_subsample, rollout
        ) for rollout in rollouts]
        metrics_values = map(
            lambda rm: [
                sum(r, 0) / len(r)
                for r in rm
            ], metrics_values)
        reward_values = sum(zip(*metrics_values), ())
        reward_values = [v / c for v, c in zip(reward_values, gen_counts)]

        return reward_values
