"""Frozen-checkpoint inference only; writes exclusively to this diagnostic directory."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from relisten import config
from relisten.end_to_end import DirectRelistening, PairDataset, _collate, _histories
from relisten.labels import with_labels
from relisten.light_models import benchmark_pairs
from relisten.metrics import evaluate_ranking
from relisten.sequence_model import PARAMETERS, _batch_inputs, device_name
from relisten.vendor.sasrec_model import SASRecEncoder

OUT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def protected():
    paths = []
    for root in ('src', 'tests', 'scripts', 'notebooks', 'results', '.codex'):
        paths.extend(p for p in (config.ROOT / root).rglob('*')
                     if p.is_file() and p.suffix != '.pyc' and OUT not in p.parents)
    paths.extend(config.ROOT / p for p in ('AGENTS.md', 'DECISIONS.md', 'README.md', 'pyproject.toml', 'requirements.lock')
                 if (config.ROOT / p).exists())
    paths.extend((config.PROCESSED / 'models').glob('SASRec_e2e_l*.pt'))
    paths.extend((config.PROCESSED / 'models/day_29').glob('SASRec*'))
    paths.extend(config.PROCESSED.glob('SASRec_e2e_*.parquet'))
    paths.append(config.PROCESSED / 'pairs.parquet')
    return {str(p.relative_to(config.ROOT)): digest(p) for p in sorted(set(paths))}


def distribution(values):
    x = np.asarray(values, dtype=float)
    qs = np.quantile(x, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {'n': len(x), 'mean': float(x.mean()), 'std': float(x.std()),
            'unique': len(np.unique(x)),
            **{k: float(v) for k, v in zip(('min', 'p01', 'p05', 'p25', 'median', 'p75', 'p95', 'p99', 'max'), qs)}}


def main():
    begin = time.monotonic()
    torch.set_num_threads(4)
    before = protected()
    (OUT / 'protected_before.json').write_text(json.dumps(before, indent=2))
    items = np.load(config.PROCESSED / 'models/day_29/SASRec_items.npy')
    pairs = benchmark_pairs().filter(pl.col('task') == 'T2')
    print(f'device={device_name()}; loading fixed histories', flush=True)
    histories = _histories(items, pairs['anchor_timestamp'].max())
    metrics_rows, dist_rows, loss_rows, checks = [], [], [], []
    for lower in config.SENSITIVITY_LOWER_DAYS:
        checkpoint = config.PROCESSED / f'models/SASRec_e2e_l{lower}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['epoch'] == 10
        encoder = SASRecEncoder(num_items=len(items), **PARAMETERS)
        encoder._encoder.use_nested_tensor = False
        model = DirectRelistening(encoder)
        model.load_state_dict(saved['model'])
        del saved
        model.to(device_name()).eval()
        scale = float(F.softplus(model.log_scale).detach().cpu())
        bias = float(model.bias.detach().cpu())
        loss = pl.read_csv(config.RESULTS / f'sasrec_e2e_training_l{lower}.csv')
        assert loss['epoch'].to_list() == list(range(1, 11))
        loss_rows.extend({'lower_days': lower, **r} for r in loss.to_dicts())
        cohort = with_labels(pairs, lower)
        for split in ('train', 'test'):
            evaluation = cohort.filter(pl.col('split') == split)
            loader = DataLoader(PairDataset(evaluation, histories, items), batch_size=128,
                                shuffle=False, num_workers=0, collate_fn=_collate)
            columns = {k: [] for k in ('pair_id', 'known', 'dot', 'score', 'logit', 'probability')}
            with torch.inference_mode():
                for step, (ids, sequences, targets, labels) in enumerate(loader):
                    inputs = _batch_inputs(sequences, device_name())
                    encoded, mask = model.encoder._apply_sequential_encoder(inputs['item.ids'], inputs['item.length'])
                    vectors = model.encoder._get_last_embedding(encoded, mask)
                    target = torch.tensor(targets, dtype=torch.long, device=device_name())
                    dot = (vectors * model.encoder.item_embeddings(target)).sum(dim=1)
                    score = torch.where(target > 0, torch.tanh(dot), torch.full_like(dot, -1.000001))
                    logit = F.softplus(model.log_scale) * score + model.bias
                    probability = logit.sigmoid()
                    columns['pair_id'].extend(ids)
                    for key, value in [('known', target > 0), ('dot', dot), ('score', score),
                                       ('logit', logit), ('probability', probability)]:
                        columns[key].extend(value.cpu().numpy())
                    if step % 1000 == 0:
                        print(f'L={lower} {split}: {step * 128}/{len(evaluation)}', flush=True)
            frame = evaluation.select('pair_id', 'uid', 'item_id', 'anchor_timestamp', 'label_c').with_columns(
                *[pl.Series(k, v) for k, v in columns.items() if k != 'pair_id'])
            assert columns['pair_id'] == evaluation['pair_id'].to_list()
            frame.write_parquet(OUT / f'scores_{split}_l{lower}.parquet')
            for subset, f in [('all', frame), ('known', frame.filter(pl.col('known')))]:
                for kind in ('score', 'probability'):
                    m, user = evaluate_ranking(f['uid'], f['label_c'], f[kind].to_numpy())
                    metrics_rows.append({'lower_days': lower, 'split': split, 'subset': subset, 'score_kind': kind, **m})
                    user.write_parquet(OUT / f'user_auc_{split}_{subset}_{kind}_l{lower}.parquet')
                for kind in ('score', 'probability', 'logit'):
                    dist_rows.append({'lower_days': lower, 'split': split, 'subset': subset, 'kind': kind,
                                      **distribution(f[kind].to_numpy())})
            known = frame.filter(pl.col('known'))
            unknown = frame.filter(~pl.col('known'))
            check = {'lower_days': lower, 'split': split, 'pairs': len(frame), 'known_pairs': len(known),
                     'floor_pairs': len(unknown), 'floor_fraction': len(unknown) / len(frame),
                     'floor_positive_rate': unknown['label_c'].mean(), 'known_positive_rate': known['label_c'].mean(),
                     'known_score_minus_one_fraction': float((known['score'].to_numpy() == -1).mean()),
                     'known_score_plus_one_fraction': float((known['score'].to_numpy() == 1).mean()),
                     'scale': scale, 'bias': bias,
                     'final_eval_bce': float(F.binary_cross_entropy_with_logits(
                         torch.tensor(frame['logit'].to_numpy()), torch.tensor(frame['label_c'].to_numpy(), dtype=torch.float32)))}
            assert np.all(unknown['score'].to_numpy() == np.float32(-1.000001))
            if split == 'test':
                old = pl.read_parquet(config.PROCESSED / f'SASRec_e2e_test_l{lower}.parquet')
                assert old.select('pair_id', 'uid', 'label_c').equals(frame.select('pair_id', 'uid', 'label_c'))
                diff = np.abs(old['probability'].to_numpy() - frame['probability'].to_numpy())
                check.update(saved_probability_max_abs_difference=float(diff.max()),
                             saved_probability_exact_fraction=float((diff == 0).mean()))
                assert float(diff.max()) < 1e-6
            checks.append(check)
            print(json.dumps(check), flush=True)
            pl.DataFrame(metrics_rows).write_csv(OUT / 'metrics.csv')
            pl.DataFrame(dist_rows).write_csv(OUT / 'score_distributions.csv')
            pl.DataFrame(checks).write_csv(OUT / 'cohort_checks.csv')
        del model, encoder
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    pl.DataFrame(loss_rows).write_csv(OUT / 'loss_by_epoch.csv')
    after = protected()
    assert before == after, 'Protected input changed'
    (OUT / 'verification.json').write_text(json.dumps({'protected_files': len(before), 'protected_unchanged': True,
        'device': device_name(), 'seconds': time.monotonic() - begin, 'training_performed': False,
        'model_epoch': 10, 'bootstrap_replicates': 1000, 'bootstrap_seed': config.SEED,
        'checkpoint_hashes': {p: h for p, h in before.items() if 'SASRec_e2e_l' in p}}, indent=2))
    print('DONE; all protected hashes unchanged', flush=True)


if __name__ == '__main__':
    main()
