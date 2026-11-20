from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ROOT, completed, fingerprint, write_csv, write_json
from .optim import MatrixMuon

DATA_COMMIT = '370cbcd448eb7daf32f21a6be560b70e0b33c4e3'
DATA_URL = f'https://raw.githubusercontent.com/karpathy/char-rnn/{DATA_COMMIT}/data/tinyshakespeare/input.txt'
MODEL_CONFIG = dict(layers=2, width=64, heads=4, ff_width=256, context=64, batch_size=16, dropout=0.0)


def prepare_data():
    path = ROOT / 'data' / 'input.txt'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'muon-course-project'})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    metadata_path = ROOT / 'data' / 'manifest.json'
    if metadata_path.exists():
        expected = json.loads(metadata_path.read_text(encoding='utf-8'))['sha256']
        if checksum != expected:
            raise RuntimeError('Dataset SHA256 mismatch; do not train with modified data')
    text = raw.decode('utf-8')
    chars = sorted(set(text))
    mapping = {c: i for i, c in enumerate(chars)}
    tokens = torch.tensor([mapping[c] for c in text], dtype=torch.long)
    n_train, n_val = int(len(tokens) * 0.9), int(len(tokens) * 0.05)
    splits = {'train': tokens[:n_train], 'val': tokens[n_train:n_train + n_val], 'test': tokens[n_train + n_val:]}
    metadata = dict(source=DATA_URL, commit=DATA_COMMIT, sha256=checksum, bytes=len(raw),
                    characters=len(text), vocabulary=chars, vocabulary_size=len(chars),
                    split_sizes={k: len(v) for k, v in splits.items()}, split_policy='contiguous 90/5/5; no window crosses a boundary')
    write_json(metadata_path, metadata)
    return splits, metadata


class Attention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.view(batch, length, self.heads, width // self.heads).transpose(1, 2) for z in (q, k, v)]
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.proj(result.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.attention = Attention()
        self.norm2 = nn.LayerNorm(64)
        self.mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, 64)
        self.position_embedding = nn.Embedding(64, 64)
        self.blocks = nn.Sequential(Block(), Block())
        self.final_norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, vocab_size, bias=False)
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.head(self.final_norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def parameter_groups(model, method):
    muon, decay, nodecay = [], [], []
    named = {'muon': [], 'adamw_decay': [], 'adamw_nodecay': []}
    for name, param in model.named_parameters():
        if method != 'adamw' and name.startswith('blocks.') and param.ndim == 2:
            muon.append(param)
            named['muon'].append(name)
        elif param.ndim >= 2:
            decay.append(param)
            named['adamw_decay'].append(name)
        else:
            nodecay.append(param)
            named['adamw_nodecay'].append(name)
    return muon, decay, nodecay, named


def make_optimizers(model, method, lr, aux_lr):
    muon, decay, nodecay, named = parameter_groups(model, method)
    adam_lr = lr if method == 'adamw' else aux_lr
    adam = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=adam_lr, betas=(0.9, 0.95), eps=1e-8, foreach=False)
    optimizers = [(adam, adam_lr)]
    if muon:
        optimizers.append((MatrixMuon(muon, lr=lr, mode='svd' if method == 'muon_svd' else 'ns'), lr))
    return optimizers, named


def batch_from_starts(tokens, starts, context=64):
    positions = starts[:, None] + torch.arange(context + 1)[None, :]
    chunk = tokens[positions]
    return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


@torch.no_grad()
def evaluate(model, tokens, full=False):
    model.eval()
    if full:
        # Each target character, except the first one, is scored once. The last
        # short window is included; positions reset within every window.
        total, count = 0.0, 0
        for start in range(0, len(tokens) - 1, 64):
            length = min(64, len(tokens) - start - 1)
            x = tokens[start:start + length].unsqueeze(0)
            y = tokens[start + 1:start + length + 1].unsqueeze(0)
            _, loss = model(x, y)
            total += float(loss) * length
            count += length
    else:
        generator = torch.Generator().manual_seed(123456)
        starts = torch.randint(0, len(tokens) - 64, (32, 16), generator=generator)
        total, count = 0.0, 0
        for s in starts:
            x, y = batch_from_starts(tokens, s)
            _, loss = model(x, y)
            total += float(loss)
            count += 1
    model.train()
    return total / count


def lr_factor(step, horizon):
    if step <= 100:
        return step / 100
    progress = (step - 100) / max(1, horizon - 100)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def train_once(out, splits, metadata, method, seed, lr, aux_lr, horizon, pilot, overwrite=False):
    category = 'pilot' if pilot else 'formal'
    stem = f'{method}_s{seed}_lr{lr:g}_T{horizon}'
    result_path = out / 'language' / category / (stem + '.json')
    config = dict(suite='language', method=method, seed=seed, lr=lr, aux_lr=aux_lr,
                  horizon=horizon, pilot=pilot, model=MODEL_CONFIG, dataset_sha256=metadata['sha256'],
                  weight_decay=0.01, adam_betas=[0.9, 0.95], adam_eps=1e-8, momentum=0.95,
                  ns_steps=5, nesterov=True, shape_scale='sqrt(max(1,rows/cols))',
                  dtype='float32', device='cpu', torch_threads=torch.get_num_threads(),
                  warmup_steps=100, min_lr_factor=0.1, evaluation_interval=50,
                  validation_batches=32, validation_seed=123456, gradient_clipping=None)
    previous = completed(result_path, config, overwrite)
    if previous:
        return previous
    torch.manual_seed(seed)
    model = TinyTransformer(metadata['vocabulary_size'])
    optimizer_list, named = make_optimizers(model, method, lr, aux_lr)
    generator = torch.Generator().manual_seed(seed + 5000)
    initial_val = evaluate(model, splits['val'])
    best_val = initial_val
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    logs = [dict(step=0, train_loss=None, val_loss=initial_val, train_seconds=0.0, tokens=0, lr_factor=0.0)]
    train_seconds = 0.0
    wall_start = time.perf_counter()
    for step in range(1, horizon + 1):
        start = time.perf_counter()
        starts = torch.randint(0, len(splits['train']) - 64, (16,), generator=generator)
        x, y = batch_from_starts(splits['train'], starts)
        factor = lr_factor(step, horizon)
        for optimizer, base_lr in optimizer_list:
            for group in optimizer.param_groups:
                group['lr'] = base_lr * factor
            optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite training loss: {stem} at {step}')
        loss.backward()
        for optimizer, _ in optimizer_list:
            optimizer.step()
        train_seconds += time.perf_counter() - start
        val_loss = None
        if step % 50 == 0 or step == horizon:
            val_loss = evaluate(model, splits['val'])
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                best_state = copy.deepcopy(model.state_dict())
        logs.append(dict(step=step, train_loss=float(loss.detach()), val_loss=val_loss,
                         train_seconds=train_seconds, tokens=step * 16 * 64, lr_factor=factor))
        if step % 300 == 0 or step == horizon:
            print(f'{category} {stem} step={step} train={float(loss.detach()):.4f} val={val_loss:.4f} train_s={train_seconds:.1f}', flush=True)
    wall_seconds = time.perf_counter() - wall_start
    test_ce = None
    if not pilot:
        model.load_state_dict(best_state)
        test_ce = evaluate(model, splits['test'], full=True)
        checkpoint = out / 'checkpoints' / (stem + '.pt')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=best_state, config=config, best_step=best_step,
                        dataset_sha256=metadata['sha256']), checkpoint)
    result = dict(config=config, config_hash=fingerprint(config), parameter_groups=named,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  initial_val_loss=initial_val, final_val_loss=logs[-1]['val_loss'],
                  best_val_loss=best_val, best_step=best_step,
                  final_train_loss_mean50=sum(x['train_loss'] for x in logs[-50:]) / min(50, horizon),
                  test_cross_entropy=test_ce, test_perplexity=None if test_ce is None else math.exp(test_ce),
                  train_seconds=train_seconds, wall_seconds_excluding_initial_eval_and_test=wall_seconds,
                  tokens_per_second=horizon * 16 * 64 / train_seconds)
    write_csv(result_path.with_suffix('.csv'), logs)
    write_json(result_path, result)
    return result


def run_language(out: Path, seeds: list[int], overwrite=False):
    splits, metadata = prepare_data()
    baseline_candidates = [3e-4, 1e-3, 3e-3]
    muon_candidates = [0.005, 0.02, 0.05]
    pilot_results = []
    for lr in baseline_candidates:
        pilot_results.append(train_once(out, splits, metadata, 'adamw', 42, lr, lr, 300, True, overwrite))
    adam_best = min(pilot_results, key=lambda x: x['final_val_loss'])['config']['lr']
    chosen = {'adamw': adam_best}
    for method in ['muon_svd', 'muon_ns']:
        candidates = [train_once(out, splits, metadata, method, 42, lr, adam_best, 300, True, overwrite)
                      for lr in muon_candidates]
        pilot_results.extend(candidates)
        chosen[method] = min(candidates, key=lambda x: x['final_val_loss'])['config']['lr']
    write_json(out / 'language' / 'selection.json', dict(selection_metric='final validation CE after 300 pilot steps',
        seed=42, selected_lrs=chosen, auxiliary_adamw_lr=adam_best,
        candidates=[dict(method=x['config']['method'], lr=x['config']['lr'], val_loss=x['final_val_loss']) for x in pilot_results]))
    # Rotate method order across seeds to reduce a fixed order/timing confound.
    methods = ['adamw', 'muon_svd', 'muon_ns']
    for index, seed in enumerate(seeds):
        for method in methods[index % 3:] + methods[:index % 3]:
            train_once(out, splits, metadata, method, seed, chosen[method], adam_best, 1500, False, overwrite)


def profile_language(out: Path):
    splits, metadata = prepare_data()
    rows = []
    for threads in [1, 2, 4]:
        torch.set_num_threads(threads)
        for method in ['adamw', 'muon_svd', 'muon_ns']:
            torch.manual_seed(7)
            model = TinyTransformer(metadata['vocabulary_size'])
            optimizers, _ = make_optimizers(model, method, 0.001 if method == 'adamw' else 0.02, 0.001)
            x, y = batch_from_starts(splits['train'], torch.arange(16) * 64)
            durations = []
            for step in range(105):
                start = time.perf_counter()
                for optimizer, _ in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                for optimizer, _ in optimizers:
                    optimizer.step()
                if step >= 5:
                    durations.append(time.perf_counter() - start)
            rows.append(dict(threads=threads, method=method, seconds_per_step=sum(durations) / len(durations)))
            print(rows[-1], flush=True)
    write_csv(out / 'language' / 'profile.csv', rows)
