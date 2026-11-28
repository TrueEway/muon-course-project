"""Official CIFAR-10 binary archive; checksum before parsing, no pickle."""
import hashlib
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from .common import ROOT, write_json

URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz'
MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'


def prepare_cifar():
    folder = ROOT / 'data' / 'cifar10'
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'cifar-10-binary.tar.gz'
    if not archive.exists():
        part = archive.with_suffix('.part')
        request = urllib.request.Request(URL, headers={'User-Agent': 'muon-course-project'})
        with urllib.request.urlopen(request, timeout=60) as response, part.open('wb') as target:
            total, announced = 0, 0
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                total += len(chunk)
                if total - announced >= 16 * 1024 * 1024:
                    print(f'CIFAR-10 downloaded {total // (1024*1024)} MiB', flush=True)
                    announced = total
        if hashlib.md5(part.read_bytes()).hexdigest() != MD5:
            raise RuntimeError('Official CIFAR-10 archive checksum mismatch')
        part.replace(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != MD5:
        raise RuntimeError('CIFAR-10 archive was modified')
    train_x, train_y = [], []
    with tarfile.open(archive, 'r:gz') as tar:
        for split in [f'data_batch_{i}' for i in range(1,6)] + ['test_batch']:
            member = tar.getmember('cifar-10-batches-bin/' + split + '.bin')
            raw = tar.extractfile(member).read()
            rows = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3073)
            pixels, labels = rows[:, 1:].copy().reshape(-1, 3, 32, 32), rows[:, 0].astype(np.int64)
            if split == 'test_batch':
                test_x, test_y = pixels, labels
            else:
                train_x.append(pixels)
                train_y.append(labels)
    train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
    generator = np.random.default_rng(20261128)
    training, validation = [], []
    for label in range(10):
        indices = generator.permutation(np.flatnonzero(train_y == label))
        validation.extend(indices[:500])
        training.extend(indices[500:])
    training, validation = np.asarray(sorted(training)), np.asarray(sorted(validation))
    arrays = {'train': (train_x[training], train_y[training]),
              'val': (train_x[validation], train_y[validation]), 'test': (test_x, test_y)}
    metadata = dict(url=URL, official_md5=MD5, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        train=45000, validation=5000, test=10000, split_seed=20261128,
        training_indices_sha256=hashlib.sha256(training.astype('<i8').tobytes()).hexdigest(),
        validation_indices_sha256=hashlib.sha256(validation.astype('<i8').tobytes()).hexdigest(),
        normalization_mean=[0.4914,0.4822,0.4465], normalization_std=[0.2470,0.2435,0.2616])
    write_json(folder / 'manifest.json', metadata)
    return {key: (torch.from_numpy(x), torch.from_numpy(y)) for key, (x,y) in arrays.items()}, metadata


if __name__ == '__main__':
    arrays, meta = prepare_cifar()
    print({key: len(value[1]) for key, value in arrays.items()}, flush=True)
