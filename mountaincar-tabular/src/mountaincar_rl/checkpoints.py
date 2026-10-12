"""Safe, non-pickle Q-table bundles linked to a full configuration."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from .config import canonical_json, config_hash


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_checkpoint(path, q, config):
    path = Path(path)
    if q.shape != tuple(config['discretization']['shape']) + (3,) or not np.isfinite(q).all():
        raise ValueError('Invalid Q table')
    with path.open('xb') as stream:
        np.savez_compressed(stream, q=q, config=np.array(canonical_json(config)),
                            config_hash=np.array(config_hash(config)))
    return file_hash(path)


def load_checkpoint(path, expected_hash=None, expected_config=None):
    if expected_hash is not None and file_hash(path) != expected_hash:
        raise ValueError('Checkpoint SHA-256 mismatch')
    with np.load(path, allow_pickle=False) as bundle:
        q = bundle['q'].copy()
        config = json.loads(str(bundle['config']))
        if str(bundle['config_hash']) != config_hash(config):
            raise ValueError('Checkpoint config hash mismatch')
    if expected_config is not None and config_hash(config) != config_hash(expected_config):
        raise ValueError('Checkpoint belongs to another run')
    if q.shape != tuple(config['discretization']['shape']) + (3,) or not np.isfinite(q).all():
        raise ValueError('Invalid saved Q table')
    return q, config
