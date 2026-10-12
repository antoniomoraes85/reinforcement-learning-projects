"""Create once / verify SHA-256 manifest. Never writes historical results."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'relatorios' / 'manifesto_historico_sha256.json'


def inventory():
    files = sorted((ROOT / 'resultados' / 'triagem').rglob('*'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def main():
    current = inventory()
    count = len(list((ROOT / 'resultados' / 'triagem').glob('*__*__s*__*.json')))
    if count != 108:
        raise RuntimeError(f'Expected 108 individual historical JSONs, found {count}')
    if TARGET.exists():
        assert json.loads(TARGET.read_text(encoding='utf-8'))['files'] == current, 'Historical files changed'
        print(f'Preserved: {count} runs; {len(current)} historical files; all hashes match')
    else:
        with TARGET.open('x', encoding='utf-8') as stream:
            json.dump({'individual_runs': count, 'files': current}, stream, indent=2)
        print(f'Created baseline: {count} runs; {len(current)} files')


if __name__ == '__main__':
    main()
