"""Restore public scientific bundles without overwriting differing existing files."""
from pathlib import Path,PurePosixPath
import hashlib,json,urllib.request,zipfile,io,argparse
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--ref',default='codex/cmp629-fase-a-20261011');args=p.parse_args()
    base=f'https://raw.githubusercontent.com/antoniomoraes85/reinforcement-learning-projects/{args.ref}/mountaincar-tabular/'
    def get(path):
        with urllib.request.urlopen(base+path,timeout=90) as r:return r.read()
    manifest=json.loads(get('dados/auditoria_v2/manifesto_publicacao.json'))
    for archive,folder in [('triagem_originais.zip','triagem'),('final_v2_bundles.zip','cmp629-final-v2-20261011')]:
        relative='dados/auditoria_v2/'+archive
        if relative not in manifest:continue
        data=get(relative); assert hashlib.sha256(data).hexdigest()==manifest[relative]
        target=(ROOT/'resultados'/folder).resolve(); target.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                name=PurePosixPath(info.filename)
                if name.is_absolute() or len(name.parts)!=1 or ':' in info.filename or '\\' in info.filename:
                    raise ValueError('Unexpected archive path')
                path=target/info.filename; content=z.read(info)
                if path.exists():
                    if path.read_bytes()!=content:raise FileExistsError(f'Existing differing file preserved: {path.name}')
                else:
                    with path.open('xb') as stream:stream.write(content)
        print(archive,'verified and extracted without replacing existing data')


if __name__=='__main__':main()
