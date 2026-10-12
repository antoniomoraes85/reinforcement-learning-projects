"""Map local training commit to exact source bytes without publishing private Git history."""
from pathlib import Path
import subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[1]


def main():
    files={}
    training_commit='f31e7b5'
    for path in sorted((ROOT/'src/mountaincar_rl').glob('*.py')):
        name=path.relative_to(ROOT).as_posix()
        committed=subprocess.check_output(['git','show',f'{training_commit}:{name}'],cwd=ROOT)
        actual=path.read_bytes()
        # Git baseline may use LF while checkout uses CRLF; Python semantics unchanged.
        assert actual.replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'),name
        files[name]=hashlib.sha256(actual).hexdigest()
    report={'training_commit_local':subprocess.check_output(['git','rev-parse',training_commit],cwd=ROOT,text=True).strip(),
            'source_sha256':files,'note':'Local Git root differs from public monorepo. Exact module hashes map the recorded local commit to public source bytes. No training module changed during the campaign.'}
    (ROOT/'relatorios/proveniencia_codigo.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('All training modules match frozen commit; exact-byte source manifest saved')


if __name__=='__main__':main()
