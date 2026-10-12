"""Stage an allowlisted publication copy, never alter source histories or Monte Carlo."""
from pathlib import Path
import sys,json,gzip,shutil,hashlib,zipfile,subprocess
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'.publication/mountaincar-tabular'
sys.path.insert(0,str(ROOT/'src'))
from mountaincar_rl.config import read_json


def copy(source,relative):
    target=DEST/relative; target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def main():
    assert (ROOT/'.publication/.git').is_dir()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT/'.publication',text=True).strip()
    if branch=='main':raise RuntimeError('Refusing direct publication on main')
    for folder in ['src','scripts','testes','configs','docs','notebooks']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ['.py','.json','.md','.ipynb']:
                copy(path,path.relative_to(ROOT))
    for name in ['requirements.txt','requirements-entrega.txt','requirements-lock-validacao.txt','STATUS_CMP629.md','COMO_EXECUTAR_POWERSHELL.md','README.md']:
        if (ROOT/name).exists():copy(ROOT/name,Path(name))
    # Do not publish local logs / reports containing personal absolute paths.
    for path in (ROOT/'relatorios').rglob('*'):
        if path.suffix in ['.md','.csv','.json'] and path.name not in ['validacao_resultados.json','smoke_fase_a.json','validacao_triagem_fase_a.json','recursos_campanha.json']:
            text=path.read_text(encoding='utf-8')
            if str(ROOT) in text or 'C:\\\\Users\\\\' in text:continue
            copy(path,path.relative_to(ROOT))
    data=DEST/'dados/auditoria_v2'; data.mkdir(parents=True,exist_ok=True)
    histories=[read_json(p) for p in sorted((ROOT/'resultados/triagem').glob('*__*__s*__*.json'))]
    (data/'triagem_historicos.json.gz').write_bytes(gzip.compress(json.dumps(histories,ensure_ascii=False,allow_nan=False).encode('utf-8'),mtime=0))
    copy(ROOT/'figuras/auditoria_v2/selected_seed_metrics.csv',Path('dados/auditoria_v2/selected_seed_metrics.csv'))
    # Compact archives preserve original bytes and include only scientific data.
    for dirname,label in [('triagem','triagem_originais'),('cmp629-final-v2-20261011','final_v2_bundles')]:
        source=ROOT/'resultados'/dirname
        if not source.exists():continue
        if label=='final_v2_bundles' and len(list(source.glob('*__*__s*__*.json')))!=60:continue
        files=[p for p in source.iterdir() if p.is_file() and (p.suffix in ['.json','.npz','.csv']) and not p.name.startswith('ultimo_manifesto')]
        with zipfile.ZipFile(data/(label+'.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(files):archive.write(path,path.name)
    for folder in ['auditoria_v2','animacoes_auditoria_v2','final_v2','animacoes_final_v2']:
        for path in (ROOT/'figuras'/folder).glob('*'):
            if path.is_file() and path.suffix in ['.png','.svg','.pdf','.mp4','.gif','.json','.csv']:
                copy(path,Path('figuras')/folder/path.name)
    demo=next((ROOT/'resultados/auditoria_v2').glob('q_learning__12x4*.npz'))
    copy(demo,Path('dados/auditoria_v2/demo_checkpoint.npz'))
    final_dir=ROOT/'resultados/cmp629-final-v2-20261011'
    final_results=[read_json(p) for p in sorted(final_dir.glob('*__*__s*__*.json'))]
    if len(final_results)==60:
        (data/'final_historicos.json.gz').write_bytes(gzip.compress(json.dumps(final_results,ensure_ascii=False,allow_nan=False).encode('utf-8'),mtime=0))
    manifest={}
    for path in DEST.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='manifesto_publicacao.json':
            manifest[path.relative_to(DEST).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (data/'manifesto_publicacao.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'{len(manifest)} public artifacts staged; source histories untouched')


if __name__=='__main__':main()
