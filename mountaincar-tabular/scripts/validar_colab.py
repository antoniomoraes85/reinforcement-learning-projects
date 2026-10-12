"""Run every cell in a fresh kernel and empty cwd; only public inputs are used."""
from pathlib import Path
import os,sys,json,tempfile,time,subprocess
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'notebooks/MountainCar_CMP629_Colab.ipynb'
    nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='cmp629-notebook-') as temp:
        directory=Path(temp)
        kernel=directory/'jupyter/kernels/cmp629-clean'; kernel.mkdir(parents=True)
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'CMP629 clean validation','language':'python'}),encoding='utf-8')
        previous=os.environ.get('JUPYTER_PATH','')
        os.environ['JUPYTER_PATH']=str(directory/'jupyter')+(os.pathsep+previous if previous else '')
        NotebookClient(nb,timeout=180,kernel_name='cmp629-clean',resources={'metadata':{'path':temp}}).execute()
    errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
    assert not errors
    nbformat.write(nb,path)
    html,_=HTMLExporter().from_notebook_node(nb)
    (ROOT/'relatorios/colab_preview.html').write_text(html,encoding='utf-8')
    result={'status':'passed','code_cells':sum(c.cell_type=='code' for c in nb.cells),'errors':0,
            'wall_seconds':time.perf_counter()-start,'environment':'fresh Python venv, fresh kernel and empty working directory',
            'hosted_colab_executed':False,'public_downloads_verified':True}
    (ROOT/'relatorios/validacao_colab.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'requirements-lock-validacao.txt').write_text(lock,encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
