"""Download and hash all public manifested artifacts; verify real video decoding."""
from pathlib import Path
import json,urllib.request,hashlib,argparse,time,tempfile
from concurrent.futures import ThreadPoolExecutor
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--ref',default='codex/cmp629-fase-a-20261011');args=p.parse_args()
    base=f'https://raw.githubusercontent.com/antoniomoraes85/reinforcement-learning-projects/{args.ref}/mountaincar-tabular/'
    def get(path):
        for attempt in range(3):
            try:
                with urllib.request.urlopen(base+path,timeout=90) as r:return r.read()
            except OSError:
                if attempt==2:raise
                time.sleep(attempt+1)
    manifest=json.loads(get('dados/auditoria_v2/manifesto_publicacao.json'))
    def check(item):
        path,expected=item; content=get(path)
        if hashlib.sha256(content).hexdigest()!=expected:raise ValueError(f'Hash mismatch: {path}')
        return path,len(content)
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(check,manifest.items()))
    videos=[]
    # Check complete decode of final videos (or smoke if final not yet available).
    media=ROOT/'figuras/animacoes_final_v2'
    if not media.exists():media=ROOT/'figuras/animacoes_auditoria_v2'
    for path in media.glob('*.mp4'):
        reader=imageio.get_reader(path); count=0
        for frame in reader:count+=1
        reader.close()
        assert count>0
        videos.append({'file':path.name,'decoded_frames':count})
    result={'ref':args.ref,'verified_files':len(results),'verified_bytes':sum(n for _,n in results),'failures':[],
            'video_decode':videos,'hosted_colab_executed':False}
    (ROOT/'relatorios/verificacao_publicacao.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
