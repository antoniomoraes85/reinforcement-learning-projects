"""Evidence-based phase-A gate. Does not run training."""
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from preservar_historico import inventory,TARGET
from mountaincar_rl.config import read_json,expand_final,config_hash
from mountaincar_rl.storage import atomic_write_json,inspect_result_file
from mountaincar_rl.checkpoints import load_checkpoint

def main():
    suites=ET.parse(ROOT/'relatorios/testes_fase_a.xml').getroot()
    tests=sum(int(s.get('tests',0)) for s in suites.iter('testsuite'))
    errors=sum(int(s.get('errors',0))+int(s.get('failures',0)) for s in suites.iter('testsuite'))
    assert tests>=25 and errors==0
    assert read_json(TARGET)['files']==inventory()
    assert read_json(ROOT/'relatorios/validacao_triagem_fase_a.json')['valid']
    smoke=read_json(ROOT/'relatorios/smoke_fase_a.json')
    for receipt in smoke['receipts']:
        path=Path(receipt['path']); assert inspect_result_file(path)[0]
        r=read_json(path); cp=r['checkpoint']; load_checkpoint(path.parent/cp['file'],cp['sha256'],r['config'])
    protocol=read_json(ROOT/'configs/protocolo_final_v2.json')
    selection=read_json(ROOT/'resultados/triagem/hiperparametros_selecionados.json')
    assert len(expand_final(protocol,selection))==60
    assert set(protocol['seeds']).isdisjoint([100,1100,2100,700001])
    atomic_write_json(ROOT/'relatorios/gate_fase_a.json',{'decision':'GO','tests':tests,'failures':errors,
        'historical_runs':108,'smoke_runs':6,'protocol_sha256':config_hash(protocol),
        'scope':'Technical readiness for authorized final campaign; not scientific conclusions or hosted Colab certification'})
    report=f'''# Parecer de liberação final — FASE A

Data: 11/10/2026. **GO técnico para a campanha final v2.** Autorização explícita recebida na ordem de finalização integral. Não é aprovação acadêmica nem conclusão sobre superioridade de métodos.

## Evidências executadas
- Suíte pytest: **{tests} testes aprovados, {errors} falhas** (`testes_fase_a.xml`).
- Validador: 108/108 JSONs da triagem; zero ausentes/corrompidos. Manifesto SHA-256: 121 arquivos históricos, preservados byte a byte.
- Smoke: seis configurações (três métodos × duas discretizações), cinco episódios cada, dois workers; **{smoke['wall_seconds']:.3f} s** wall-clock. Não são seis execuções finais.
- Seis checkpoints NPZ carregados e hashes conferidos; CSVs/JSONs gravados separadamente. Cache completo testado sem retreino; parcial inválido preservado e rejeitado.
- Ambiente, limites e seeds; TD manual Q/SARSA/Dyna; traços acumulativos e reset; terminal versus truncamento; modelo empírico 70/30 com seed fixa; Dyna N=0 equivalente a Q-learning; reprodução seed-a-seed; política greedy e desempates; avaliação não altera Q.

## Correções
Checkpoints verificáveis, avaliação v2 com 100 seeds comuns fixas, logs de retorno/término/epsilon, separação de TD real/planejamento/componentes dos traços e latência. Dyna já armazenava múltiplos desfechos: nenhum bug de sobrescrita foi inventado. Escalas e nomenclatura corrigidas nas figuras novas, sem alterar CSVs antigos. `total_q_updates` mantido apenas por compatibilidade; não é custo homogêneo entre algoritmos.

## Decisão metodológica
Bootstrap em truncamento temporal; zero somente em terminal real. Trata-se do processo subjacente contínuo com avaliação limitada a 200 passos, não de solução exata de horizonte finito com tempo no estado. Traços zerados a cada episódio. Avaliação histórica e v2 têm esquemas distintos de seeds: não misturar campanhas como replicações idênticas.

## Estimativas medidas, não garantias
Smoke ocupa {smoke['artifact_bytes']} bytes (sem vídeos). Extrapolação linear conservadora: {smoke['conservative_storage_bytes_linear_extrapolation']/1e6:.1f} MB; treino serial {smoke['rough_final_serial_training_seconds']/60:.1f} min ou cerca de {smoke['rough_final_two_worker_seconds']/60:.1f} min com dois workers, mais avaliação/inicialização. Cinco episódios são insuficientes para previsão precisa; convergência/carga variam. RAM será observada durante a campanha; não foi medida no primeiro smoke.

## Ressalvas não bloqueantes
Warnings de depreciação do Pyparsing/Matplotlib, sem falhas. Só três seeds por célula na triagem; não alegar superioridade geral. Não existem checkpoints históricos da triagem. Vídeos do smoke não representam desempenho final. Colab hospedado e publicação ainda em preparação na emissão deste gate.

## Comandos efetivamente executados
```powershell
& '..\\.venv\\Scripts\\python.exe' -B -m pytest -p no:cacheprovider -q --junitxml=relatorios/testes_fase_a.xml
& '..\\.venv\\Scripts\\python.exe' -B scripts/preservar_historico.py
& '..\\.venv\\Scripts\\python.exe' -B scripts/preparar_fase_a.py
& '..\\.venv\\Scripts\\python.exe' -B scripts/validar_resultados.py --fase triagem --relatorio relatorios/validacao_triagem_fase_a.json
& '..\\.venv\\Scripts\\python.exe' -B scripts/gerar_figuras_auditadas.py
```

Ambiente testado: Python 3.12.14, Gymnasium 1.2.1, NumPy 2.3.3. A `.venv` interna (Python 3.14) não foi adotada; usamos a `.venv` do diretório pai, sem apagar nenhuma delas.
'''
    (ROOT/'relatorios/parecer_liberacao_final.md').write_text(report,encoding='utf-8')
    print('GO:',tests,'tests; 108 historical runs preserved; 6 verified smoke bundles')

if __name__=='__main__':main()
