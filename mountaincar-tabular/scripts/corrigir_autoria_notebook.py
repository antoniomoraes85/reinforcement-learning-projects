"""Troca apenas quatro cadeias no Markdown inicial; mantém código e resultados intactos."""
import json
from pathlib import Path

file=Path("mountaincar-tabular/entrega_final/CMP629_MOUNTAINCAR_ENTREGA_FINAL.ipynb")
before=file.read_bytes()
old=[
    "JOSE ANTONIO COUTINHO DE MORAES FILHO",
    "FERNANDO BOSQUETTI TEDESCO",
    "GABRIEL DE SENNE AMORIM",
    "CAETANO GUBERT MENEGHETTI",
]
new=[
    "Caetano gubert meneghetti",
    "Fernando bosquetti tedesco",
    "Gabriel de senne amorim",
    "José antônio coutinho de moraes filho",
]
data=before
for old_name,new_name in zip(old,new):
    source=old_name.encode("utf-8")
    assert data.count(source)>=1, f"Nome original nao encontrado: {old_name}"
    data=data.replace(source,new_name.encode("utf-8"),1)
nb_before=json.loads(before)
nb_after=json.loads(data)
assert len(nb_before["cells"])==len(nb_after["cells"])
assert nb_before["cells"][1:]==nb_after["cells"][1:],"Alteracao cientifica indevida no notebook."
src="".join(nb_after["cells"][0]["source"])
positions=[src.find(s) for s in new]
assert all(v>=0 for v in positions) and positions==sorted(positions)
assert len(nb_after["cells"])==31
file.write_bytes(data)
print("NOTEBOOK_AUTORIA_CORRIGIDA",file,"old_bytes",len(before),"new_bytes",len(data))
print("VALIDACAO",len(nb_after["cells"]),"celulas; celulas 1..30 identicas byte a byte")
