"""Entrena la libreta con varias semillas, la evalúa y guarda los resultados.

Uso (desde la carpeta principal del repositorio):
    python scripts/entrenar_q.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rlstock.config import cargar_config, RAIZ
from rlstock.discretizador import Discretizador
from rlstock.entorno import crear_entorno
from rlstock.entrenamiento import entrenar, PoliticaLibreta
from rlstock.evaluacion import evaluar

config = cargar_config()
ini = config["semillas"]["evaluacion_inicio"]
n = config["semillas"]["episodios_evaluacion"]
hp = config["q_learning"]

env = crear_entorno(config, discreto=True)
disc = Discretizador(config["tramos"])
(RAIZ / "results" / "models").mkdir(parents=True, exist_ok=True)

evaluaciones, entrenamientos, medias = [], [], []
for semilla in config["semillas"]["entrenamiento"]:
    agente, historial = entrenar(config, hp["alfa"], hp["gamma"], hp["eps_inicial"],
                                 hp["eps_final"], hp["decaimiento"], hp["episodios"], semilla)
    r = evaluar(env, PoliticaLibreta(agente, disc), ini, n)   # sin explorar y sin aprender
    medias.append(r.mean())
    print(f"semilla {semilla}: media {r.mean():,.0f}  desviación {r.std():,.0f}")
    for k, b in enumerate(r):
        evaluaciones.append({"semilla": semilla, "episodio_eval": ini + k, "beneficio": b})
    for ep, b in enumerate(historial):
        entrenamientos.append({"semilla": semilla, "episodio": ep, "beneficio": b})
    np.save(RAIZ / "results" / "models" / f"q_semilla{semilla}.npy", agente.q)

print(f"\nENTRE SEMILLAS: media {np.mean(medias):,.0f}  desviación {np.std(medias):,.0f}")
pd.DataFrame(evaluaciones).to_csv(RAIZ / "results" / "q_evaluacion.csv", index=False)
pd.DataFrame(entrenamientos).to_csv(RAIZ / "results" / "q_entrenamiento.csv", index=False)
print("Guardado en results/ (q_evaluacion.csv, q_entrenamiento.csv y models/)")