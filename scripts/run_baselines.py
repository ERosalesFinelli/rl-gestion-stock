"""Evalúa los dos agentes de referencia (azar y base-stock) y guarda los resultados en CSV.

Uso (desde la carpeta principal del repositorio):
    python scripts/run_baselines.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from rlstock.config import cargar_config, RAIZ
from rlstock.entorno import crear_entorno
from rlstock.baselines import AgenteAzar, BaseStock
from rlstock.evaluacion import evaluar

config = cargar_config()
ini = config["semillas"]["evaluacion_inicio"]
n = config["semillas"]["episodios_evaluacion"]

filas = []
for nombre in ["azar", "base_stock"]:
    # Los baselines usan el entorno original (pedidos continuos).
    env = crear_entorno(config, discreto=False)
    if nombre == "azar":
        agente = AgenteAzar(env, semilla=ini)
    else:
        agente = BaseStock(env, config["baselines"]["base_stock"]["niveles_objetivo"])
    beneficios = evaluar(env, agente, ini, n)
    print(f"{nombre:>10}: media {beneficios.mean():,.0f}  desviación {beneficios.std():,.0f}  ({n} episodios)")
    for k, b in enumerate(beneficios):
        filas.append({"agente": nombre, "semilla_episodio": ini + k, "beneficio": b})

salida = RAIZ / "results" / "baselines.csv"
pd.DataFrame(filas).to_csv(salida, index=False)
print("Guardado en", salida)
