"""Mira qué valores toman los 4 datos que ve la libreta (para decidir los tramos).

Los 4 datos son: stock de la tienda, stock del distribuidor, stock de la fábrica
y total de pedidos en camino. Se juegan 50 episodios con dos encargados:
  - base-stock (el bueno)
  - azar usando los 5 niveles de pedido (lo que verá la IA al empezar a explorar)

Uso (desde la carpeta principal del repositorio):
    python scripts/explorar_estados.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rlstock.config import cargar_config
from rlstock.entorno import crear_entorno
from rlstock.baselines import BaseStock

config = cargar_config()
INI = config["semillas"]["evaluacion_inicio"]
EPISODIOS = 50


def cuatro_datos(obs):
    """De los 15 números del parte, saca los 4 que usará la libreta."""
    stock_tienda, stock_distrib, stock_fabrica = obs[2], obs[3], obs[4]
    en_camino = obs[5:13].sum()          # los 8 números de pedidos en camino
    return [stock_tienda, stock_distrib, stock_fabrica, en_camino]


def recoger(nombre):
    datos = []
    if nombre == "base-stock":
        env = crear_entorno(config, discreto=False)
        agente = BaseStock(env, config["baselines"]["base_stock"]["niveles_objetivo"])
        elegir = lambda obs: agente.actuar(obs)
    else:
        env = crear_entorno(config, discreto=True)
        env.action_space.seed(INI)
        elegir = lambda obs: env.action_space.sample()
    for k in range(EPISODIOS):
        obs, _ = env.reset(seed=INI + k)
        fin = False
        while not fin:
            datos.append(cuatro_datos(obs))
            obs, _, terminado, truncado, _ = env.step(elegir(obs))
            fin = terminado or truncado
    return np.array(datos)


etiquetas = ["stock tienda", "stock distribuidor", "stock fábrica", "pedidos en camino (total)"]
percentiles = [0, 5, 25, 50, 75, 95, 100]
for nombre in ["base-stock", "azar (5 niveles)"]:
    d = recoger("base-stock" if nombre == "base-stock" else "azar")
    print(f"\n=== {nombre}: {len(d)} días jugados ===")
    print(f"{'dato':<28}" + "".join(f"{('p' + str(p)):>9}" for p in percentiles))
    for j, et in enumerate(etiquetas):
        print(f"{et:<28}" + "".join(f"{np.percentile(d[:, j], p):>9.0f}" for p in percentiles))
