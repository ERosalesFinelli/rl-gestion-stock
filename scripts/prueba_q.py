"""Primera prueba de la libreta:  python scripts/prueba_q.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlstock.config import cargar_config
from rlstock.discretizador import Discretizador
from rlstock.entorno import crear_entorno
from rlstock.baselines import AgenteAzar
from rlstock.entrenamiento import entrenar, PoliticaLibreta
from rlstock.evaluacion import evaluar

config = cargar_config()
ini = config["semillas"]["evaluacion_inicio"]
n = config["semillas"]["episodios_evaluacion"]

# Entorno con las mismas 125 acciones que usará la libreta
env = crear_entorno(config, discreto=True)

# 1) Azar con las mismas 125 acciones (comparación justa)
r = evaluar(env, AgenteAzar(env, ini), ini, n)
print(f"Azar (125 acciones):  media {r.mean():,.0f}  desviación {r.std():,.0f}")

# 2) Entrenar la libreta y evaluarla aparte, sin explorar ni aprender
agente, historial = entrenar(config, alfa=0.1, gamma=0.99, eps_inicial=1.0,
                             eps_final=0.05, decaimiento=0.999, episodios=3000, semilla=0)
politica = PoliticaLibreta(agente, Discretizador(config["tramos"]))
r = evaluar(env, politica, ini, n)
print(f"Libreta (evaluada):   media {r.mean():,.0f}  desviación {r.std():,.0f}")
print(f"Casillas de la libreta que se han usado: {(agente.q != 0).sum():,} de {agente.q.size:,}")