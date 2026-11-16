"""Pruebas rápidas: python -m pytest  (o ejecutarlo con: python tests/test_entorno.py)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rlstock.config import cargar_config
from rlstock.entorno import crear_entorno


def test_acciones_discretas():
    config = cargar_config()
    env = crear_entorno(config, discreto=True)
    assert env.action_space.n == 125
    # la acción 62 es (nivel 2, nivel 2, nivel 2) = (20, 20, 20)
    assert np.allclose(env.action(62), [20, 20, 20])
    assert np.allclose(env.action(0), [0, 0, 0])
    assert np.allclose(env.action(124), [80, 80, 80])


def test_episodio_dura_30_dias():
    config = cargar_config()
    env = crear_entorno(config, discreto=True)
    env.reset(seed=0)
    pasos, fin = 0, False
    while not fin:
        _, _, terminado, truncado, _ = env.step(env.action_space.sample())
        pasos += 1
        fin = terminado or truncado
    assert pasos == 30 and truncado and not terminado


if __name__ == "__main__":
    test_acciones_discretas()
    test_episodio_dura_30_dias()
    print("Todo correcto")
