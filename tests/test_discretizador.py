"""Pruebas del discretizador:  python tests/test_discretizador.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from rlstock.config import cargar_config
from rlstock.discretizador import Discretizador
from rlstock.entorno import crear_entorno


def parte(stock_tienda=0, stock_distrib=0, stock_fabrica=0, en_camino=0):
    """Fabrica un parte de 15 números con los datos que nos interesan."""
    obs = np.zeros(15)
    obs[2], obs[3], obs[4] = stock_tienda, stock_distrib, stock_fabrica
    obs[5] = en_camino            # basta con meterlo en una casilla de las 8
    return obs


def test_tramos_de_la_tienda():
    d = Discretizador(cargar_config()["tramos"])
    # (stock de la tienda, tramo esperado entre 0 y 4)
    for stock, esperado in [(0, 0), (1, 1), (20, 1), (21, 2), (35, 2), (60, 2), (61, 3), (120, 3), (121, 4), (300, 4)]:
        assert d.tramos(parte(stock_tienda=stock))[0] == esperado, (stock, esperado)


def test_tramos_de_pedidos_en_camino():
    d = Discretizador(cargar_config()["tramos"])
    for total, esperado in [(0, 0), (50, 0), (51, 1), (120, 1), (121, 2), (300, 3), (301, 4)]:
        assert d.tramos(parte(en_camino=total))[3] == esperado, (total, esperado)


def test_numero_de_situaciones():
    d = Discretizador(cargar_config()["tramos"])
    assert d.n_situaciones == 625
    assert d.situacion(parte(0, 0, 0, 0)) == 0
    assert d.situacion(parte(999, 999, 999, 999)) == 624


def test_siempre_dentro_de_la_libreta():
    """Jugando de verdad, la situación nunca se sale del 0-624."""
    config = cargar_config()
    d = Discretizador(config["tramos"])
    env = crear_entorno(config, discreto=True)
    env.action_space.seed(0)
    for k in range(5):
        obs, _ = env.reset(seed=k)
        fin = False
        while not fin:
            assert 0 <= d.situacion(obs) < d.n_situaciones
            obs, _, terminado, truncado, _ = env.step(env.action_space.sample())
            fin = terminado or truncado


if __name__ == "__main__":
    test_tramos_de_la_tienda()
    test_tramos_de_pedidos_en_camino()
    test_numero_de_situaciones()
    test_siempre_dentro_de_la_libreta()
    print("Todo correcto")
