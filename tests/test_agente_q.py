"""Pruebas del agente Q:  python tests/test_agente_q.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlstock.agente_q import AgenteQ


def test_correccion_de_la_casilla():
    ag = AgenteQ(n_situaciones=3, n_acciones=4, alfa=0.5, gamma=0.9)

    # 1) Libreta vacía: hoy pierdo 10 y mañana "no sé nada" (futuro = 0)
    #    objetivo = -10 ; casilla = 0 + 0.5 * (-10 - 0) = -5
    ag.aprender(fila=0, accion=1, recompensa=-10, fila_nueva=1, terminado=False)
    assert abs(ag.q[0, 1] - (-5.0)) < 1e-9

    # 2) Ahora mañana tiene una casilla buena (20)
    #    objetivo = -10 + 0.9 * 20 = 8 ; casilla = -5 + 0.5 * (8 - (-5)) = 1.5
    ag.q[1, 2] = 20
    ag.aprender(fila=0, accion=1, recompensa=-10, fila_nueva=1, terminado=False)
    assert abs(ag.q[0, 1] - 1.5) < 1e-9

    # 3) Si el episodio termina de verdad, no hay mañana (futuro = 0)
    #    objetivo = -10 ; casilla = 1.5 + 0.5 * (-10 - 1.5) = -4.25
    ag.aprender(fila=0, accion=1, recompensa=-10, fila_nueva=1, terminado=True)
    assert abs(ag.q[0, 1] - (-4.25)) < 1e-9


def test_elegir():
    ag = AgenteQ(n_situaciones=3, n_acciones=4, alfa=0.5, gamma=0.9)
    ag.q[1, 2] = 20

    # Sin explorar (epsilon = 0): siempre la mejor casilla de la fila 1
    assert all(ag.elegir(1, 0.0) == 2 for _ in range(50))

    # Explorando siempre (epsilon = 1): sale de todo y siempre una acción válida
    vistas = {ag.elegir(1, 1.0) for _ in range(200)}
    assert vistas == {0, 1, 2, 3}


if __name__ == "__main__":
    test_correccion_de_la_casilla()
    test_elegir()
    print("Todo correcto")