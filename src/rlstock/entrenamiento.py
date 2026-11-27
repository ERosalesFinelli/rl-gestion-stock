"""Entrenamiento del agente Q (la libreta) y su política para evaluar sin explorar."""
import numpy as np

from rlstock.agente_q import AgenteQ
from rlstock.discretizador import Discretizador
from rlstock.entorno import crear_entorno


class PoliticaLibreta:
    """Juega mirando la libreta SIN explorar (epsilon = 0) y SIN aprender.

    Tiene el método 'actuar', igual que el azar y el base-stock, para poder
    usar la misma función 'evaluar' con todos y compararlos de forma justa.
    """

    def __init__(self, agente, discretizador):
        self.agente = agente
        self.disc = discretizador

    def actuar(self, obs):
        fila = self.disc.situacion(obs)
        return self.agente.elegir(fila, epsilon=0.0)


def entrenar(config, alfa, gamma, eps_inicial, eps_final, decaimiento, episodios, semilla):
    """Entrena una libreta jugando 'episodios' episodios de 30 días.

    epsilon empieza en eps_inicial y se multiplica por 'decaimiento' al acabar
    cada episodio, sin bajar nunca de eps_final.
    Devuelve (agente, lista con el beneficio de cada episodio de entrenamiento).
    """
    env = crear_entorno(config, discreto=True)
    disc = Discretizador(config["tramos"])
    agente = AgenteQ(disc.n_situaciones, env.action_space.n, alfa, gamma, semilla)

    epsilon = eps_inicial
    beneficios = []
    for ep in range(episodios):
        # Solo el primer episodio fija la semilla; los demás continúan la secuencia
        obs, _ = env.reset(seed=semilla) if ep == 0 else env.reset()
        fila = disc.situacion(obs)
        total, fin = 0.0, False
        while not fin:
            accion = agente.elegir(fila, epsilon)
            obs, recompensa, terminado, truncado, _ = env.step(accion)
            fila_nueva = disc.situacion(obs)
            agente.aprender(fila, accion, recompensa, fila_nueva, terminado)
            fila = fila_nueva
            total += recompensa
            fin = terminado or truncado
        beneficios.append(total)
        epsilon = max(eps_final, epsilon * decaimiento)
    return agente, beneficios