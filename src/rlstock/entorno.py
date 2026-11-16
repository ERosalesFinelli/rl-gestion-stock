"""Creación del entorno de la tienda y conversión de las acciones a 125 opciones discretas."""
import gymnasium as gym
import numpy as np
import gym_invmgmt  # noqa: F401  (al importarlo se registra el entorno)


class AccionesDiscretas(gym.ActionWrapper):
    """Convierte los 3 pedidos continuos en UNA decisión entre 125 opciones.

    Cada pedido (tienda, distribuidor, fábrica) puede tomar uno de 5 niveles.
    La acción 0..124 se traduce a una terna de niveles. Ejemplo con niveles
    [0, 10, 20, 40, 80]: la acción 62 equivale a pedir (20, 20, 20).
    """

    def __init__(self, env, niveles):
        super().__init__(env)
        self.niveles = np.array(niveles, dtype=np.float64)
        self.k = len(self.niveles)
        self.action_space = gym.spaces.Discrete(self.k ** 3)

    def action(self, accion):
        i, j, l = np.unravel_index(int(accion), (self.k, self.k, self.k))
        return np.array([self.niveles[i], self.niveles[j], self.niveles[l]])


def crear_entorno(config, discreto=True):
    """Crea el entorno. Con discreto=True el agente elige entre 125 acciones."""
    env = gym.make(config["entorno"]["id"])
    if discreto:
        env = AccionesDiscretas(env, config["acciones"]["niveles"])
    return env
