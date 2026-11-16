"""Los dos agentes de referencia: al azar y regla de reposición 'base-stock'."""
import numpy as np


class AgenteAzar:
    """Pide cantidades al azar (el 'suelo': lo mínimo que hay que superar)."""

    def __init__(self, env, semilla=0):
        self.env = env
        self.env.action_space.seed(semilla)

    def actuar(self, observacion):
        return self.env.action_space.sample()


class BaseStock:
    """Regla clásica: cada día pide lo que falta para llegar a un nivel objetivo.

    pedido = nivel_objetivo - (stock actual + pedidos que ya vienen de camino)
    Si sale negativo, no se pide nada.
    Usa los datos internos del entorno (stock y pedidos en camino).
    """

    def __init__(self, env, niveles_objetivo):
        self.u = env.unwrapped
        self.objetivo = np.array(niveles_objetivo, dtype=np.float64)

    def actuar(self, observacion):
        p = self.u.period
        stock = np.array([self.u.inventory[p][1], self.u.inventory[p][2], self.u.inventory[p][3]])
        en_camino = np.array(self.u.pipeline[p][:3])
        return np.clip(self.objetivo - stock - en_camino, 0, 3200)
