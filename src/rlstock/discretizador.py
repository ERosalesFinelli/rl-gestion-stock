"""Convierte el parte de la mañana (15 números) en UNA situación de la libreta.

La libreta mira 4 datos y cada uno se agrupa en 5 tramos:
  stock de la tienda, stock del distribuidor, stock de la fábrica y total de pedidos en camino.
Salen 5 x 5 x 5 x 5 = 625 situaciones posibles, numeradas del 0 al 624.
"""
import numpy as np


class Discretizador:
    NOMBRES = ["stock_tienda", "stock_distribuidor", "stock_fabrica", "pedidos_en_camino"]

    def __init__(self, cortes):
        # cortes: los límites de cada dato, tal como están en configs/base.yaml
        self.cortes = [np.array(cortes[nombre], dtype=np.float64) for nombre in self.NOMBRES]
        self.tramos_por_dato = len(self.cortes[0]) + 1                      # 5
        self.forma = (self.tramos_por_dato,) * len(self.NOMBRES)            # (5, 5, 5, 5)
        self.n_situaciones = int(np.prod(self.forma))                       # 625

    def cuatro_datos(self, obs):
        """Del parte de 15 números, saca los 4 que mira la libreta."""
        stock_tienda, stock_distribuidor, stock_fabrica = obs[2], obs[3], obs[4]
        pedidos_en_camino = obs[5:13].sum()   # los 8 números de pedidos en camino, sumados
        return [stock_tienda, stock_distribuidor, stock_fabrica, pedidos_en_camino]

    def tramos(self, obs):
        """Devuelve el tramo (0 a 4) de cada uno de los 4 datos."""
        datos = self.cuatro_datos(obs)
        # searchsorted con side="left": un valor igual al corte se queda en el tramo de abajo
        return [int(np.searchsorted(cortes, valor, side="left"))
                for valor, cortes in zip(datos, self.cortes)]

    def situacion(self, obs):
        """Devuelve el número de situación (0 a 624): la fila de la libreta."""
        return int(np.ravel_multi_index(self.tramos(obs), self.forma))
