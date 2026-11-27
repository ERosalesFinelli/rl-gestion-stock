import numpy as np


class AgenteQ:
    def __init__(self, n_situaciones, n_acciones, alfa, gamma, semilla=0):
        self.q = np.zeros((n_situaciones, n_acciones))   # la libreta: 625 x 125, todo a 0
        self.alfa = alfa                                  # tamaño del "pasito"
        self.gamma = gamma                                # cuánto importa el futuro
        self.rng = np.random.default_rng(semilla)         # dado con semilla fija (reproducible)

    def elegir(self, fila, epsilon):
        # Con probabilidad epsilon: pedido al azar (explorar)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        # Si no: el pedido con la nota más alta en esa fila (aprovechar)
        return int(np.argmax(self.q[fila]))

    def aprender(self, fila, accion, recompensa, fila_nueva, terminado):
        futuro = 0.0 if terminado else self.q[fila_nueva].max()
        objetivo = recompensa + self.gamma * futuro
        self.q[fila, accion] += self.alfa * (objetivo - self.q[fila, accion])