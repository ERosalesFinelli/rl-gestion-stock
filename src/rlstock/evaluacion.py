"""Evaluación de un agente: juega episodios completos y mide el beneficio."""
import numpy as np


def evaluar(env, agente, semilla_inicio, episodios):
    """Juega 'episodios' episodios (semillas semilla_inicio, +1, +2...) y devuelve la lista de beneficios.

    Aquí el agente no aprende ni explora: solo se mide cuánto gana.
    """
    beneficios = []
    for k in range(episodios):
        obs, _ = env.reset(seed=semilla_inicio + k)
        total, fin = 0.0, False
        while not fin:
            obs, recompensa, terminado, truncado, _ = env.step(agente.actuar(obs))
            total += recompensa
            fin = terminado or truncado
        beneficios.append(total)
    return np.array(beneficios)
