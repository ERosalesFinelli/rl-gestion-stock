"""
Comprobación rápida del entorno de la tienda (gym-invmgmt, Serial-v0).

Qué hace (tarda unos 15 segundos):
  1. Comprueba que las librerías están bien instaladas.
  2. Pone a un "encargado" que pide al azar a jugar durante 10 segundos.
  3. Mide a qué velocidad va tu ordenador y cuánto gana el encargado al azar.
  4. Si tienes PyTorch instalado, mira si ve tu tarjeta gráfica (opcional).

Cómo se usa:
  pip install gymnasium gym-invmgmt numpy
  python comprobar_entorno.py
"""
import os
import platform
import sys
import time

try:
    import numpy as np
    import gymnasium as gym
    import gym_invmgmt  # noqa: F401  (al importarlo se registra el entorno)
except ImportError as error:
    print("Falta instalar algo:", error)
    print("Ejecuta:  pip install gymnasium gym-invmgmt numpy")
    sys.exit(1)

SEGUNDOS = 10

print("=== Tu equipo ===")
print("Python:", sys.version.split()[0])
print("Sistema:", platform.platform())
print("Hilos de procesador:", os.cpu_count())
print("Gymnasium:", gym.__version__)

print("\n=== Entorno de la tienda (Serial-v0) ===")
env = gym.make("GymInvMgmt/Serial-v0")
print("Lo que ve el programa (observación):", env.observation_space.shape, "números")
print("Lo que decide (acción):", env.action_space.shape, "pedidos, uno por eslabón")

print(f"\nEl encargado al azar juega {SEGUNDOS} segundos...")
pasos = 0
retornos = []
episodio = 0
inicio = time.time()
while time.time() - inicio < SEGUNDOS:
    env.reset(seed=episodio)
    fin = False
    total = 0.0
    while not fin:
        _, recompensa, terminado, truncado, _ = env.step(env.action_space.sample())
        total += recompensa
        pasos += 1
        fin = terminado or truncado
    retornos.append(total)
    episodio += 1
tiempo = time.time() - inicio

print("\n=== Resultado ===")
print(f"Episodios jugados: {episodio}")
print(f"Pasos por segundo: {pasos / tiempo:,.0f}")
print(f"Beneficio medio del encargado al azar: {np.mean(retornos):,.0f} (desviación {np.std(retornos):,.0f})")

print("\n=== Tarjeta gráfica (opcional) ===")
try:
    import torch
    print("PyTorch:", torch.__version__)
    if torch.cuda.is_available():
        print("GPU detectada:", torch.cuda.get_device_name(0))
    else:
        print("PyTorch no ve ninguna GPU (no pasa nada: este proyecto va en procesador).")
except ImportError:
    print("PyTorch no está instalado todavía (no hace falta para esta comprobación).")

print("\nCópiame todo lo que ha salido en pantalla.")
