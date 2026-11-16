# Gestión de stock con aprendizaje por refuerzo

Proyecto de la asignatura **Introducción a la Inteligencia Artificial** (Grado en Ciencia de Datos e IA, UMH, curso 2026/27).
Autor: Eduardo Rosales Finelli.

Un agente aprende a decidir **cuánto pedir cada día** en una cadena de suministro (materias primas → fábrica → distribuidor → tienda)
con demanda aleatoria. Se comparan dos métodos programados desde cero (**Q-learning con tabla** y **DQN**) con dos referencias
(**agente al azar** y **regla base-stock**). Como ampliaciones: Double DQN, reward shaping y un vídeo del agente en acción.

- Entorno: [`gym-invmgmt`](https://pypi.org/project/gym-invmgmt/) (`GymInvMgmt/Serial-v0`), compatible con Gymnasium. No se ha programado ningún entorno propio.
- Coste: **solo CPU** (≈ 8.300 pasos/s medidos en un Core i9-14900HX).

## Estado del proyecto

| Parte | Estado |
|---|---|
| Estructura del repositorio y configuración | hecho |
| Baselines (azar y base-stock) | hecho |
| Q-learning con tabla | pendiente |
| DQN | pendiente |
| Barridos de hiperparámetros, 5 semillas, evaluación voraz | pendiente |
| Double DQN, reward shaping, vídeo | pendiente |

## Instalación (Windows, Python 3.12)

```powershell
git clone https://github.com/ERosalesFinelli/rl-gestion-stock.git
cd rl-gestion-stock
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cómo reproducir los experimentos

Todos los ajustes (semillas, niveles de pedido, etc.) están en `configs/*.yaml`, no dentro del código.

```powershell
python scripts/comprobar_entorno.py   # comprueba la instalación y mide la velocidad
python tests/test_entorno.py          # pruebas rápidas
python scripts/run_baselines.py       # evalúa azar y base-stock -> results/baselines.csv
```

## Estructura

```
configs/    ajustes de los experimentos (YAML)
src/rlstock/  código: entorno, baselines, evaluación (y pronto los agentes)
scripts/    programas que se ejecutan: comprobación, baselines (y pronto los entrenamientos)
tests/      pruebas rápidas
results/    resultados en crudo (CSV/JSON) de todo lo que aparece en la memoria
models/     modelos entrenados (menos de 50 MB)
figures/    gráficas de la memoria
videos/     vídeo del agente en acción
```

## Formulación como MDP (resumen)

- **Estado:** 15 números (stock de cada almacén, pedidos en camino, demanda reciente). Para la tabla se agrupan en tramos.
- **Acciones:** 3 pedidos (tienda, distribuidor, fábrica), cada uno con 5 niveles (`[0, 10, 20, 40, 80]`) → 125 acciones.
- **Recompensa:** beneficio de cada día. **γ** = 0,99 (se varía en el barrido).
- **Final:** 30 días, truncamiento por tiempo (no es estado terminal).

## Resultados provisionales (referencias)

50 episodios de evaluación con semillas 1000–1049 (`results/baselines.csv`):

| Agente | Beneficio medio por episodio | Desviación |
|---|---|---|
| Azar | ≈ −30.400 | ≈ 2.700 |
| Base-stock (200, 100, 100) | ≈ 857 | ≈ 21 |

## Declaración de uso de IA

Se ha usado un asistente de IA (Claude, de Anthropic) para planificar el proyecto, explicar conceptos y ayudar a escribir y depurar código.
El autor revisa, ejecuta y se responsabiliza de todo lo entregado. (Este apartado se ampliará en la memoria.)

## Licencia

MIT (ver `LICENSE`).
