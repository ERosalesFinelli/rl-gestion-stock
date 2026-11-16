"""Lectura de los archivos de configuración YAML."""
from pathlib import Path
import yaml

RAIZ = Path(__file__).resolve().parents[2]  # carpeta principal del repositorio


def cargar_config(ruta="configs/base.yaml"):
    """Lee un archivo YAML y lo devuelve como diccionario."""
    with open(RAIZ / ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
