# 00_cpu_limitation usage
from modules.env import *  # Env functions
import os
import signal

# Set before importing NumPy/SciPy
print(load_env_with_path()['max_core'])
os.environ['OMP_NUM_THREADS'] = load_env_with_path()['max_core']
os.environ['MKL_NUM_THREADS'] = load_env_with_path()['max_core']
os.environ['OPENBLAS_NUM_THREADS'] = load_env_with_path()['max_core']
os.environ['NUMEXPR_NUM_THREADS'] = load_env_with_path()['max_core']

import sys

def cleanup():
    """Nettoyage explicite des ressources OpenMP"""
    print("Nettoyage des threads OpenMP...")

    # Forcer la fermeture des threads NumPy/OpenMP
    try:
        # Pour NumPy avec MKL
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass

    try:
        # Pour OpenBLAS
        import openblas
        openblas.set_num_threads(1)
    except ImportError:
        pass


def signal_handler(sig, frame):
    print(f'Signal {sig} - {frame}reçu, arrêt en cours...')
    cleanup()
    sys.exit(0)


# Capturer les signaux
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Enregistrer le nettoyage à la sortie
import atexit

atexit.register(cleanup)