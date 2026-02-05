"""
Script de Gestion de l'Utilisation des Cœurs CPU

Auteur: VICTOR DJIEMBOU
Date de création: 15/11/2023
Dernière modification: 15/11/2023

Description:
    Ce script configure et limite l'utilisation des cœurs CPU pour les bibliothèques
    de calcul scientifique (NumPy, SciPy, scikit-learn) afin d'optimiser les performances
    et d'éviter la surcharge du système lors des expérimentations.

    Il met en place:
    - Configuration des threads OpenMP pour les bibliothèques de calcul
    - Gestion des signaux système (SIGINT, SIGTERM)
    - Nettoyage automatique des ressources à la sortie

Fonctionnalités principales:
    - Limitation dynamique du nombre de threads via variable d'environnement
    - Gestion propre des interruptions (Ctrl+C)
    - Libération des ressources OpenMP/MKL/OpenBLAS

Usage:
    Ce module est importé au début des autres scripts via:
    from .cpu_limitation_usage import *

Dépendances:
    - modules.env: Configuration environnement (.env)
    - os, signal, sys, atexit: Gestion système
"""
# 00_cpu_limitation usage
from modules.env import *  # Env functions
import os
import signal

# Configuration du nombre maximum de threads AVANT l'importation de NumPy/SciPy
# Ceci est crucial car ces bibliothèques initialisent leurs pools de threads à l'importation

# Chargement de la configuration depuis le fichier .env
print(load_env_with_path()['max_core'])

# Configuration des variables d'environnement pour limiter les threads
# Ces variables contrôlent différentes bibliothèques de calcul parallèle:

# OMP_NUM_THREADS: OpenMP (utilisé par de nombreuses bibliothèques C/C++)
os.environ['OMP_NUM_THREADS'] = load_env_with_path()['max_core']

# MKL_NUM_THREADS: Intel Math Kernel Library (backend de NumPy)
os.environ['MKL_NUM_THREADS'] = load_env_with_path()['max_core']

# OPENBLAS_NUM_THREADS: OpenBLAS (backend alternatif de NumPy)
os.environ['OPENBLAS_NUM_THREADS'] = load_env_with_path()['max_core']

# NUMEXPR_NUM_THREADS: NumExpr (évaluations d'expressions numériques rapides)
os.environ['NUMEXPR_NUM_THREADS'] = load_env_with_path()['max_core']

import sys # Gestion de la sortie du programme

def cleanup():
    """
    Nettoyage explicite des ressources OpenMP et des threads de calcul

    Cette fonction est appelée:
    - À la sortie normale du programme (via atexit)
    - Lors d'une interruption (SIGINT, SIGTERM)

    Elle force la fermeture des pools de threads pour libérer les ressources système
    et éviter les threads orphelins.
    """
    print("Nettoyage des threads OpenMP...")

    # Tentative de forcer la fermeture des threads NumPy avec MKL
    try:
        # Pour NumPy avec MKL
        import mkl
        # Réduction à 1 thread pour forcer la libération des autres
        mkl.set_num_threads(1)
    except ImportError:
        # MKL n'est pas installé ou pas utilisé comme backend
        pass

    # Tentative de forcer la fermeture des threads OpenBLAS
    try:
        # Pour OpenBLAS
        import openblas
        # Réduction à 1 thread pour forcer la libération des autres
        openblas.set_num_threads(1)
    except ImportError:
        # OpenBLAS n'est pas installé ou pas utilisé comme backend
        pass


def signal_handler(sig, frame):
    """
    Gestionnaire de signaux pour interruptions propres

    Cette fonction est appelée lorsque le programme reçoit un signal système
    (par exemple, Ctrl+C génère SIGINT).

    Paramètres:
        sig (int): Numéro du signal reçu
        frame: Frame d'exécution actuel (contexte d'appel)

    Actions:
        1. Affiche le signal reçu
        2. Appelle cleanup() pour libérer les ressources
        3. Quitte proprement le programme avec code de sortie 0
    """
    print(f'Signal {sig} - {frame}reçu, arrêt en cours...')
    cleanup() # Nettoyage des ressources
    sys.exit(0) # Sortie propre du programme


# Enregistrement des gestionnaires de signaux
# SIGINT: Interruption depuis le clavier (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

# SIGTERM: Signal de terminaison (commande kill par défaut)
signal.signal(signal.SIGTERM, signal_handler)

# Enregistrement du nettoyage à la sortie normale du programme
import atexit

atexit.register(cleanup)  # cleanup() sera appelé automatiquement à la fin du programme
