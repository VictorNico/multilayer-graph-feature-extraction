"""
Module de Gestion des Variables d'Environnement

Auteur: VICTOR DJIEMBOU
Date de création: 15/11/2023

Description:
    Ce module fournit des fonctions pour charger et accéder aux variables
    d'environnement depuis un fichier .env, notamment pour:
    - Les identifiants de messagerie Gmail
    - Les destinataires d'emails
    - Les paramètres d'expérimentation (alpha, CPU)

Fonctionnalités principales:
    - Chargement sécurisé des variables d'environnement depuis .env
    - Parsing automatique des listes (emails, alphas)
    - Conversion des types (float, int)

Usage:
    >>> from modules.env import load_env_with_path
    >>> config = load_env_with_path()
    >>> print(config['gmail_user'])
"""

from dotenv import load_dotenv  # Chargement des variables d'environnement depuis .env
import os                       # Accès aux variables d'environnement système
from pathlib import Path        # Manipulation de chemins de fichiers


def load_env_with_path():
    """
    Charge les variables d'environnement depuis un fichier .env et les retourne structurées

    Cette fonction:
    1. Localise le fichier .env dans le répertoire courant
    2. Charge toutes les variables d'environnement définies dans ce fichier
    3. Parse et convertit les valeurs dans les types appropriés
    4. Retourne un dictionnaire structuré avec tous les paramètres

    Variables d'environnement attendues dans .env:
        GMAIL_USER: Adresse email Gmail pour l'envoi de rapports
        GMAIL_APP_PASSWORD: Mot de passe d'application Gmail
        EMAIL_RECIPIENTS: Liste d'emails séparés par des virgules (destinataires principaux)
        EMAIL_CC: Liste d'emails séparés par des virgules (destinataires en copie)
        ALPHAS: Liste de valeurs alpha séparées par des virgules (paramètres PageRank)
        MAX_CORE: Nombre maximum de cœurs CPU à utiliser
        SIZE_DIVIDER: Diviseur de taille pour la normalisation

    Retourne:
        dict: Dictionnaire contenant:
            - gmail_user (str): Adresse Gmail
            - gmail_password (str): Mot de passe d'application Gmail
            - recipients (list): Liste des emails destinataires
            - email_cc (list): Liste des emails en copie
            - alphas (list): Liste des valeurs alpha (float) pour expérimentations
            - max_core (str): Nombre de cœurs CPU maximum
            - size_divider (float): Diviseur pour normalisation

    Exemple de fichier .env:
        GMAIL_USER=example@gmail.com
        GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
        EMAIL_RECIPIENTS=user1@example.com,user2@example.com
        EMAIL_CC=cc@example.com
        ALPHAS=0.5,0.75,0.85,0.9
        MAX_CORE=4
        SIZE_DIVIDER=1000.0
    """
    # Construction du chemin vers le fichier .env dans le répertoire courant
    # Path('.') désigne le répertoire de travail actuel
    env_path = Path('.') / '.env'

    # Chargement du fichier .env avec chemin explicite
    # Ceci charge toutes les variables dans l'environnement système
    load_dotenv(dotenv_path=env_path)

    # Alternative commentée: Chargement depuis un répertoire parent
    # load_dotenv(dotenv_path=Path('..') / '.env')

    # Construction et retour du dictionnaire de configuration
    return {
        # Identifiants Gmail pour l'envoi d'emails
        'gmail_user': os.getenv('GMAIL_USER'),
        'gmail_password': os.getenv('GMAIL_APP_PASSWORD'),

        # Listes de destinataires d'emails
        # .split(',') convertit la chaîne "a,b,c" en liste ['a', 'b', 'c']
        # Valeur par défaut '' si non défini, ce qui donne [''] après split
        'recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
        'email_cc': os.getenv('EMAIL_CC', '').split(','),

        # Paramètres d'expérimentation
        # List comprehension pour convertir chaque alpha en float
        'alphas': [float(al) for al in os.getenv('ALPHAS', '').split(',')],

        # Paramètres de performance
        'max_core': os.getenv('MAX_CORE'),              # Nombre de cœurs CPU (string)
        'size_divider': float(os.getenv('SIZE_DIVIDER'))  # Diviseur de normalisation (float)
    }
