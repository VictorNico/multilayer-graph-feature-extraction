from dotenv import load_dotenv
import os

from pathlib import Path


def load_env_with_path():
    """Chargement avec chemin spécifique vers le fichier .env"""
    # Chemin vers le fichier .env
    env_path = Path('.') / '.env'

    # Chargement avec chemin explicite
    load_dotenv(dotenv_path=env_path)

    # Ou depuis un répertoire parent
    # load_dotenv(dotenv_path=Path('..') / '.env')

    return {
        'gmail_user': os.getenv('GMAIL_USER'),
        'gmail_password': os.getenv('GMAIL_APP_PASSWORD'),
        'recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
        'email_cc': os.getenv('EMAIL_CC', '').split(','),
        'alphas': [float(al) for al in os.getenv('ALPHAS','').split(',')],
        'max_core': os.getenv('MAX_CORE'),
        'size_divider': int(os.getenv('SIZE_DIVIDER'))
    }