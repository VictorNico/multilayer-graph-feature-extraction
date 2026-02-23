from dotenv import load_dotenv
import os

from pathlib import Path


def load_env_with_path():
    """Load environment variables from the .env file located in the current directory.

    Returns:
        dict: A dictionary containing all pipeline configuration values:
            - gmail_user (str): Gmail address for notifications.
            - gmail_password (str): Gmail App Password.
            - recipients (list[str]): Primary email recipients.
            - email_cc (list[str]): CC email recipients.
            - alphas (list[float]): Cost-sensitive alpha values.
            - max_core (str): Maximum number of CPU cores to use.
            - size_divider (float): Memory management divisor for graph size.
    """
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
        'size_divider': float(os.getenv('SIZE_DIVIDER'))
    }