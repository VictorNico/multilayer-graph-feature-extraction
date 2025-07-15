#!/bin/bash
set -e  # Arrête le script si une commande échoue

# Fichier de configuration
VENV_PATH=".env_mlna"


#if [[ -z "$param1" ]]; then
#    echo "Usage: $0 <nom_repertoire_dataset>"
#    exit 1
#fi


# Activation de l'environement
# shellcheck disable=SC1090
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "L'environnement virtuel n'existe pas à l'emplacement $VENV_PATH"
    # Crée un nouvel environnement pour tester
    echo "creation venv activé"
    python3.9 -m venv $VENV_PATH

    source $VENV_PATH/bin/activate
    echo "venv activé"
    echo "Installation des dependances"
    pip install --upgrade pip
    pip install setuptools
    pip install -r requirements.txt


else
    source "$VENV_PATH/bin/activate"
    echo "venv activé"
fi