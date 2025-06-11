#!/bin/bash
set -e  # Arrête le script si une commande échoue

# Fichier de configuration
CONFIG_FILE="env.sh"
VENV_PATH=".env_mlna"

# paramètres d'exécution
param1=$1

short_name="${param1:0:3}"
# Récupère la liste des sessions screen et les tue une par une
screen -ls | grep  "${short_name}_a" | awk '{print $1}' | while read session; do
  echo "Terminaison de la session screen: $session"
  screen -X -S "$session" quit
done

echo "Toutes les sessions screen ont été terminées."
