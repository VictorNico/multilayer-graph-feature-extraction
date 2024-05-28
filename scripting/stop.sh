#!/bin/bash

# Fichier de configuration
CONFIG_FILE="config.ini"

# Lire le fichier de configuration
source $CONFIG_FILE

param1=$1
param2=$2

# Trouver et arrêter tous les processus en cours
echo "Arrêt des processus en cours..."
for alpha in "${alphas[@]}"; do
    SCREEN_NAME="${param1}_GT_"
    pids=$(screen -ls | grep "$SCREEN_NAME" | awk '{print $1}' | tr -d '.')
    if [ -n "$pids" ]; then
        echo "Arrêt des processus pour '$SCREEN_NAME'"
        screen -S "$SCREEN_NAME" -X quit
    fi

    SCREEN_NAME="${param1}_GF_"
    pids=$(screen -ls | grep "$SCREEN_NAME" | awk '{print $1}' | tr -d '.')
    if [ -n "$pids" ]; then
        echo "Arrêt des processus pour '$SCREEN_NAME'"
        screen -S "$SCREEN_NAME" -X quit
    fi
done