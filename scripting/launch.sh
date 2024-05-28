#!/bin/bash

# Fichier de configuration
CONFIG_FILE="config.ini"

# Lire le fichier de configuration
source $CONFIG_FILE

param1=$1
param2=$2

# Exécuter le pipeline pour chaque valeur d'alpha
# source mlnaEnv/bin/activate &
for alpha in "${alphas[@]}"; do
    SCREEN_NAME="${param1}_GT_"
    # ${alpha:1}"
    screen -S "$SCREEN_NAME" -dm bash -c "python3.9 $param2 "
#    \ --alpha=$alpha"

    echo "Le pipeline avec alpha=${alpha:1} avec class a été lancé dans l'écran '$SCREEN_NAME'."
done

for alpha in "${alphas[@]}"; do
    SCREEN_NAME="${param1}_GF_"
    # ${alpha:1}"
    screen -S "$SCREEN_NAME" -dm bash -c "python3.9 $param2 \
        --graph"

#        --alpha=$alpha \

    echo "Le pipeline avec alpha=${alpha:1} avec class a été lancé dans l'écran '$SCREEN_NAME'."
done
