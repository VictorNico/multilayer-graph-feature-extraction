#!/bin/bash

# Fichier de configuration
CONFIG_FILE="env.sh"
VENV_PATH="../scripting/.env/bin/activate"

# paramètres d'exécution
param1=$1

if [[ -z "$param1" ]]; then
    echo "Usage: $0 <nom_repertoire_dataset>"
    exit 1
fi


# Lire le fichier de configuration
if [[ -f "$CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    echo "environnement script chargé"
else
    echo "Fichier de configuration $CONFIG_FILE introuvable."
    exit 1
fi
# Activation de l'environement
# shellcheck disable=SC1090
if [[ ! -f "$VENV_PATH" ]]; then
    echo "L'environnement virtuel n'existe pas à l'emplacement $VENV_PATH"
    exit 1
else
    source "$VENV_PATH"
    echo "venv activé"
fi




# Exécution du chargement et du prétraitement des données
echo "Étape [1/5] : Prétraitement des données..."
python3.9 -m scripts.01_data_preprocessing --cwd="$cwd" --dataset_folder="$param1"

# Exécution de la séparation des données en jeux de test et d'entrainement
echo "Étape [2/5] : Split train/test..."
python3.9 -m scripts.02_data_split --cwd="$cwd" --dataset_folder="$param1"


# Exécution de la construction des graphes et extraction des descripteurs
echo "Étape [3/5] : Construction du graphe et extraction des descripteurs..."
# Exécuter le pipeline pour chaque valeur d'alpha
for alpha in "${alphas[@]}"; do
    short_name="${param1:0:3}"    # 2 lettres seulement
    alpha_short="${alpha:2}"   # Retire le point

    SCREEN_NAME="${short_name}_a${alpha_short}"
    screen -S "${SCREEN_NAME:0:15}" -dm bash -c "
      source $VENV_PATH && \
      python3.9 -m scripts.03_graph_construction --cwd=$cwd --dataset_folder=$param1  --alpha=$alpha --turn=1
      "
#    python3.9 -m scripts.03_graph_construction --cwd=$cwd --dataset_folder=$param1  --alpha=$alpha --turn=1
    echo "Construction du graphe alpha=${alpha:1} '$SCREEN_NAME'."
done

# Exécution de l'entrainement des modèles de Machine learning
echo "Étape [4/5] : Entraînement du modèle..."


# Exécution de la génération de rapport
echo "Étape [5/5] : Génération du rapport..."



#echo "Tous les écrans ont été lancés. Utilise 'screen -ls' pour les visualiser."
echo "💯=== PIPELINE TERMINÉ AVEC SUCCÈS ===💯"