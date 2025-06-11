#!/bin/bash
set -e  # Arrête le script si une commande échoue

# Fichier de configuration
CONFIG_FILE="env.sh"
VENV_PATH=".env_mlna"

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
if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "L'environnement virtuel n'existe pas à l'emplacement $VENV_PATH"
    # Crée un nouvel environnement pour tester
    echo "creation venv activé"
    python3.9 -m venv $VENV_PATH

    source $VENV_PATH/bin/activate
    echo "venv activé"
    echo "Installation des dependances"
    pip install -r requirements.txt


else
    source "$VENV_PATH/bin/activate"
    echo "venv activé"
fi




# Exécution du chargement et du prétraitement des données
echo "Étape [1/6] : Prétraitement des données..."
python3.9 -m scripts.01_data_preprocessing --cwd="$cwd" --dataset_folder="$param1"

# Exécution de la séparation des données en jeux de test et d'entrainement
echo "Étape [2/6] : Split train/test..."
python3.9 -m scripts.02_data_split --cwd="$cwd" --dataset_folder="$param1"

# Exécution de l'entrainement des baselines
echo "Étape [3/6] : Entrainement des modèles baseline..."
python3.9 -m scripts.04_model_training --baseline --cwd="$cwd" --dataset_folder="$param1"  --alpha=0.1 --turn=1


# Exécution de la construction des graphes et extraction des descripteurs
echo "Étape [4/5] : Construction du graphe et extraction des descripteurs..."

# Exécution de l'entrainement des modèles de Machine learning
echo "Étape [4/5] : Entraînement du modèle..."

# Exécuter le pipeline pour chaque valeur d'alpha
for alpha in "${alphas[@]}"; do
    short_name="${param1:0:3}"    # 2 lettres seulement
    alpha_short="${alpha:2}"   # Retire le point

    SCREEN_NAME="${short_name}_a${alpha_short}"

    LOG_FILE="logs/${short_name}/${SCREEN_NAME} $(date '+%Y-%m-%d %H:%M:%S').log"
    mkdir -p "logs/${short_name}"  # Assure que le dossier logs existe

    screen -S "${SCREEN_NAME:0:15}" -dm bash -c "
      source $VENV_PATH
      {
        echo \"🔹 [\$(date '+%Y-%m-%d %H:%M:%S')] DÉBUT du traitement pour alpha=$alpha\"

        python3.9 -m scripts.03_graph_construction --cwd=$cwd --dataset_folder=$param1 --alpha=$alpha --turn=1
        python3.9 -m scripts.04_model_training --cwd=$cwd --dataset_folder=$param1 --alpha=$alpha --turn=1

        parallel ::: \
          \"python3.9 -m scripts.03_graph_construction --graph_with_class --cwd=$cwd --dataset_folder=$param1 --alpha=$alpha --turn=2\" \
          \"python3.9 -m scripts.03_graph_construction --cwd=$cwd --dataset_folder=$param1 --alpha=$alpha --turn=2\"

        parallel ::: \
          \"python3.9 -m scripts.04_model_training --cwd=$cwd --dataset_folder=$param1 --alpha=$alpha --turn=2\" \
          \"python3.9 -m scripts.04_model_training --graph_with_class --cwd=$cwd --dataset_folder=$param1 --alpha=$alpha --turn=2\"

        echo \"✅ [\$(date '+%Y-%m-%d %H:%M:%S')] FIN du traitement pour alpha=$alpha\"
      } > \"$LOG_FILE\" 2>&1
    "

#    python3.9 -m scripts.03_graph_construction --cwd=$cwd --dataset_folder=$param1  --alpha=$alpha --turn=1
#    python3.9 -m scripts.04_model_training --cwd="$cwd" --dataset_folder="$param1"  --alpha="$alpha" --turn=1
    echo "Construction du graphe et entrainement pour alpha=${alpha:1} '$SCREEN_NAME'."
done

echo "⏳ Attente de la fin de tous les écrans screen..."

# Attendre que tous les écrans de traitement soient terminés
while screen -list | grep -q "${short_name}_a"; do
  echo "🔄 Screens encore actifs... attente de 10s"
  sleep 10
done


# Exécution de la génération de rapport
echo "Étape [5/5] : Génération du rapport..."

python3.9 -m scripts.05_report_generation --cwd=$cwd --dataset_folder=$param1
echo "✅ Tous les écrans sont terminés. Lancement du rapport..."


#echo "Tous les écrans ont été lancés. Utilise 'screen -ls' pour les visualiser."
echo "💯=== PIPELINE TERMINÉ AVEC SUCCÈS ===💯"