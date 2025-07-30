#!/bin/bash
set -e  # Arrête le script si une commande échoue

# Fichier de configuration
CONFIG_FILE="env.sh"
VENV_PATH=".env_mlna"

# Vérification des paramètres
if [ $# -eq 0 ]; then
    echo "Usage: $0 <nom_projet>"
    echo "Exemple: $0 mon_projet"
    exit 1
fi

param1=$1
short_name="${param1:0:3}"

echo "=== Nettoyage du projet: $param1 (préfixe: $short_name) ==="

# Fonction pour terminer les sessions screen
cleanup_screen_sessions() {
    echo "--- Recherche des sessions screen avec le préfixe '_a' ---"

    # Vérifier si screen est installé
    if ! command -v screen &> /dev/null; then
        echo "Screen n'est pas installé sur ce système"
        return 0
    fi

    # Obtenir la liste des sessions et les traiter
    sessions=$(screen -ls 2>/dev/null | grep "_a" | awk '{print $1}' || true)

    if [ -z "$sessions" ]; then
        echo "Aucune session screen trouvée avec le préfixe '${short_name}_a'"
        return 0
    fi

    echo "$sessions" | while read -r session; do
        if [ -n "$session" ]; then
            echo "Terminaison de la session screen: $session"
            screen -X -S "$session" quit 2>/dev/null || echo "Impossible de terminer: $session"
        fi
    done

    echo "Toutes les sessions screen ont été traitées."
}

# Fonction pour terminer les processus
cleanup_processes() {
    echo "--- Terminaison des processus ---"

    local scripts=(
        "scripts.04_model_training"
        "scripts.03_graph_construction"
        "scripts.02_data_split"
        "scripts.05_report_generation"
    )

    for script in "${scripts[@]}"; do
        echo "Recherche des processus: $script"

        # Compter les processus avant terminaison
        count=$(pgrep -f "$script" | wc -l || echo "0")

        if [ "$count" -gt 0 ]; then
            echo "  → $count processus trouvé(s)"
            # Tentative de terminaison propre d'abord
            pkill -TERM -f "$script" 2>/dev/null || true
            sleep 2

            # Vérifier s'il reste des processus et forcer la terminaison
            remaining=$(pgrep -f "$script" | wc -l || echo "0")
            if [ "$remaining" -gt 0 ]; then
                echo "  → Terminaison forcée de $remaining processus restant(s)"
                pkill -9 -f "$script" 2>/dev/null || true
            fi
            echo "  → Processus $script terminés"
        else
            echo "  → Aucun processus trouvé pour $script"
        fi
    done
}

# Fonction pour afficher un résumé
show_summary() {
    echo "--- Résumé final ---"
    echo "Sessions screen restantes avec préfixe '$short_name':"
    screen -ls 2>/dev/null | grep "$short_name" || echo "  Aucune"

    echo "Processus ML restants:"
    for script in "scripts.04_model_training" "scripts.03_graph_construction" "scripts.02_data_split" "scripts.05_report_generation"; do
        count=$(pgrep -f "$script" | wc -l || echo "0")
        if [ "$count" -gt 0 ]; then
            echo "  $script: $count processus"
        fi
    done
}

# Exécution du nettoyage
cleanup_screen_sessions
cleanup_processes
show_summary

echo "=== Nettoyage terminé ==="