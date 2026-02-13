# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PIPELINE STEPS @@@@@@@@@@@@@@

## Structure

    .root/
    |- .env_mlna/                   # venv
    |- configs/                     # contient les configurations d'exécution de chaque dataset
    |------ dataset_folder_name/
    |--------- config.ini           # fichier de config
    |- data/                        # contient le dataset à utiliser
    |------ dataset_folder_name/
    |- modules/                     # contient la logique de modules (eda, preprocessing, graph, mailing, ...)
    |------ *.py                    # fichier module
    |- scripts/                     # contient la logique des différentes phases de traitement lors de l'exécution
    |----- 01_data_preprocessing.py # chargement, aed, pretraitement et extration de la portion utile avec conservation de deséquilibre si existant
    |----- 02_data_split.py         # identifier une separation garantissant le critère < seuil
    |----- 03_graph_construction.py # mise sur pieds des graphes et extractions des différents descriptions et sauvegarde
    |----- 04_model_training.py     # chargement des descripteurs et des données splitées pour construire 8 predicteurs et evaluer leur performance
    |----- 05_report_generation.py  # analyser les resultats expérimentaux pour evaluer la pertinences des contributions
    |- env.sh                       # environnement utile pour le declenchement en bash script
    |- env_setup.sh                 # configuration de base du venv nécessaire et primordial
    |- kill_screens.sh              # terminer des screens détachés
    |- latex_install.sh             # Installer latex
    |- launch.sh                    # lancer la logique pipeline avec tous ses etapes sachant un repertoire de dataset
    |- Makefile                     # Faciliter l'exécution avec des symboles d'appel
    |- README.md                    # Manuel utilisateur
    |- requirements.txt             # list des depentances
    |_ vim.md                       # manuel d'utilisation de vim, un editeur en terminal comme nano


## Requierement (.env)
```dotenv
# ==============================================
# CONFIGURATION EMAIL - Fichier .env
# ==============================================

# Identifiants Gmail
GMAIL_USER=<EmailAccount>
GMAIL_APP_PASSWORD=<yourAppPassword>

# Configuration SMTP (optionnel, par défaut Gmail)
SMTP_SERVER=<smtp.gmail.com | yourSMTP>
SMTP_PORT=587
# Can value in set be : 0 < value < 1
ALPHAS="0.20, 0.50, 0.8"
# ==============================================
# DESTINATAIRES EMAIL
# ==============================================

# Destinataires principaux (séparés par des virgules)
EMAIL_RECIPIENTS=recipientName1 <Email1>, <...>

# Destinataires en copie (optionnel)
EMAIL_CC=followName1 <Email1>,<...>


# ============================ CODE ===========================
MAX_CORE=<yourLimit>
SIZE_DIVIDER=1
```


## Etapes

* Installer Latex [Facultatif]
```{bash}
chmod +x latex_install.sh
./latex_install.sh
```

* Installer l'environnement virtuel avec `python3`
```{bash}
chmod +x env_setup.sh
./env_setup.sh
```
Cette exécution va mettre en place une environement venv sous le repertoire `./.env_mlna` et installer tous les 
les dépendances associées à l'exécution de notre pipeline.
Après quoi, elle activera l'environnement.
    
* Créer un screen pour déclancher le pipeline
```{bash}
screen -S pipeline_launcher
# $name>|
```

* Définir quel est le type de lancement entre le framework basé sur le protocole et l'analyse exploratoire avec combination aléatoire de tails 2
```{bash}
vim Makefile
# dans l'entête du fichier, y'a une variable STEP
# mettre la valeur à 2 pour lancer le framework (par défaut)
# mettre la valeur à 3 pour la combinaire aléatoire 
```

* Dans le screen, exécuter a commande
```{bash}
make run-all
# 🚀 Démarrage de toutes les exécutions en parallèle -...
```

* détacher le screen en maintenant à la fois, `ctrl+a` puis `d`. 

* Rattacher le screen pour monitorer l'exécution
```{bash}
screen -r pipeline_launcher
# 🚀 Démarrage de toutes les exécutions en parallèle -...
# ....
# ....
```

* Pour plus de monitoring, il existe des symboles dans le file Makefile, pour afficher les logs
* le fichier __vim.md__, montre `les raccourcis action dans vim`