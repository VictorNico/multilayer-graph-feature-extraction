"""
    Author: VICTOR DJIEMBOU
    addedAt: 07/01/2024
    changes:
        - 07/01/2024:
            - add pipeline called
"""





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #################################################
    ##          Libraries importation
    #################################################

    ###### Begin

    from modules.pipeline import *

    ###### End
    # Définition des arguments
    import argparse

    parser = argparse.ArgumentParser(description='Exécution du pipeline MLNA')
    # parser.add_argument('--cwd', type=str, required=True, help='Répertoire de travail courant')
    # parser.add_argument('--domain', type=str, required=True, help='Domaine du jeu de données')
    # parser.add_argument('--dataset-link', type=str, required=True, help='Lien vers le jeu de données')
    # parser.add_argument('--target-variable', type=str, required=True, help='Variable cible')
    # parser.add_argument('--dataset-delimiter', type=str, required=True, help='Délimiteur du jeu de données')
    # parser.add_argument('--all-nominal', type=bool, required=True, help='Toutes les variables sont nominales')
    # parser.add_argument('--all-numeric', type=bool, required=True, help='Toutes les variables sont numériques')
    # parser.add_argument('--verbose', type=bool, required=True, help='Mode verbeux')
    # parser.add_argument('--fix-imbalance', type=bool, required=True, help='Corriger le déséquilibre des classes')
    # parser.add_argument('--levels', type=int, nargs='+', required=True, help='Niveaux des classes')
    # parser.add_argument('--to-remove', type=str, nargs='+', required=True, help='Variables à supprimer')
    # parser.add_argument('--encoding', type=str, required=True, help='Encodage du jeu de données')
    # parser.add_argument('--index-col', type=int, required=True, help='Colonne d\'index')
    parser.add_argument('--alpha', type=float, required=True, help='Valeur d\'alpha')
    # parser.add_argument('--portion', type=float, required=True, help='Portion du jeu de données')
    parser.add_argument('--graph', action="store_true", required=False, help='Afficher les graphiques avec les classes')
    # parser.add_argument('--financial-option-amount', type=str, required=True, help='Colonne pour le montant')
    # parser.add_argument('--financial-option-rate', type=str, required=True, help='Colonne pour le taux')
    # parser.add_argument('--financial-option-duration', type=str, required=True, help='Colonne pour la durée')
    # parser.add_argument('--duration-divider', type=int, required=True, help='Diviseur pour la durée')
    # parser.add_argument('--rate-divider', type=int, required=True, help='Diviseur pour le taux')

    # Récupération des arguments
    args = parser.parse_args()

    #################################################
    ##          définition du GUI
    #################################################
    #print_hi('PyCharm')
    mlnaPipeline(
        cwd= os.getcwd(),
        domain= 'AER',
        dataset_link= './datasets/private/3. Kaggle/credit-card-econometrics/AER_credit_card_data.csv',
        target_variable= 'card',
        dataset_delimiter=',', 
        all_nominal=True, 
        all_numeric=False, 
        verbose=True, 
        fix_imbalance=False, 
        levels=[2],
        to_remove= [], 
        encoding="utf-8",
        index_col=None,
        alphas=[args.alpha],
        portion=1.,
        graphWithClass=args.graph,
        withCost=False
        )
    contenu = f'END OF 1 AT {time.strftime("%Y_%m_%d_%H_%M_%S")} \n'
    with open("process.dtvni", "a") as fichier:
        fichier.write(contenu)
