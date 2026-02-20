"""
=============================================================================
Module de Nettoyage de Fichiers par Expression Régulière
=============================================================================

Auteur: Non spécifié
Date de création: Non spécifiée
Dernière modification: Non spécifiée

Description:
    Ce module fournit des utilitaires pour supprimer des fichiers dans une
    arborescence de répertoires en utilisant des expressions régulières
    pour le filtrage des noms de fichiers.

    Fonctionnalités principales:
    - Parcours récursif de répertoires
    - Filtrage de fichiers par pattern regex
    - Suppression sécurisée avec gestion d'erreurs

Dépendances:
    - os: Navigation dans le système de fichiers
    - re: Expressions régulières pour le matching de patterns

Cas d'usage:
    - Nettoyage de fichiers temporaires d'expérimentations
    - Suppression de résultats obsolètes
    - Maintenance automatique de répertoires de sortie

Avertissement:
    Ce module effectue des suppressions de fichiers IRRÉVERSIBLES.
    Utilisez avec précaution et testez d'abord avec des données non critiques.

=============================================================================
"""

import os
import re


def delete_files_by_regex(root_path, regex_pattern):
    """
    Parcourt récursivement les sous-répertoires et supprime les fichiers
    correspondant à un pattern d'expression régulière.

    Cette fonction effectue une recherche en profondeur dans l'arborescence
    des répertoires à partir du chemin racine spécifié, teste chaque nom de
    fichier contre le pattern regex, et supprime les fichiers correspondants.

    Args:
        root_path (str): Chemin du répertoire initial à partir duquel commencer
                        la recherche récursive
        regex_pattern (str): Expression régulière pour matcher les noms de fichiers
                            à supprimer

    Returns:
        None: La fonction effectue des suppressions mais ne retourne rien

    Exemples d'utilisation:
        >>> # Exemple 1: Supprimer tous les fichiers .txt
        >>> delete_files_by_regex('/home/user/project', r'.*\\.txt$')

        >>> # Exemple 2: Supprimer les fichiers commençant par 'temp_'
        >>> delete_files_by_regex('/project/output', r'^temp_.*')

        >>> # Exemple 3: Supprimer les fichiers CSV de version 2
        >>> delete_files_by_regex('/project/outputs', r'v2.*\\.csv$')

        >>> # Exemple 4: Supprimer les fichiers avec timestamp
        >>> delete_files_by_regex('/logs', r'.*_\\d{4}_\\d{2}_\\d{2}.*')

    Patterns regex courants:
        - r'.*\\.log$'           : Tous les fichiers .log
        - r'^test_.*'            : Fichiers commençant par 'test_'
        - r'.*_backup\\.'        : Fichiers contenant '_backup.'
        - r'v[0-9]+.*\\.csv$'    : Fichiers CSV versionnés (v1, v2, etc.)
        - r'.*_\\d{8}\\.txt$'    : Fichiers .txt avec date au format YYYYMMDD

    Comportement:
        - Parcourt TOUS les sous-répertoires (récursif)
        - Ne supprime QUE les fichiers (pas les répertoires)
        - Continue en cas d'erreur sur un fichier (affiche erreur et passe au suivant)
        - Les suppressions sont DÉFINITIVES (pas de corbeille)

    Sécurité:
        - Testez toujours d'abord avec un pattern restrictif
        - Vérifiez le root_path pour éviter de supprimer des fichiers système
        - Considérez faire une sauvegarde avant les opérations importantes
        - Le matching est sensible à la casse (utilisez (?i) pour ignorer la casse)

    Gestion d'erreurs:
        En cas d'erreur lors de la suppression d'un fichier:
        - Affiche un message d'erreur avec le chemin et la cause
        - Continue le traitement des autres fichiers
        - Erreurs possibles: permissions, fichier verrouillé, fichier en cours d'utilisation

    Note technique:
        La fonction utilise pattern.match() qui teste uniquement le DÉBUT du nom.
        Pour matcher n'importe où dans le nom, commencez le pattern par '.*'
    """

    # Compilation de l'expression régulière pour efficacité
    # La compilation une seule fois est plus rapide que de recompiler
    # à chaque test dans la boucle
    pattern = re.compile(regex_pattern)

    # Parcours récursif de l'arborescence des répertoires
    # os.walk() génère un tuple (dirpath, dirnames, filenames) pour chaque répertoire
    # - dirpath: chemin du répertoire courant
    # - dirnames: liste des sous-répertoires (modifiable pour filtrer la descente)
    # - filenames: liste des fichiers dans le répertoire courant
    for dirpath, dirnames, filenames in os.walk(root_path):

        # Parcours de tous les fichiers du répertoire courant
        for filename in filenames:

            # Test du nom de fichier contre l'expression régulière
            # pattern.match() cherche une correspondance au DÉBUT du string
            # Pour chercher n'importe où, utilisez pattern.search() ou commencez par '.*'
            # print(pattern.match(filename), filename)  # Ligne de debug commentée
            if pattern.match(filename):

                # Construction du chemin complet du fichier
                file_path = os.path.join(dirpath, filename)

                try:
                    # Tentative de suppression du fichier
                    os.remove(file_path)

                    # Message de confirmation (commenté pour éviter le spam)
                    # print(f"Supprimé : {file_path}")

                except Exception as e:
                    # Gestion des erreurs de suppression
                    # Erreurs possibles:
                    # - PermissionError: pas les droits de suppression
                    # - FileNotFoundError: fichier supprimé entre-temps
                    # - OSError: fichier verrouillé ou en cours d'utilisation
                    print(f"Erreur lors de la suppression de {file_path} : {e}")


# ============================================================
# SECTION PRINCIPALE - EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    """
    Section principale qui s'exécute uniquement si le script est lancé directement.

    Cette section peut être utilisée pour:
    - Tester la fonction avec des paramètres réels
    - Créer un outil en ligne de commande
    - Documenter des cas d'usage typiques
    """

    # ========================================
    # VERSION INTERACTIVE (commentée)
    # ========================================
    # Cette version demande à l'utilisateur de saisir les paramètres
    # Utile pour un outil interactif, mais commentée pour éviter les erreurs

    # root_path = input("Entrez le chemin du répertoire initial : ")
    # regex_pattern = input("Entrez l'expression régulière pour les fichiers à supprimer (ex: r'.*\\.txt$') : ")

    # ========================================
    # VERSION AUTOMATIQUE (active)
    # ========================================
    # Cette version utilise des paramètres hardcodés
    # Configuration actuelle: supprime les fichiers CSV commençant par 'v2' dans outputs_lts

    # Exemple d'utilisation concrète:
    # Supprime tous les fichiers CSV de version 2 dans le répertoire outputs_lts
    delete_files_by_regex(
        f"{os.getcwd()}/outputs_lts",  # Répertoire: <current_dir>/outputs_lts
        r'v2.*\.csv$'                   # Pattern: v2*.csv (ex: v2_results.csv, v2_data.csv)
    )

    # Explication du pattern r'v2.*\.csv$':
    # - r'...'      : raw string (pas d'échappement des backslashes)
    # - v2          : doit commencer par 'v2'
    # - .*          : suivi de n'importe quels caractères (0 ou plus)
    # - \.          : point littéral (échappé car '.' signifie "tout caractère" en regex)
    # - csv         : extension 'csv'
    # - $           : fin du nom de fichier

    # Exemples de fichiers qui SERONT supprimés:
    # - v2.csv
    # - v2_experiment.csv
    # - v2_results_final.csv
    # - v2_2023_11_15.csv

    # Exemples de fichiers qui NE SERONT PAS supprimés:
    # - v1.csv          (commence par v1, pas v2)
    # - v2.txt          (extension .txt, pas .csv)
    # - data_v2.csv     (ne commence pas par v2)
    # - v2_data.CSV     (sensible à la casse, CSV en majuscules)

    # Pour rendre insensible à la casse, utilisez: r'(?i)v2.*\.csv$'

    # ========================================
    # AUTRES EXEMPLES D'UTILISATION
    # ========================================

    # Exemple 1: Supprimer tous les fichiers de log
    # delete_files_by_regex(f"{os.getcwd()}/logs", r'.*\.log$')

    # Exemple 2: Supprimer les fichiers temporaires
    # delete_files_by_regex(f"{os.getcwd()}", r'^temp_.*')

    # Exemple 3: Supprimer les fichiers avec timestamp spécifique
    # delete_files_by_regex(f"{os.getcwd()}/outputs", r'.*_2023_.*\.csv$')

    # Exemple 4: Supprimer les fichiers de sauvegarde
    # delete_files_by_regex(f"{os.getcwd()}", r'.*\.bak$')

    # Exemple 5: Supprimer les fichiers LaTeX temporaires
    # delete_files_by_regex(f"{os.getcwd()}", r'.*\.(aux|log|out|toc)$')
