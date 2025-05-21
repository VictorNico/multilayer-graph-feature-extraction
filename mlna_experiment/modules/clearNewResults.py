import os
import re

def delete_files_by_regex(root_path, regex_pattern):
    """
    Parcourt les sous-répertoires à partir de root_path et supprime les fichiers correspondant au pattern regex.

    Args:
    root_path (str): Le chemin initial à partir duquel la recherche commence.
    regex_pattern (str): L'expression régulière utilisée pour rechercher les fichiers à supprimer.
                         Exemple : r'.*\.txt$' pour supprimer tous les fichiers .txt.
    """
    # Compile l'expression régulière
    pattern = re.compile(regex_pattern)

    # Parcourir tous les fichiers et sous-répertoires
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            # Vérifie si le nom de fichier correspond à l'expression régulière
            # print(pattern.match(filename), filename)
            if pattern.match(filename):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    # print(f"Supprimé : {file_path}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {file_path} : {e}")

if __name__ == "__main__":
    # Demander à l'utilisateur de saisir le chemin de base et le pattern regex
    # root_path = input("Entrez le chemin du répertoire initial : ")
    # regex_pattern = input("Entrez l'expression régulière pour les fichiers à supprimer (ex: r'.*\\.txt$') : ")
    
    # Appeler la fonction pour supprimer les fichiers correspondants
    delete_files_by_regex(f"{os.getcwd()}/outputs_lts", r'v2.*\.csv$')
