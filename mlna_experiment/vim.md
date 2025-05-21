Voici une **liste pratique des raccourcis Vim essentiels**, classés par catégorie pour t’aider à naviguer, éditer et manipuler efficacement :

---

## 🧭 Navigation de base

| Raccourci       | Action                                                     |
| --------------- | ---------------------------------------------------------- |
| `h` / `l`       | Aller à gauche / droite                                    |
| `j` / `k`       | Aller en bas / en haut                                     |
| `w` / `W`       | Mot suivant (minuscule = mot, majuscule = mot+ponctuation) |
| `b` / `B`       | Mot précédent                                              |
| `0` / `^` / `$` | Début de ligne / premier mot / fin de ligne                |
| `gg` / `G`      | Aller au début / à la fin du fichier                       |
| `:n`            | Aller à la ligne `n`                                       |
| `zz`            | Centrer la ligne actuelle                                  |

---

## ✍️ Insertion et modification

| Raccourci        | Action                                              |
| ---------------- | --------------------------------------------------- |
| `i` / `I`        | Insertion (avant le curseur / au début de la ligne) |
| `a` / `A`        | Append (après le curseur / fin de ligne)            |
| `o` / `O`        | Nouvelle ligne en dessous / au-dessus               |
| `r`              | Remplacer un caractère                              |
| `R`              | Mode remplacement                                   |
| `cw`, `cc`, etc. | Change word / ligne complète                        |
| `u`              | Undo                                                |
| `Ctrl + r`       | Redo                                                |

---

## ✂️ Copier / Couper / Coller

| Raccourci     | Action                                                            |
| ------------- | ----------------------------------------------------------------- |
| `yy` / `Y`    | Copier ligne                                                      |
| `dd`          | Supprimer (couper) ligne                                          |
| `p` / `P`     | Coller après / avant                                              |
| `x`           | Supprimer caractère                                               |
| `daw`, `diw`  | Supprimer un mot (avec ou sans espaces)                           |
| `"*y` / `"*p` | Copier / coller depuis/vers le presse-papiers système (si activé) |

---

## 🔍 Recherche et remplacement

| Raccourci              | Action                               |
| ---------------------- | ------------------------------------ |
| `/mot`                 | Rechercher "mot"                     |
| `n` / `N`              | Résultat suivant / précédent         |
| `:%s/ancien/nouveau/g` | Remplacer globalement                |
| `:noh`                 | Supprimer la coloration de recherche |

---

## 🧱 Bloc / sélection

| Raccourci     | Action                         |
| ------------- | ------------------------------ |
| `v`           | Mode visuel (caractères)       |
| `V`           | Mode visuel (lignes)           |
| `Ctrl + v`    | Mode visuel en bloc (colonnes) |
| `y`, `d`, `>` | Copier, supprimer, indent bloc |
| `=`           | Auto-indent                    |

---

## 🗂️ Fichiers / buffers

| Raccourci           | Action                                         |
| ------------------- | ---------------------------------------------- |
| `:e nom_fichier`    | Ouvrir un fichier                              |
| `:w` / `:q` / `:wq` | Sauvegarder / quitter / sauvegarder et quitter |
| `:bnext` / `:bprev` | Aller au buffer suivant / précédent            |
| `:ls` / `:bd`       | Lister / fermer buffer                         |

---

## ⚙️ Divers

| Raccourci      | Action                                    |
| -------------- | ----------------------------------------- |
| `.`            | Répéter la dernière commande              |
| `:set nu`      | Afficher les numéros de ligne             |
| `:set paste`   | Activer mode collage                      |
| `:!cmd`        | Exécuter une commande shell               |
| `Ctrl + o / i` | Aller à la position précédente / suivante |

---

Souhaites-tu un **fichier PDF récapitulatif**, un **cheat sheet visuel**, ou une liste personnalisée selon ton usage (par exemple pour le code Python, Git, Markdown, etc.) ?

