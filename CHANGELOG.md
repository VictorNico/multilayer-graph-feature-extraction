# CHANGELOG — Pipeline MLNA

Toutes les modifications notables de ce projet sont documentées ici.
Format basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

---

## [Non publié] — branche `code` — février 2026

Optimisations grande échelle pour l'article DMKD (passage à >>1 000 exemples).

### Corrigé
- **`04_model_training.py`** — `make_mlna_1_variable_v2` : `config_df2` non chargé lors de `both=True` → `TypeError: 'NoneType' object is not subscriptable` à la ligne 179. Correction : chargement symétrique à `make_mlna_top_k_variable_v2`.
- **`modules/graph.py`** — `build_modalities_graph()` : `data.astype('int')` no-op (résultat non assigné) ; corrigé en `data = data.astype('int')`.

### Amélioré
- **`03_graph_construction.py`** — `extract_descriptors_from_graph_model` : remplacement des 6 appels `nx.pagerank()` séquentiels par un power iteration batché `(N×6)` avec `scipy.sparse` (1 conversion graph→scipy + 1 SpMM/itération au lieu de 6 SpMV). Gain ~3.5x sur la construction des descripteurs. Bloc de validation iso-résultat disponible en commentaire.
- **`modules/modeling.py`** — `train_classifier()` :
  - `SVC(kernel='linear')` : passage de `KernelExplainer` (O(n_train × n_features × n_test)) à `LinearExplainer` (O(n_features × n_test)), gain ×100-1000.
  - `TreeExplainer` : ajout de `check_additivity=False`, évite une vérification O(n_test × n_features) redondante.
  - `KernelExplainer` fallback : fond d'échantillonnage limité à 100 samples au lieu de `X_train` complet.
  - 6 appels `print()` de diagnostic commentés (exécutés en boucle interne).
- **`modules/statistical.py`** — `load_results()` : 22×N appels `os.walk()` (un par combinaison `(approach, logic, config, attribut)`) remplacés par 1 walk unique + helper `collect()` en mémoire. Gain O(22×N) → O(1) I/O.
- **`modules/preprocessing.py`** :
  - Boucle row-by-row `.loc[]` pour la normalisation → `np.where` vectorisé (×10–100).
  - 7 appels `.quantile()` séparés → 1 appel vectorisé `.quantile([0.05, 0.15, ...])`

### Infra / Qualité
- **`modules/statistical.py`** — `load_results` : `attributs=[]` (mutable default) → `attributs=None` avec guard.
- **`BRAINSTORMING.md`** — créé : document complet de session couvrant architecture, optimisations, bugs, estimations de temps, GPU, issues restantes.
- **`CLAUDE.md`** — créé : guide d'architecture pour Claude Code.

---

## [0.8.0] — 2026-02-13 — `7fd5476`

### Ajouté
- README mis à jour avec instructions détaillées d'exécution.
- Template de fichier `.env` pour la configuration des variables d'environnement.

---

## [0.7.1] — 2026-02-07 — `abb03a8` / `44f04c9`

### Ajouté
- Données d'exemple compressées (`data.zip`).
- Analyse par niveau de classe (class level 1) et par métrique dans les rapports.

---

## [0.7.0] — 2026-02-06 — `cd3480f`

### Modifié
- Réorganisation du code : étape 1 de refactoring structurel.

---

## [0.6.0] — 2026-01-12 — `5dc1782`

### Stable
- Pipeline stable pour dépôt DMKD #1.
- Point de référence avant optimisations grande échelle.

---

## [0.5.0] — 2025-12-26 — `78540fa`

### Amélioré
- Amélioration du reporting basé sur LaTeX : mise en page des tableaux, figures, et statistiques comparatives.

---

## [0.4.1] — 2025-07-30 — `c27cf7d`

### Corrigé
- Conversion du diviseur en `float` au lieu de `int` (division entière silencieuse).

---

## [0.4.0] — 2025-07-30 — `02e635d`

### Amélioré
- Optimisation de l'utilisation des cœurs CPU (limitation via `cpu_limitation_usage`).
- Correction des logiques analytiques.

---

## [0.3.1] — 2025-07-16 — `4bc1d43` / `64ca754`

### Modifié
- Configuration rendue portable pour tout utilisateur (chemins relatifs, setup universel).
- Révision du flux d'exécution (`launch.sh`).

---

## [0.3.0] — 2025-06-11/12 — `9cf47c6` / `73186973`

### Ajouté
- Pipeline complet et stable (100%) : prétraitement → graphes → entraînement → rapports.
- Script de reporting (`05_report_generation.py`).
- Suppression des mentions "Test" dans les e-mails de notification.
- Correction des proportions pour l'exécution sur données allemandes.
- Fichiers d'initialisation des datasets.

---

## [0.2.0] — 2025-05-21 — `b3c079e`

### Ajouté
- Démarrage du nouveau pipeline unifié (remplacement de l'ancienne version).
- Progression à 45 % des fonctionnalités cibles.

---

## [0.1.0] — 2024-09-12 — `df1eef8`

### Ajouté
- Premières modifications structurelles du projet.
- Mise en place de l'architecture initiale du pipeline MLNA.

---

## Issues connues (à corriger)

| ID | Fichier | Description | Priorité |
|---|---|---|---|
| ERR-2 | `modules/modeling.py:336` | `shap_vals_mean` : mauvaise forme pour cas multiclasse (`axis=0` au lieu de `axis=(0,1)`) | Grave |
| ERR-3 | `modules/graph.py` | `build_mlg_with_class` : nœuds modaux dupliqués (`list_of_nodes` vs `nodes_dict`) | Grave |
| WARN-1 | `03_graph_construction.py` | Ordre de suppression inversé dans les boucles de nettoyage CX (×3 fonctions) | Moyen |
| WARN-2 | `03_graph_construction.py` | `os.walk()` par variable pour vérification de complétion → `os.path.exists()` suffisant | Moyen |
| ERR-1* | `modules/graph.py` | `nx.pagerank()` sans `max_iter=500` hors de `extract_descriptors_from_graph_model` | Faible |

*ERR-1 résolu dans `extract_descriptors_from_graph_model` (batched PR, `max_iter=500`). Vérifier les appels restants dans `modules/graph.py`.