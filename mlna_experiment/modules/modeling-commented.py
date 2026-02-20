"""
================================================================================
MODULE DE MODÉLISATION ML POUR L'ÉVALUATION DU RISQUE DE CRÉDIT
================================================================================

Auteur: VICTOR DJIEMBOU
Date de création: 28/11/2023
Dernière modification: 30/09/2024

Description:
-----------
Ce module implémente une infrastructure complète d'apprentissage automatique
pour l'évaluation du risque de crédit avec apprentissage sensible au coût.
Il fournit des fonctionnalités pour l'entraînement, l'évaluation et l'analyse
de multiples algorithmes de classification.

Algorithmes implémentés:
-----------------------
1. XGBoost (Extreme Gradient Boosting)
   - Arbres de décision boostés avec gradient
   - Régularisation L1/L2 pour éviter le surapprentissage
   - Optimisé pour les grandes dimensions

2. Random Forest (Forêt Aléatoire)
   - Ensemble d'arbres de décision
   - Bagging et sous-échantillonnage de caractéristiques
   - Robuste au bruit et aux valeurs aberrantes

3. SVM (Support Vector Machine)
   - Classification à marge maximale
   - Noyau linéaire pour séparation hyperplan
   - Probabilités calibrées via Platt scaling

4. Logistic Regression (Régression Logistique)
   - Modèle linéaire généralisé
   - Fonction sigmoïde pour probabilités
   - Interprétable et rapide

5. LDA (Linear Discriminant Analysis)
   - Analyse discriminante linéaire
   - Réduction de dimensionnalité
   - Maximise la séparation entre classes

6. Decision Tree (Arbre de Décision)
   - Partitionnement récursif de l'espace
   - Règles if-then interprétables
   - Base pour Random Forest et XGBoost

7. MLP (Multi-Layer Perceptron)
   - Réseau de neurones feedforward
   - Activation ReLU et optimisation Adam
   - Capable d'apprendre des relations non-linéaires

8. Perceptron
   - Réseau de neurones à une couche
   - Apprentissage linéaire simple
   - Base historique de l'apprentissage profond

Caractéristiques principales:
-----------------------------
- Apprentissage sensible au coût financier
- Calcul de valeurs SHAP pour l'interprétabilité
- Validation croisée stratifiée
- Métriques macro-moyennées (précision, rappel, F1)
- Gestion robuste des erreurs de convergence
- Support des déséquilibres de classes

Architecture du coût financier:
-------------------------------
Le module implémente un système de coût asymétrique qui pénalise différemment
les erreurs de classification:

- Faux Négatif (FN): Prédire "bon" pour un "mauvais" client
  Coût = montant_prêt × 1 (perte du capital)

- Faux Positif (FP): Prédire "mauvais" pour un "bon" client
  Coût = montant_prêt × taux × durée (perte d'opportunité)

Formule du coût total:
Cost_total = Σ(Cost_FN + Cost_FP) pour tous les exemples mal classés

Valeurs SHAP (SHapley Additive exPlanations):
----------------------------------------------
Les valeurs SHAP fournissent une explication cohérente et locale de la
contribution de chaque caractéristique à la prédiction du modèle.

Principe:
- Basé sur la théorie des jeux coopératifs de Shapley
- Distribue équitablement la prédiction entre toutes les caractéristiques
- Satisfait les propriétés de cohérence, linéarité et nullité

Types d'explainers utilisés:
1. TreeExplainer: Pour RF, DT, XGBoost (algorithmique, très rapide)
2. LinearExplainer: Pour LR, Perceptron, LDA (exact, rapide)
3. KernelExplainer: Pour SVM, MLP (approximatif, lent mais général)

Historique des modifications:
-----------------------------
- 28/11/2023: Ajout des méthodes pipeline
- 30/09/2024: Remplacement de l'importance des coefficients par valeurs SHAP

Dépendances:
-----------
- numpy, pandas: Manipulation de données
- scikit-learn: Algorithmes ML et métriques
- xgboost: Gradient boosting optimisé
- shap: Calcul d'explications de modèles

Usage:
------
>>> clfs = init_models()
>>> store = init_training_store(X_train)
>>> results = train(clfs, X_train, y_train, X_test, y_test, store, ...)
>>> print(results['f1-score'])

================================================================================
"""

#################################################
##          IMPORTATION DES BIBLIOTHÈQUES
#################################################

# 1. Bibliothèques standard Python
import sys

# 2. Bibliothèques externes de calcul scientifique et ML
import numpy as np
import pandas as pd
import shap  # Pour le calcul des valeurs SHAP d'interprétabilité
import random

# 3. Bibliothèques scikit-learn pour le machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score

# 4. Modules locaux du projet
from .file import save_model, save_dataset


#################################################
##          DÉFINITION DES MÉTHODES
#################################################


def init_models():
    """
    Initialise et configure tous les classificateurs utilisés dans l'analyse.

    Cette fonction crée une instance de chaque algorithme de classification
    avec des paramètres par défaut optimisés pour l'évaluation du risque de crédit.

    Modèles initialisés:
    -------------------
    1. LDA (Linear Discriminant Analysis):
       - Solver par défaut avec décomposition SVD
       - Réduit la dimensionnalité tout en maximisant la séparabilité

    2. LR (Logistic Regression):
       - Modèle linéaire pour classification binaire
       - Régularisation L2 par défaut

    3. SVM (Support Vector Machine):
       - Noyau linéaire pour problèmes linéairement séparables
       - probability=True pour obtenir des probabilités calibrées

    4. DT (Decision Tree):
       - Arbre de décision sans contrainte de profondeur
       - Peut capturer des relations non-linéaires complexes

    5. RF (Random Forest):
       - Ensemble de 100 arbres par défaut
       - Bootstrap et sélection aléatoire de features

    6. XGB (XGBoost):
       - Gradient boosting avec arbres (gbtree)
       - use_label_encoder=False pour éviter les avertissements

    7. MLP (Multi-Layer Perceptron):
       - 1 couche cachée de 100 neurones
       - Activation ReLU et optimisation Adam
       - max_iter=500 pour convergence

    8. PER (Perceptron):
       - Réseau de neurones simple à une couche
       - Apprentissage en ligne

    Returns:
    --------
    dict: Dictionnaire {nom_modèle: instance_classificateur}
          Les clés sont des chaînes courtes pour identification
          Les valeurs sont des instances de classificateurs non entraînés

    Example:
    --------
    >>> models = init_models()
    >>> print(models.keys())
    dict_keys(['LDA', 'LR', 'SVM', 'DT', 'RF', 'XGB', 'MLP', 'PER'])
    >>> rf_model = models['RF']
    >>> rf_model.fit(X_train, y_train)

    Notes:
    ------
    - Les modèles sont créés avec random_state non fixé sauf pour MLP
    - Pour reproductibilité, considérez définir random_state
    - Les hyperparamètres peuvent être ajustés via GridSearchCV
    """

    # DT: Arbre de décision - Classification basée sur des règles if-then
    dtc = DecisionTreeClassifier()

    # LR: Régression logistique - Modèle linéaire avec fonction sigmoïde
    lrc = LogisticRegression()

    # RF: Forêt aléatoire - Ensemble d'arbres avec bagging
    rfc = RandomForestClassifier()

    # XGB: XGBoost - Gradient boosting optimisé avec régularisation
    xgb = XGBClassifier(booster='gbtree', use_label_encoder=False)

    # LDA: Analyse discriminante linéaire - Maximise séparation inter-classes
    lda = LinearDiscriminantAnalysis()

    # SVM: Machine à vecteurs de support - Classification à marge maximale
    svc = SVC(
        kernel='linear',      # Noyau linéaire pour séparation hyperplan
        probability=True      # Active calibration pour probabilités
    )

    # MLP: Perceptron multi-couches - Réseau de neurones feedforward
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),  # 1 couche cachée de 100 neurones
        activation='relu',          # Fonction d'activation ReLU
        solver='adam',              # Optimiseur Adam (adaptatif)
        max_iter=500,               # Nombre max d'itérations
        random_state=42             # Graine pour reproductibilité
    )

    # PER: Perceptron simple - Réseau à une couche
    perceptron = Perceptron()

    # Dictionnaire regroupant tous les classificateurs
    clfs = {
        'LDA': lda,
        'LR': lrc,
        'SVM': svc,
        'DT': dtc,
        'RF': rfc,
        'XGB': xgb,
        'MLP': mlp,
        'PER': perceptron,
    }

    return clfs


def init_training_store(dataframe, withCost=True):
    """
    Initialise un DataFrame pour stocker les résultats d'entraînement.

    Cette fonction crée une structure de données vide pour collecter les
    valeurs SHAP de chaque caractéristique ainsi que les métriques de
    performance pour chaque modèle entraîné.

    Structure du DataFrame retourné:
    -------------------------------
    Colonnes = [feat_1, feat_2, ..., feat_n, precision, accuracy, recall,
                f1-score, financial-cost]
    Index = Noms des modèles (rempli lors de l'entraînement)

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame de référence pour extraire les noms de colonnes
        Généralement X_train avec toutes les features

    withCost : bool, default=True
        Si True, ajoute la colonne 'financial-cost'
        Si False, stocke uniquement les métriques classiques

    Returns:
    --------
    pandas.DataFrame
        DataFrame vide avec colonnes appropriées pour stocker:
        - Valeurs SHAP moyennes pour chaque feature
        - Métriques de performance (precision, accuracy, recall, f1-score)
        - Coût financier optionnel

    Example:
    --------
    >>> X_train = pd.DataFrame({'age': [25, 30], 'income': [50000, 60000]})
    >>> store = init_training_store(X_train, withCost=True)
    >>> print(store.columns.tolist())
    ['age', 'income', 'precision', 'accuracy', 'recall', 'f1-score', 'financial-cost']

    Notes:
    ------
    - Le DataFrame est vide initialement (0 lignes)
    - Les lignes seront ajoutées par train_classifier()
    - Chaque ligne correspondra à un modèle entraîné
    """

    # Récupération des noms de colonnes du DataFrame d'entrée
    cols = dataframe.columns.to_list()

    # Ajout des colonnes de métriques selon l'option de coût
    if withCost:
        cols.extend([
            'precision',        # Précision macro-moyennée
            'accuracy',         # Exactitude globale
            'recall',           # Rappel macro-moyenné
            'f1-score',         # F1-score macro-moyenné
            'financial-cost'    # Coût financier total
        ])
    else:
        cols.extend([
            'precision',
            'accuracy',
            'recall',
            'f1-score'
        ])

    # Création du DataFrame vide avec les colonnes définies
    return pd.DataFrame(columns=cols)


def get_xgb_imp(xgb):
    """
    Calcule l'importance normalisée des features pour XGBoost.

    Note: Cette fonction est conservée pour compatibilité mais n'est plus
    utilisée car remplacée par les valeurs SHAP qui fournissent une mesure
    d'importance plus cohérente et théoriquement fondée.

    XGBoost calcule l'importance par le nombre de fois qu'une feature
    est utilisée pour diviser les nœuds dans tous les arbres (F-score).

    Parameters:
    -----------
    xgb : XGBClassifier
        Un classificateur XGBoost entraîné

    Returns:
    --------
    dict : {nom_feature: importance_normalisée}
        Importance de chaque feature normalisée pour sommer à 1.0
        Les features non utilisées n'apparaissent pas dans le dict

    Example:
    --------
    >>> xgb_model = XGBClassifier()
    >>> xgb_model.fit(X_train, y_train)
    >>> importances = get_xgb_imp(xgb_model)
    >>> print(importances)
    {'age': 0.35, 'income': 0.45, 'credit_score': 0.20}

    Notes:
    ------
    - Remplacé par valeurs SHAP dans le pipeline actuel
    - get_fscore() retourne le nombre de splits par feature
    - La normalisation assure que Σ(importances) = 1.0
    """

    # Récupération du F-score (nombre de splits) pour chaque feature
    imp_vals = xgb.get_booster().get_fscore()

    # Calcul du total pour normalisation
    total = sum(imp_vals.values())

    # Normalisation: chaque importance divisée par le total
    return {k: v / total for k, v in imp_vals.items()}


def train_classifier(
        name,
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
        store,
        domain,
        prefix,
        cwd,
        financialOption,
        duration_divider,
        rate_divider,
        original,
        withCost=True
):
    """
    Entraîne un classificateur et évalue ses performances avec SHAP.

    Cette fonction constitue le cœur du pipeline d'entraînement. Elle:
    1. Entraîne le modèle avec gestion robuste des erreurs
    2. Génère des prédictions sur l'ensemble de test
    3. Calcule la matrice de confusion et les métriques
    4. Calcule le coût financier des erreurs
    5. Génère les valeurs SHAP pour l'interprétabilité
    6. Stocke tous les résultats dans le DataFrame store

    Gestion des erreurs par modèle:
    -------------------------------
    - LDA: Si SVD ne converge pas, passe au solver 'lsqr' avec shrinkage
    - Logistic Regression: Si échec, utilise solver 'liblinear'
    - SVM: Réduit max_iter à 500 si ne converge pas
    - Decision Tree: Limite max_depth=10 si problème
    - Random Forest: Réduit n_estimators=50 si nécessaire
    - XGBoost: Réduit n_estimators et learning_rate si échec
    - Perceptron: Ajuste max_iter et tol si ne converge pas
    - MLP: Utilise paramètres par défaut robustes

    Calcul des valeurs SHAP:
    ------------------------
    Les valeurs SHAP décomposent chaque prédiction en contributions des features.

    Pour un exemple x et une prédiction f(x):
    f(x) = φ_0 + Σ(φ_i) où φ_i est la valeur SHAP de la feature i

    Propriétés des valeurs SHAP:
    - Additivité locale: Les contributions somment à la prédiction
    - Cohérence: Si un modèle change pour augmenter l'impact d'une feature,
                 sa valeur SHAP augmente
    - Nullité: Features inutilisées ont valeur SHAP = 0

    Sélection de l'explainer SHAP:
    1. TreeExplainer: Pour modèles à base d'arbres (RF, DT, XGBoost)
       - Algorithme exact et rapide O(TLD²) où T=arbres, L=feuilles, D=profondeur
       - Exploite la structure arborescente

    2. LinearExplainer: Pour modèles linéaires (LR, Perceptron, LDA)
       - Calcul exact des valeurs SHAP en temps linéaire O(NF)
       - Utilise les coefficients du modèle

    3. KernelExplainer: Pour tous les autres modèles (SVM, MLP)
       - Méthode agnostique au modèle basée sur LIME
       - Plus lent O(2^F) mais applicable universellement

    Gestion des valeurs SHAP multi-classes:
    ---------------------------------------
    - Modèles binaires: shap_values est un array 2D (n_samples, n_features)
    - Modèles multi-classes: shap_values est une liste de arrays, un par classe
    - On calcule la moyenne absolue sur toutes les classes et samples

    Calcul du coût financier:
    -------------------------
    Voir compute_classification_financial_cost() pour détails.
    Principe: Pénalise différemment FP et FN selon impact financier réel.

    Parameters:
    -----------
    name : str
        Nom du classificateur (ex: 'RF', 'XGB', 'SVM')

    clf : sklearn classifier
        Instance du classificateur à entraîner

    X_train : pandas.DataFrame
        Features d'entraînement (n_samples, n_features)

    y_train : pandas.Series ou array
        Labels d'entraînement (n_samples,)

    X_test : pandas.DataFrame
        Features de test (n_test_samples, n_features)

    y_test : pandas.Series ou array
        Labels de test (n_test_samples,)

    store : pandas.DataFrame
        DataFrame pour stocker les résultats (modifié in-place)

    domain : str
        Nom du domaine/dataset pour identification

    prefix : str
        Préfixe pour les fichiers de sortie

    cwd : str
        Répertoire de travail courant pour sauvegardes

    financialOption : dict
        Configuration des colonnes pour calcul de coût:
        {'rate': nom_colonne_taux, 'amount': nom_colonne_montant,
         'duration': nom_colonne_durée}

    duration_divider : float
        Diviseur pour normaliser la durée (ex: 12 pour mois->années)

    rate_divider : float
        Diviseur pour normaliser le taux (ex: 100 pour %->décimal)

    original : tuple
        (X_original, test_dataset_original) pour calcul de coût

    withCost : bool, default=True
        Si True, calcule et stocke le coût financier

    Returns:
    --------
    pandas.DataFrame
        Le DataFrame store mis à jour avec une nouvelle ligne contenant:
        - Valeurs SHAP moyennes pour chaque feature
        - Métriques de performance
        - Coût financier (si withCost=True)

    Side Effects:
    -------------
    - Sauvegarde les prédictions dans un fichier CSV
    - Affiche des diagnostics SHAP sur stdout
    - Modifie le DataFrame store in-place

    Example:
    --------
    >>> clf = RandomForestClassifier()
    >>> store = init_training_store(X_train)
    >>> store = train_classifier('RF', clf, X_train, y_train, X_test, y_test,
    ...                          store, 'german', 'exp1', '/data',
    ...                          financial_opts, 12, 100, (X_orig, test_orig))
    >>> print(store.loc['RF', 'f1-score'])
    0.85

    Notes:
    ------
    - Les valeurs NaN dans store sont remplies avec 0
    - Les prédictions sont sauvegardées pour analyse ultérieure
    - Le calcul SHAP peut être lent pour KernelExplainer
    - Les erreurs d'entraînement sont attrapées et le modèle est reconfiguré
    """

    # ==========================================
    # PHASE 1: ENTRAÎNEMENT AVEC GESTION D'ERREURS
    # ==========================================

    # LDA: Analyse Discriminante Linéaire
    if isinstance(clf, LinearDiscriminantAnalysis):
        try:
            clf.fit(X_train, y_train)
        except np.linalg.LinAlgError:
            # Si la décomposition SVD échoue (matrices singulières)
            print("SVD did not converge for LDA. Trying with 'lsqr' solver.")
            # Passer au solver lsqr avec régularisation shrinkage
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_train, y_train)

    # Logistic Regression: Régression Logistique
    elif isinstance(clf, LogisticRegression):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Logistic Regression: {str(e)}")
            # Utiliser liblinear qui est plus robuste pour petits datasets
            clf = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
            clf.fit(X_train, y_train)

    # SVM: Machine à Vecteurs de Support
    elif isinstance(clf, SVC):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Support Vector Classifier: {str(e)}")
            # Réduire max_iter si problème de convergence
            clf = SVC(kernel='linear', probability=True, random_state=42, max_iter=500)
            clf.fit(X_train, y_train)

    # Decision Tree: Arbre de Décision
    elif isinstance(clf, DecisionTreeClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Decision Tree Classifier: {str(e)}")
            # Limiter la profondeur pour éviter le surapprentissage
            clf = DecisionTreeClassifier(random_state=42, max_depth=10)
            clf.fit(X_train, y_train)

    # Random Forest: Forêt Aléatoire
    elif isinstance(clf, RandomForestClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Random Forest Classifier: {str(e)}")
            # Réduire le nombre d'arbres si problème de mémoire
            clf = RandomForestClassifier(random_state=42, n_estimators=50)
            clf.fit(X_train, y_train)

    # XGBoost: Extreme Gradient Boosting
    elif isinstance(clf, XGBClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with XGBoost Classifier: {str(e)}")
            # Réduire learning rate et nombre d'arbres
            clf = XGBClassifier(
                booster='gbtree',
                use_label_encoder=False,
                random_state=42,
                n_estimators=50,
                learning_rate=0.05
            )
            clf.fit(X_train, y_train)

    # Perceptron: Réseau Simple à Une Couche
    elif isinstance(clf, Perceptron):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with Perceptron: {str(e)}")
            # Augmenter max_iter et ajuster tolérance
            clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
            clf.fit(X_train, y_train)

    # MLP: Perceptron Multi-Couches
    elif isinstance(clf, MLPClassifier):
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with MLPClassifier: {str(e)}")
            # Réinitialiser avec paramètres par défaut robustes
            clf = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            clf.fit(X_train, y_train)

    # Fallback pour tout autre classificateur
    else:
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f"Error with classifier: {str(e)}")

    # ==========================================
    # PHASE 2: PRÉDICTION ET SAUVEGARDE
    # ==========================================

    # Génération des prédictions sur l'ensemble de test
    y_pred = clf.predict(X_test)

    # Création d'un DataFrame avec les prédictions
    df_predictions = pd.DataFrame({'Prediction': y_pred})

    # Sauvegarde des prédictions dans un fichier CSV
    save_dataset(
        cwd=cwd,
        dataframe=df_predictions,
        name=f'{domain}_{name}_y_pred',
        prefix=prefix,
        sep=',',
        sub='/predictions'
    )

    # ==========================================
    # PHASE 3: CALCUL DES MÉTRIQUES DE PERFORMANCE
    # ==========================================

    # Calcul de la matrice de confusion
    # Format: [[TN, FP], [FN, TP]] pour classification binaire
    cfm = compute_confusion_matrix(y_test, y_pred, list(np.unique(y_test)))

    # Calcul des métriques macro-moyennées
    accuracy_r = accuracy_macro(cfm)      # Exactitude: (TP+TN)/(TP+TN+FP+FN)
    precision_r = precision_macro(cfm)    # Précision: TP/(TP+FP)
    f1_score_r = f1_macro(cfm)           # F1: 2*(P*R)/(P+R)
    recall_r = recall_macro(cfm)         # Rappel: TP/(TP+FN)

    # Calcul du coût financier si demandé
    if withCost:
        cost = compute_classification_financial_cost(
            list(y_pred),
            list(y_test),
            financialOption,
            original[1],              # Dataset original de test
            duration_divider,
            rate_divider
        )

    # ==========================================
    # PHASE 4: CALCUL DES VALEURS SHAP
    # ==========================================

    # Initialisation des variables SHAP
    explainer = None
    shap_values = None

    # Affichage de diagnostics pour debugging
    print(f"=== DIAGNOSTIC SHAP pour {name} ===")
    print(f"Type du modèle: {type(clf)}")
    print(f"Classes du modèle: {clf.classes_}")
    print(f"Nombre de classes: {len(clf.classes_)}")

    # Sélection et application de l'explainer SHAP approprié

    # Modèles linéaires: Utilisation de LinearExplainer (exact et rapide)
    if isinstance(clf, (LogisticRegression, Perceptron, LinearDiscriminantAnalysis)):
        print("Utilisation de LinearExplainer")
        explainer = shap.LinearExplainer(clf, X_train)
        shap_values = explainer.shap_values(X_test)

    # Modèles à base d'arbres: Utilisation de TreeExplainer (exact et rapide)
    elif isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier)):
        print("Utilisation de TreeExplainer")
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)

    # Autres modèles: Utilisation de KernelExplainer (approximatif mais général)
    else:
        print("Utilisation de KernelExplainer")
        # KernelExplainer nécessite les probabilités prédites
        explainer = shap.KernelExplainer(clf.predict_proba, X_train)
        shap_values = explainer.shap_values(X_test)

    # Diagnostic de la structure des valeurs SHAP retournées
    if isinstance(shap_values, list):
        # Multi-classes: une matrice SHAP par classe
        print(f"shap_values est une liste de {len(shap_values)} éléments")
        for i, sv in enumerate(shap_values):
            print(f"  Classe {i}: forme {sv.shape}")
    elif isinstance(shap_values, np.ndarray):
        # Binaire ou array unique
        print(f"shap_values est un array de forme: {shap_values.shape}")
    else:
        print(f"shap_values type inattendu: {type(shap_values)}")

    # ==========================================
    # PHASE 5: AGRÉGATION ET STOCKAGE DES RÉSULTATS
    # ==========================================

    # Calcul de la moyenne des valeurs absolues des SHAP
    # Cela donne une mesure globale de l'importance de chaque feature
    # abs() car on veut l'ampleur de l'effet, pas la direction
    # mean(axis=0) agrège sur tous les exemples de test
    shap_vals_mean = np.mean(np.abs(shap_values), axis=0)

    # Construction du vecteur de valeurs à stocker
    vals = list(shap_vals_mean)
    if withCost:
        vals.extend([precision_r, accuracy_r, recall_r, f1_score_r, cost])
    else:
        vals.extend([precision_r, accuracy_r, recall_r, f1_score_r])

    # Construction de la liste des clés (noms de colonnes)
    keys = X_train.columns.to_list()
    if withCost:
        keys.extend(['precision', 'accuracy', 'recall', 'f1-score', 'financial-cost'])
    else:
        keys.extend(['precision', 'accuracy', 'recall', 'f1-score'])

    # Ajout d'une nouvelle ligne au DataFrame store avec les résultats
    # Index = nom du modèle, colonnes = features + métriques
    store.loc[name] = pd.Series(vals, index=keys)

    # Remplissage des valeurs manquantes avec 0
    store.fillna(0, inplace=True)

    return store


def train(
        clfs,
        x_train,
        y_train,
        x_test,
        y_test,
        store,
        domain,
        prefix,
        cwd,
        duration_divider,
        rate_divider,
        financialOption,
        original,
        withCost=True
):
    """
    Entraîne tous les classificateurs et collecte les résultats.

    Cette fonction orchestre l'entraînement complet de tous les modèles
    du dictionnaire clfs en appelant train_classifier pour chacun.

    Pipeline d'exécution:
    --------------------
    Pour chaque modèle dans clfs:
    1. Affiche le nom du modèle et du domaine
    2. Appelle train_classifier avec tous les paramètres
    3. Accumule les résultats dans le DataFrame store
    4. Continue même si un modèle échoue

    Gestion d'erreurs:
    -----------------
    - Erreurs individuelles gérées dans train_classifier
    - Cette fonction peut échouer si store n'est pas initialisé
    - Les exceptions sont commentées mais peuvent être activées

    Parameters:
    -----------
    clfs : dict
        Dictionnaire {nom_modèle: instance_classificateur}
        Obtenu généralement via init_models()

    x_train : pandas.DataFrame
        Features d'entraînement (n_samples, n_features)

    y_train : pandas.Series ou array
        Labels d'entraînement (n_samples,)

    x_test : pandas.DataFrame
        Features de test (n_test_samples, n_features)

    y_test : pandas.Series ou array
        Labels de test (n_test_samples,)

    store : pandas.DataFrame
        DataFrame initialisé pour stocker résultats
        Généralement créé via init_training_store()

    domain : str
        Nom du domaine/dataset (ex: 'german', 'australian')

    prefix : str
        Préfixe pour identification des fichiers

    cwd : str
        Répertoire de travail courant

    duration_divider : float
        Diviseur pour normaliser durée des prêts

    rate_divider : float
        Diviseur pour normaliser taux d'intérêt

    financialOption : dict
        Configuration pour calcul de coût financier

    original : tuple
        (X_original, y_original) datasets originaux

    withCost : bool, default=True
        Si True, calcule le coût financier

    Returns:
    --------
    pandas.DataFrame
        Le DataFrame store complété avec une ligne par modèle:
        - Lignes: Noms des modèles (index)
        - Colonnes: Features SHAP + métriques + coût

    Side Effects:
    -------------
    - Affiche progression sur stdout
    - Sauvegarde fichiers via train_classifier
    - Peut prendre beaucoup de temps selon nombre de modèles

    Example:
    --------
    >>> clfs = init_models()
    >>> store = init_training_store(x_train)
    >>> results = train(clfs, x_train, y_train, x_test, y_test,
    ...                store, 'german', 'exp1', '/data', 12, 100,
    ...                financial_opts, (X_orig, y_orig))
    >>> print(results)
           feature1  feature2  ...  f1-score  financial-cost
    LDA        0.12      0.08  ...      0.75            1200
    RF         0.25      0.19  ...      0.82             800
    XGB        0.30      0.22  ...      0.85             650
    ...

    Notes:
    ------
    - L'ordre d'entraînement suit l'ordre du dictionnaire clfs
    - Chaque modèle est indépendant (pas de transfert d'apprentissage)
    - Le DataFrame store est modifié in-place mais aussi retourné
    - Les modèles entraînés ne sont pas sauvegardés par défaut
    """

    # Itération sur tous les classificateurs du dictionnaire
    for name, clf in clfs.items():
        # Affichage de la progression
        print(name, domain)

        # Entraînement du classificateur et mise à jour du store
        store = train_classifier(
            name=name,
            clf=clf,
            X_train=x_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            store=store,
            domain=domain,
            prefix=prefix,
            cwd=cwd,
            financialOption=financialOption,
            withCost=withCost,
            duration_divider=duration_divider,
            rate_divider=rate_divider,
            original=original
        )

    # Note: Le code commenté ci-dessous pourrait être activé pour
    # sauvegarder automatiquement les logs d'évaluation
    # link_to_evaluations_data = save_dataset(
    #     cwd=cwd,
    #     dataframe=store,
    #     name=domain,
    #     prefix=prefix,
    #     sep=',',
    #     sub="/evaluation"
    # )

    return store


def compute_classification_financial_cost(ypred, yreal, financialOption, test_dataset, duration_divider, rate_divider):
    """
    Calcule le coût financier total des erreurs de classification.

    Cette fonction implémente un système de coût asymétrique qui reflète
    l'impact financier réel des erreurs dans l'évaluation du risque de crédit.

    Contexte du problème:
    --------------------
    Dans le crédit scoring, toutes les erreurs n'ont pas le même coût:

    1. Faux Négatif (FN): Prédire classe 0 (bon) pour un vrai 1 (mauvais)
       Impact: Le prêt est accordé à un mauvais payeur
       Conséquence: Perte du capital prêté
       Coût = montant_prêt × 1 (Loss Given Default simplifié)

    2. Faux Positif (FP): Prédire classe 1 (mauvais) pour un vrai 0 (bon)
       Impact: Le prêt est refusé à un bon payeur
       Conséquence: Perte d'opportunité (intérêts non perçus)
       Coût = montant_prêt × taux_intérêt × durée_prêt

    Formulation mathématique:
    -------------------------
    Cost_total = Σ(Cost_FN_i + Cost_FP_i) pour i ∈ exemples_test

    Où:
    Cost_FN_i = montant_i × 1 si (pred_i=0 et vrai_i=1)
    Cost_FP_i = montant_i × taux_i × durée_i si (pred_i=1 et vrai_i=0)

    Simplification Loss Given Default (LGD):
    ----------------------------------------
    En pratique, LGD = (1 - Recovery Rate) varie selon:
    - Type de prêt (hypothécaire vs personnel)
    - Garanties disponibles
    - Législation locale sur faillites

    Ici, LGD = 1 (simplification conservative):
    - Difficile d'obtenir Recovery Rate pour datasets publics
    - Impact négligeable observé empiriquement sur classement des modèles
    - Approche worst-case pour majorer le risque

    Normalisation des paramètres:
    -----------------------------
    Les diviseurs permettent de convertir les unités:
    - duration_divider: Convertit mois en années (généralement 12)
                       Car taux d'intérêt est annuel
    - rate_divider: Convertit pourcentage en décimal (généralement 100)
                   Ex: 15% → 0.15

    Parameters:
    -----------
    ypred : list
        Prédictions du modèle (0 ou 1) pour ensemble de test
        Longueur = n_test_samples

    yreal : list
        Vraies étiquettes (0 ou 1) pour ensemble de test
        Longueur = n_test_samples

    financialOption : dict
        Configuration des colonnes financières:
        {
            'rate': str,      # Nom de colonne du taux d'intérêt
            'amount': str,    # Nom de colonne du montant du prêt
            'duration': str   # Nom de colonne de la durée du prêt
        }

    test_dataset : pandas.DataFrame
        Dataset de test original avec colonnes financières
        Index doit correspondre aux exemples dans ypred/yreal

    duration_divider : float
        Diviseur pour convertir durée en années
        Ex: 12 si durée en mois, 1 si déjà en années

    rate_divider : float
        Diviseur pour convertir taux en décimal
        Ex: 100 si taux en %, 1 si déjà en décimal

    Returns:
    --------
    float
        Coût financier total sur l'ensemble de test
        Unité = même que montant_prêt (généralement $, €, etc.)

    Example:
    --------
    >>> y_pred = [0, 1, 0, 1]  # Prédictions
    >>> y_true = [0, 0, 1, 1]  # Vraies classes
    >>> test_df = pd.DataFrame({
    ...     'loan_amount': [10000, 15000, 8000, 12000],
    ...     'interest_rate': [5, 7, 6, 8],  # En %
    ...     'loan_duration': [24, 36, 12, 48]  # En mois
    ... }, index=[0, 1, 2, 3])
    >>> financial_opts = {
    ...     'amount': 'loan_amount',
    ...     'rate': 'interest_rate',
    ...     'duration': 'loan_duration'
    ... }
    >>> cost = compute_classification_financial_cost(
    ...     y_pred, y_true, financial_opts, test_df, 12, 100
    ... )
    >>> print(f"Coût total: ${cost:.2f}")
    Coût total: $10525.00

    Détail du calcul pour l'exemple:
    - Index 0: TN (0,0) → Coût = 0
    - Index 1: FP (1,0) → Coût = 15000 × 0.07 × 3 = 3150
    - Index 2: FN (0,1) → Coût = 8000 × 1 = 8000
    - Index 3: TP (1,1) → Coût = 0
    Total = 3150 + 8000 = 11150

    Notes:
    ------
    - Vrais Positifs (TP) et Vrais Négatifs (TN) ont coût = 0
    - Le coût est toujours positif (ou nul)
    - Plus le coût est bas, meilleur est le modèle
    - Cette métrique est plus pertinente que accuracy pour crédit scoring
    - Les diviseurs doivent être cohérents avec les unités du dataset
    """

    # Initialisation du coût total à 0
    cost = 0

    # Itération sur chaque exemple du dataset de test
    for i, example in enumerate(list(test_dataset.index)):
        # Extraction des paramètres financiers pour cet exemple
        # Normalisation immédiate avec les diviseurs
        rate = test_dataset[financialOption['rate']][example] / rate_divider
        amount = test_dataset[financialOption['amount']][example]
        duration = test_dataset[financialOption['duration']][example] / duration_divider

        # Récupération des labels (vrai et prédit) pour cet exemple
        true_label = yreal[i]
        predicted_label = ypred[i]

        # Calcul du coût selon le type d'erreur

        # Cas 1: Faux Négatif (FN)
        # Prédiction = 0 (bon payeur) mais réalité = 1 (mauvais payeur)
        # Impact: Perte du capital car prêt accordé à mauvais payeur
        if predicted_label == 0 and true_label == 1:
            # Coût = montant total du prêt (LGD = 1)
            # Note: Idéalement cost += amount * loss_given_default
            # mais LGD difficile à obtenir et impact faible observé
            cost += amount * 1

        # Cas 2: Faux Positif (FP)
        # Prédiction = 1 (mauvais payeur) mais réalité = 0 (bon payeur)
        # Impact: Perte d'opportunité car prêt refusé à bon payeur
        elif predicted_label == 1 and true_label == 0:
            # Coût = intérêts qui auraient été perçus
            # Formule: Capital × Taux × Durée (intérêts simples)
            deficit = amount * rate * duration
            cost += deficit

        # Cas 3 et 4: Classifications correctes (TP et TN)
        # Pas de coût ajouté (implicitement cost += 0)

    return cost


def compute_confusion_matrix(true_labels, predicted_labels, labels):
    """
    Calcule la matrice de confusion pour évaluation multi-classes.

    La matrice de confusion est un tableau carré qui résume les performances
    d'un classificateur en comptant les prédictions correctes et incorrectes
    pour chaque classe.

    Structure de la matrice de confusion:
    -------------------------------------
    Pour classification binaire (classe 0 et 1):

                      Prédit
                   0         1
    Vrai    0    [[TN,      FP],
            1     [FN,      TP]]

    Où:
    - TN (True Negative): Prédit 0, Vrai 0 - Correct
    - FP (False Positive): Prédit 1, Vrai 0 - Erreur Type I
    - FN (False Negative): Prédit 0, Vrai 1 - Erreur Type II
    - TP (True Positive): Prédit 1, Vrai 1 - Correct

    Pour multi-classes (n classes):
    Matrice n×n où élément (i,j) = nombre de fois que:
    - Vraie classe = i
    - Prédiction = j

    La diagonale contient les classifications correctes.
    Les éléments hors-diagonale sont les erreurs.

    Algorithme:
    -----------
    1. Initialiser matrice n×n à zéros
    2. Pour chaque (vrai, prédit) dans les labels:
       a. Trouver index de la vraie classe
       b. Trouver index de la classe prédite
       c. Incrémenter matrice[index_vrai][index_prédit]

    Complexité: O(m × n) où m = nombre d'exemples, n = nombre de classes

    Parameters:
    -----------
    true_labels : list ou array
        Vraies étiquettes de classe (n_samples,)

    predicted_labels : list ou array
        Prédictions du modèle (n_samples,)

    labels : list
        Liste ordonnée des classes possibles
        L'ordre détermine les indices dans la matrice

    Returns:
    --------
    list of lists
        Matrice de confusion n×n où n = len(labels)
        Format: [[c00, c01, ...], [c10, c11, ...], ...]

    Example:
    --------
    >>> y_true = [0, 1, 0, 1, 1, 0]
    >>> y_pred = [0, 1, 1, 1, 0, 0]
    >>> labels = [0, 1]
    >>> cm = compute_confusion_matrix(y_true, y_pred, labels)
    >>> print(cm)
    [[2, 1],   # Classe 0: 2 TN, 1 FP
     [1, 2]]   # Classe 1: 1 FN, 2 TP

    >>> # Multi-classes
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 2, 0, 1, 0]
    >>> labels = [0, 1, 2]
    >>> cm = compute_confusion_matrix(y_true, y_pred, labels)
    >>> print(cm)
    [[2, 0, 0],  # Classe 0: 2 corrects
     [0, 1, 1],  # Classe 1: 1 correct, 1 confondu avec 2
     [1, 0, 1]]  # Classe 2: 1 correct, 1 confondu avec 0

    Notes:
    ------
    - Les vraies classes sont en lignes, prédictions en colonnes
    - Somme de la ligne i = nombre total d'exemples de classe i
    - Somme de la colonne j = nombre total de prédictions de classe j
    - Somme de la diagonale = nombre total de prédictions correctes
    - Implémentation manuelle (pas sklearn) pour contrôle et pédagogie
    """

    # Détermination du nombre de classes
    num_classes = len(labels)

    # Initialisation de la matrice de confusion à zéros
    # Structure: liste de listes pour matrice n×n
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    # Itération sur toutes les paires (vrai, prédit)
    for true, predicted in zip(true_labels, predicted_labels):
        # Récupération des indices dans la liste des labels
        true_index = labels.index(true)
        predicted_index = labels.index(predicted)

        # Incrémentation du compteur approprié
        confusion_matrix[true_index][predicted_index] += 1

    return confusion_matrix


def precision_macro(confusion_matrix):
    """
    Calcule la précision macro-moyennée à partir de la matrice de confusion.

    La précision mesure la proportion de prédictions positives qui sont
    correctes. Pour multi-classes, on calcule la précision par classe puis
    on moyenne.

    Définition mathématique:
    -----------------------
    Pour une classe i:
    Precision_i = TP_i / (TP_i + FP_i)

    Où:
    - TP_i (True Positives): Éléments de classe i correctement prédits
    - FP_i (False Positives): Éléments d'autres classes prédits comme i

    Dans la matrice de confusion:
    - TP_i = confusion_matrix[i][i] (diagonale)
    - FP_i = Σ(confusion_matrix[j][i]) pour j≠i (somme colonne i sans diag)

    Macro-moyenne:
    Precision_macro = (1/n) × Σ(Precision_i) pour i=1 à n

    Différence avec micro-moyenne:
    ------------------------------
    - Macro: Moyenne des précisions par classe (traite classes équitablement)
    - Micro: Précision globale en agrégeant TP et FP (favorise classes fréquentes)

    Interprétation:
    ---------------
    - Precision = 1.0: Toutes les prédictions positives sont correctes
    - Precision = 0.0: Aucune prédiction positive n'est correcte
    - Haute précision: Peu de faux positifs (faible Type I error)

    Cas particulier:
    ----------------
    Si TP_i + FP_i = 0 (aucune prédiction de classe i), on définit:
    Precision_i = 0.0 (convention pour éviter division par zéro)

    Parameters:
    -----------
    confusion_matrix : list of lists
        Matrice de confusion n×n
        Format: [[c00, c01, ...], [c10, c11, ...], ...]

    Returns:
    --------
    float
        Précision macro-moyennée entre 0.0 et 1.0

    Example:
    --------
    >>> cm = [[90, 10],   # Classe 0: 90 TN, 10 FP
    ...       [5, 95]]    # Classe 1: 5 FN, 95 TP
    >>> precision = precision_macro(cm)
    >>> # Classe 0: 90/(90+5) = 0.947
    >>> # Classe 1: 95/(95+10) = 0.905
    >>> # Macro: (0.947 + 0.905)/2 = 0.926
    >>> print(f"{precision:.3f}")
    0.926

    Notes:
    ------
    - Utile pour datasets déséquilibrés (donne poids égal à chaque classe)
    - Sensible aux performances sur classes minoritaires
    - Complémentaire au rappel (trade-off précision-rappel)
    """

    num_classes = len(confusion_matrix)
    precisions = []

    # Calcul de la précision pour chaque classe
    for i in range(num_classes):
        # TP pour classe i: élément diagonal
        tp = confusion_matrix[i][i]

        # FP pour classe i: somme de la colonne i sans la diagonale
        # C'est-à-dire tous les éléments prédits comme i mais qui ne le sont pas
        fp = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)

        # Dénominateur: total des prédictions positives pour classe i
        denom = tp + fp

        # Calcul de la précision avec gestion de division par zéro
        precisions.append(tp / denom if denom else 0.0)

    # Retour de la macro-moyenne
    return sum(precisions) / num_classes


def recall_macro(confusion_matrix):
    """
    Calcule le rappel macro-moyenné à partir de la matrice de confusion.

    Le rappel (ou sensibilité) mesure la proportion d'exemples positifs
    qui sont correctement identifiés. Pour multi-classes, on calcule
    le rappel par classe puis on moyenne.

    Définition mathématique:
    -----------------------
    Pour une classe i:
    Recall_i = TP_i / (TP_i + FN_i)

    Où:
    - TP_i (True Positives): Éléments de classe i correctement prédits
    - FN_i (False Negatives): Éléments de classe i prédits comme autre chose

    Dans la matrice de confusion:
    - TP_i = confusion_matrix[i][i] (diagonale)
    - FN_i = Σ(confusion_matrix[i][j]) pour j≠i (somme ligne i sans diag)

    Macro-moyenne:
    Recall_macro = (1/n) × Σ(Recall_i) pour i=1 à n

    Différence avec précision:
    --------------------------
    - Precision: De toutes les prédictions positives, combien sont correctes?
                (focus sur qualité des prédictions positives)

    - Recall: De tous les vrais positifs, combien sont détectés?
             (focus sur complétude de la détection)

    Trade-off précision-rappel:
    ---------------------------
    Il existe souvent un compromis:
    - Seuil bas → Rappel élevé, Précision faible (beaucoup de prédictions)
    - Seuil haut → Précision élevée, Rappel faible (prédictions sélectives)

    Interprétation:
    ---------------
    - Recall = 1.0: Tous les positifs sont détectés
    - Recall = 0.0: Aucun positif n'est détecté
    - Haut rappel: Peu de faux négatifs (faible Type II error)

    Importance dans le crédit scoring:
    ----------------------------------
    - Haut rappel pour classe "mauvais": Détecte la plupart des mauvais payeurs
    - Mais peut augmenter les faux positifs (rejets de bons clients)

    Cas particulier:
    ----------------
    Si TP_i + FN_i = 0 (aucun exemple de classe i), on définit:
    Recall_i = 0.0 (convention pour éviter division par zéro)

    Parameters:
    -----------
    confusion_matrix : list of lists
        Matrice de confusion n×n
        Format: [[c00, c01, ...], [c10, c11, ...], ...]

    Returns:
    --------
    float
        Rappel macro-moyenné entre 0.0 et 1.0

    Example:
    --------
    >>> cm = [[90, 10],   # Classe 0: 90 vrais 0, dont 10 prédits comme 1
    ...       [5, 95]]    # Classe 1: 95 vrais 1, dont 5 prédits comme 0
    >>> recall = recall_macro(cm)
    >>> # Classe 0: 90/(90+10) = 0.900
    >>> # Classe 1: 95/(95+5) = 0.950
    >>> # Macro: (0.900 + 0.950)/2 = 0.925
    >>> print(f"{recall:.3f}")
    0.925

    Notes:
    ------
    - Aussi appelé "sensibilité", "True Positive Rate" ou "couverture"
    - Complémentaire à la précision
    - Utile pour datasets déséquilibrés avec macro-moyenne
    - F1-score combine précision et rappel de manière harmonique
    """

    num_classes = len(confusion_matrix)
    recalls = []

    # Calcul du rappel pour chaque classe
    for i in range(num_classes):
        # TP pour classe i: élément diagonal
        tp = confusion_matrix[i][i]

        # FN pour classe i: somme de la ligne i sans la diagonale
        # C'est-à-dire tous les éléments de classe i prédits incorrectement
        fn = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)

        # Dénominateur: total réel d'exemples de classe i
        denom = tp + fn

        # Calcul du rappel avec gestion de division par zéro
        recalls.append(tp / denom if denom else 0.0)

    # Retour de la macro-moyenne
    return sum(recalls) / num_classes


def f1_macro(confusion_matrix):
    """
    Calcule le F1-score macro-moyenné à partir de la matrice de confusion.

    Le F1-score est la moyenne harmonique de la précision et du rappel.
    Il fournit une mesure unique qui équilibre ces deux métriques.

    Définition mathématique:
    -----------------------
    F1 = 2 × (Precision × Recall) / (Precision + Recall)

    Ou de manière équivalente:
    F1 = 2 × TP / (2×TP + FP + FN)

    Pour multi-classes (macro):
    F1_macro = 2 × (P_macro × R_macro) / (P_macro + R_macro)

    Où:
    - P_macro = precision_macro(confusion_matrix)
    - R_macro = recall_macro(confusion_matrix)

    Pourquoi moyenne harmonique?
    ----------------------------
    La moyenne harmonique pénalise les déséquilibres:

    - Si P=0.9 et R=0.9: F1 = 0.90 (bon équilibre)
    - Si P=0.9 et R=0.1: F1 = 0.18 (déséquilibre pénalisé)
    - Moyenne arithmétique: (0.9+0.1)/2 = 0.50 (trop optimiste)

    Formule de la moyenne harmonique:
    H = n / (Σ(1/x_i)) = 2 / (1/P + 1/R) = 2PR / (P+R)

    Interprétation:
    ---------------
    - F1 = 1.0: Précision et rappel parfaits
    - F1 = 0.0: Précision ou rappel nul
    - F1 proche de min(P,R): Un des deux est limitant

    Quand utiliser F1-score?
    ------------------------
    - Classes déséquilibrées (mieux que accuracy)
    - Besoin d'équilibrer précision et rappel
    - Métrique unique pour optimisation/comparaison
    - Reporting de performance standardisé

    Variantes:
    ----------
    - F1-macro: Moyenne des F1 par classe (implémenté ici via métriques macro)
    - F1-micro: F1 global après agrégation des TP/FP/FN
    - F1-weighted: Moyenne pondérée par fréquence des classes
    - F-beta: Généralisation avec paramètre β pour privilégier P ou R

    Cas particulier:
    ----------------
    Si Precision + Recall = 0, on définit F1 = 0.0
    (Arrive si TP = 0, donc aucune prédiction positive correcte)

    Parameters:
    -----------
    confusion_matrix : list of lists
        Matrice de confusion n×n
        Format: [[c00, c01, ...], [c10, c11, ...], ...]

    Returns:
    --------
    float
        F1-score macro-moyenné entre 0.0 et 1.0

    Example:
    --------
    >>> cm = [[85, 15],   # Classe 0
    ...       [10, 90]]   # Classe 1
    >>> f1 = f1_macro(cm)
    >>> # Precision_0 = 85/(85+10) = 0.894
    >>> # Recall_0 = 85/(85+15) = 0.850
    >>> # Precision_1 = 90/(90+15) = 0.857
    >>> # Recall_1 = 90/(90+10) = 0.900
    >>> # P_macro = (0.894+0.857)/2 = 0.876
    >>> # R_macro = (0.850+0.900)/2 = 0.875
    >>> # F1 = 2*0.876*0.875/(0.876+0.875) = 0.875
    >>> print(f"{f1:.3f}")
    0.875

    Notes:
    ------
    - Métrique recommandée pour évaluation de classificateurs
    - Robuste aux déséquilibres de classes avec version macro
    - Interprétation intuitive (entre 0 et 1)
    - Largement utilisé dans la littérature ML
    """

    # Calcul de la précision macro-moyennée
    p = precision_macro(confusion_matrix)

    # Calcul du rappel macro-moyenné
    r = recall_macro(confusion_matrix)

    # Calcul du dénominateur pour la moyenne harmonique
    denom = p + r

    # Calcul du F1-score avec gestion de division par zéro
    # Si P+R=0, alors F1=0 (aucune prédiction positive correcte)
    return (2 * p * r) / denom if denom else 0.0


def accuracy_macro(confusion_matrix):
    """
    Calcule l'exactitude (accuracy) globale à partir de la matrice de confusion.

    L'exactitude mesure la proportion totale de prédictions correctes,
    tous classes confondues. C'est la métrique de performance la plus intuitive
    mais peut être trompeuse pour les datasets déséquilibrés.

    Définition mathématique:
    -----------------------
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Ou de manière équivalente:
    Accuracy = Nombre de prédictions correctes / Nombre total de prédictions

    Dans la matrice de confusion:
    - Prédictions correctes = Σ(confusion_matrix[i][i]) (diagonale)
    - Total de prédictions = Σ(confusion_matrix[i][j]) (tous éléments)

    Pour multi-classes:
    Accuracy = Σ(TP_i) / Σ(n_i) pour toutes les classes i

    Limitations de l'accuracy:
    --------------------------
    1. Paradoxe de l'accuracy pour classes déséquilibrées:
       Si 95% des exemples sont de classe 0:
       - Modèle naïf prédisant toujours 0 → Accuracy = 95%
       - Mais modèle inutile (ne détecte jamais classe 1)

    2. Ne distingue pas types d'erreurs:
       - Un FP et un FN ont même impact sur accuracy
       - Mais coûts réels peuvent être très différents

    3. Ne reflète pas performance par classe:
       - Peut être élevée même si une classe est ignorée
       - Nécessite analyse de la matrice de confusion complète

    Quand utiliser l'accuracy?
    --------------------------
    - Classes équilibrées (distributions similaires)
    - Coûts d'erreurs symétriques (FP ≈ FN)
    - Métrique simple pour communication
    - Baseline rapide pour comparaison

    Alternatives recommandées:
    --------------------------
    - F1-score macro: Pour classes déséquilibrées
    - Balanced accuracy: Moyenne des rappels par classe
    - Cohen's Kappa: Corrige l'accord dû au hasard
    - Coût financier: Pour problèmes avec coûts asymétriques

    Relation avec autres métriques:
    -------------------------------
    Pour classification binaire:
    Accuracy = (TP + TN) / (P + N)
    Balanced_Accuracy = (TPR + TNR) / 2

    Où:
    - TPR (True Positive Rate) = Recall = Sensitivity
    - TNR (True Negative Rate) = Specificity

    Parameters:
    -----------
    confusion_matrix : list of lists
        Matrice de confusion n×n
        Format: [[c00, c01, ...], [c10, c11, ...], ...]

    Returns:
    --------
    float
        Exactitude globale entre 0.0 et 1.0

    Example:
    --------
    >>> cm = [[90, 10],   # Classe 0: 90 corrects, 10 erreurs
    ...       [5, 95]]    # Classe 1: 95 corrects, 5 erreurs
    >>> accuracy = accuracy_macro(cm)
    >>> # Corrects: 90 + 95 = 185
    >>> # Total: 90+10+5+95 = 200
    >>> # Accuracy: 185/200 = 0.925
    >>> print(f"{accuracy:.3f}")
    0.925

    >>> # Cas déséquilibré
    >>> cm_unbalanced = [[950, 5],   # Classe 0 (95% des exemples)
    ...                  [45, 0]]    # Classe 1 (5% des exemples)
    >>> acc = accuracy_macro(cm_unbalanced)
    >>> # Corrects: 950 + 0 = 950
    >>> # Total: 1000
    >>> # Accuracy: 0.950 (mais classe 1 jamais détectée!)
    >>> print(f"{acc:.3f}")
    0.950

    Notes:
    ------
    - Métrique par défaut dans scikit-learn
    - Simple à calculer et interpréter
    - Peut être trompeuse pour problèmes déséquilibrés
    - Compléter avec précision, rappel et F1-score
    - Pour crédit scoring, coût financier est plus pertinent
    """

    # Calcul du nombre total de prédictions correctes
    # Somme de la diagonale de la matrice de confusion
    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))

    # Calcul du nombre total de prédictions
    # Somme de tous les éléments de la matrice
    total = sum(sum(row) for row in confusion_matrix)

    # Calcul de l'exactitude avec gestion de division par zéro
    # (Cas théorique où total=0, ne devrait pas arriver en pratique)
    return correct / total if total else 0.0
