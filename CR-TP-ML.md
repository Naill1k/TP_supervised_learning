#######################################################################
#  Binome : ALINOT Killian / SIRI Baptiste
# #######################################################################

## Jeu de données : pré-traitement

Le jeu de données utilisé est l'ACSIncome, restreint à l'état de Californie.

Les colonnes du dataset sont :
* AGEP : âge de la personne (seules les personnes de plus de 16 ans sont dans le dataset)
* COW : catégorie d'emploi
* SCHL : niveau d'éducation
* MAR : statut marital
* OCCP : code métier (colonne retirée lors du pré-traitement)
* POBP : pays de naissance (colonne modifiée lors du pré-traitement)
* RELP : relation avec la personne de référence du ménage (colonne modifiée lors du pré-traitement)
* WKHP : nombre d'heures travaillées par semaine
* SEX : sexe de la personne
* RAC1P : origine ethnique de la personne

Lors du pré-traitement, nous avons décidé de retirer la colonne OCCP correspondant aux codes métiers en raison du trop grand nombre de valeurs différentes.

Nous avons aussi modifé la colonne POBP correspondant au pays de naissance en regroupant par continent. Les valeurs de la colonne POBP correspondent maintenant, par ordre croissant, à l'Europe, l'Asie, l'Amérique du Nord (hors Etats-Unis), les Etats-Unis, l'Amérique latine, l'Afrique, l'Océanie et les pays restants.

Enfin, nous avons simplifié la colonne RELP correspondant à la relation avec la personne de référence du ménage en regroupant les enfants biologiques, adoptés, les beaux enfants, les petits enfants et ceux issus de famille d'accueil en une seule catégorie (2).

Ces 9 features sont utilisées pour prédire si le label PINCP (correspondant au revenu annuel) est supérieur ou inférieur à 50 000$.

## Expérimentation 1 : Comparaison de modèles par défaut

* Jeux de données utilisé : 
  * Taille ensemble d'entrainement (nb lignes et nb colonnes) : 133052 lignes et 9 colonnes
  * Taille ensemble de test (nb lignes et nb colonnes) : 33263 lignes et 9 colonnes

* Résultats (hyper-paramètres par défaut)

|  Evaluation en train | Random Forest |   XGBoost   |   Adaboost  |
|----------------------|---------------|-------------|-------------|
|  Accuracy            |     91.71%    |    79.96%   |    79.22%   |
|----------------------|---------------|-------------|-------------|
|  Temps calcul        |     13.97s    |    7.35s    |    3.24s    |
|----------------------|---------------|-------------|-------------|
|  Matrice confusion   |   72713  5805 | 65305 13213 | 65587 12931 |
|                      |   5226  49308 | 13457 41077 | 14713 39821 |
|----------------------|---------------|-------------|-------------|

|  Evaluation en test  | Random Forest |   XGBoost  |  Adaboost  |
|----------------------|---------------|------------|------------|
|  Accuracy            |     77.71%    |   79.71%   |   79.23%   |
|----------------------|---------------|------------|------------|
|  Matrice confusion   |   16053  3541 | 16280 3314 | 16388 3206 |
|                      |   3874  9795  | 3435 10234 | 3703 9966  |
|----------------------|---------------|------------|------------|

* Commentaires et Analyse : c bizarre !


## Expérimentation 2 : Comparaison Modèles ML par défaut
* Jeux de données utilisé :  
  * Taille ensemble d'entrainement (nb lignes et nb colonnes) : 133052 lignes et 9 colonnes
  * Taille ensemble de test (nb lignes et nb colonnes) : 33263 lignes et 9 colonnes

### Random Forest (RF)
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
    * n_estimators: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    * max_depth: [None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    * min_samples_split: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  * Nombre de plis pour la validation croisée : 5
  * Nombre total d'entrainement : 9075
* Résultats : 
  * Meilleurs hyperparamètres : 
    * max_depth: 14
    * min_samples_split: 11
    * n-estimators: 150
  * Performances en entraintement : 
   * Accuracy : 83,04%
   * Temps de calcul : 4578,441
   * Matrice de Confusion : 
    67009  11509
    11058  43476
  * Performance en test : 
   * Accuracy : 80,0%
   * Temps de calcul : 4578,441
   * Matrice de Confusion : 
    16271  3323
    3329  10340
  * Commentaires / analyses (par rapport résultat expe 1)

### ADABOOST 
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
    * estimator: [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=2),
            DecisionTreeClassifier(max_depth=3),
            DecisionTreeClassifier(max_depth=4),
        ]
    * n_estimators: [50, 100, 200, 400, 800]
    * learning_rate: [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
  * Nombre de plis pour la validation croisée : 5
  * Nombre total d'entrainement : 600
* Résultats : 
  * Meilleurs hyperparamètres : 
    * estimator: DecisionTreeClassifier(max_depth=3)
    * n_estimators: 800
    * learning_rate: 0.5
  * Performances en entraintement : 
   * Accuracy : 80.12%
   * Temps de calcul : 4566s (1h16)
   * Matrice de Confusion : 
    65285  13233
    13215  41319
  * Performance en test : 
   * Accuracy : 79.84%
   * Temps de calcul : 4566s (1h16)
   * Matrice de Confusion : 
    16270  3324
    3381  10288
  * Commentaires / analyses (par rapport résultat expe 1)


### XGBOOST
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
    * n_estimators: [100, 200, 400]
    * learning_rate: [0.01, 0.05, 0.1]
    * max_depth: [2, 3, 4]
    * min_samples_split: [2, 5, 10]
    * subsample: [0.6, 0.8, 1.0]
  * Nombre de plis pour la validation croisée : 5
  * Nombre total d'entrainement : 1215
* Résultats : 
  * Meilleurs hyperparamètres : 
    * n_estimators: 400
    * learning_rate: 0.1
    * max_depth: 4
    * min_samples_split: 2
    * subsample: 0.6
  * Performances en entraintement : 
   * Accuracy : 80.89%
   * Temps de calcul : 1849s (30min)
   * Matrice de Confusion : 
    65546  12972
    12453  42081
  * Performance en test : 
   * Accuracy : 80.06%
   * Temps de calcul : 1849s (30min)
   * Matrice de Confusion : 
    16215  3379
    3254  10415
  * Commentaires / analyses (par rapport résultat expe 1)

## Expérimentation 3 : Comparaison des "meilleurs modèles

* Jeux de données utilisé : 
  * Taille ensemble d'entrainement (nb lignes et nb colonnes) : 
  * Taille ensemble de test (nb lignes et nb colonnes) : 

* Résultats des meilleurs modèles obtenus dans Expe 2

|  Evaluation en train | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            |               |          |         |
|----------------------|---------------|----------|---------|
|  Temps calcul        |               |          |         |
|----------------------|---------------|----------|---------|
|  Matrice confusion   |               |          |         |
|----------------------|---------------|----------|---------|

|   Evaluation en test | Random Forest | Adaboost | XGBoost |
|----------------------|---------------|----------|---------|
|  accuracy            |               |          |         |
|----------------------|---------------|----------|---------|
|  Temps calcul        |               |          |         |
|----------------------|---------------|----------|---------|
|  Matrice confusion   |               |          |         |
|----------------------|---------------|----------|---------|

* Commentaires et Analyse : 

## Expérimentation 4 : inférence sur un autre jeu de données (optionnel)
Résultats / Commentaires / Analyses : 

## Expérimentation 5 : impact de la taille du jeu de données
Résultats / Commentaires / Analyses : 

## Modèle choisi pour la suite : 
* quel modèle : 
* pourquoi ? 

## Explicabilité : "permutation feature importance"

* Résultats obtenus : 
* Analyses :

## Explicabilité : avec LIME et SHAP

* Méthode LIME
  * Exemple(s) choisi(s)
  * Résultats
  * Commentaires / analyses
* Méthode SHAP
  * Exemple(s) choisi(s)
  * Résultats
  * Commentaires / analyses
* Comparaison LIME et SHAP
* Analyse summary-plot de SHAP

## Explicabilité : contrefactuelle
Résultats / Commentaires / Analyses : 


