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

Nous avons aussi modifé la colonne POBP correspondant au pays de naissance en regroupant par continent. Les valeurs de la colonne POBP correspondent maintenant, par ordre croissant, à l'Europe, l'Asie, l'Amérique du Nord (hors Etats-Unis), les Etats-Unis, l'Amérique latine, l'Afrique, l'Océanie et les pays restants.<br>
Il aurait été intéressant de séparer l'Europe en quatre régions (Nord, Sud, Est, Ouest), car beaucoup d'information est perdue en regroupant tous les pays d'Europe ensemble.

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
|  Matrice confusion   |  72713   5805 | 65305 13213 | 65587 12931 |
|                      |   5226  49308 | 13457 41077 | 14713 39821 |
|----------------------|---------------|-------------|-------------|

|  Evaluation en test  | Random Forest |   XGBoost   |   Adaboost  |
|----------------------|---------------|-------------|-------------|
|  Accuracy            |     77.71%    |    79.71%   |    79.23%   |
|----------------------|---------------|-------------|-------------|
|  Matrice confusion   |  16053  3541  | 16280  3314 | 16388  3206 |
|                      |   3874  9795  |  3435 10234 |  3703  9966 |
|----------------------|---------------|-------------|-------------|

Nous pouvons constater que Random Forest a une très bonne performance en entrainement, mais une performance plus faible en test, ce qui montre un sur-apprentissage des données d'entraînement. En revanche, Gradient Boosting et Adaboost ont des précisions similaires en entrainement et en test, ce qui suggère que ces modèles parviennent à éviter ce problème et améliorent légèrement la précision sur le jeu de test. <br>
Random Forest est aussi le modèle le plus lent à entrainer, suivi de XGBoost puis d'Adaboost, qui est le plus rapide. <br>
Par ailleurs, il y a environ le même nombre de faux positifs et de faux négatifs pour chacun des trois modèles.


## Expérimentation 2 : Optimisation des hyperparamètres
* Jeux de données utilisé :  
  * Taille ensemble d'entrainement : **133052** lignes et **9** colonnes
  * Taille ensemble de test : **33263** lignes et **9** colonnes

### Random Forest (RF)
* Processus d'entrainement : 
  * Listes des hyperparamètres testés et valeurs : 
    * n_estimators: `[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]`,
    * max_depth: `[None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]`,
    * min_samples_split: `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`
  * Nombre de plis pour la validation croisée : **5**
  * Nombre total d'entrainement : **9075**

* Résultats : 
  * Meilleurs hyperparamètres : 
    * max_depth: `14`
    * min_samples_split: `11`
    * n_estimators: `150`
    
  * Performances en entraintement : 
    * Accuracy : **83,04%**
    * Temps de calcul : **4578s (1h16)**
    * Matrice de Confusion : <br>
    `67009  11509` <br>
    `11058  43476`

  * Performance en test : 
    * Accuracy : **80,0%**
    * Temps de calcul : **4578s (1h16)**
    * Matrice de Confusion : <br>
    `16271  3323` <br>
    ` 3329  10340`

  Nous avons testé avec GridSearch un très grand nombre de combinaisons d'hyperparamètres (**9075** fit réalisés en comptant la cross validation), ne sachant pas de quelle manière restreindre les plages de valeurs des hyperparamètres. Cette recherche a nécessité un temps de calcul conséquent (environ **1h16**) et n'a permi d'améliorer que légèrementles la précision, passant de **77,71%** à **80,0%**.

### ADABOOST 
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
    * estimator: `[DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3), DecisionTreeClassifier(max_depth=4)]`
    * n_estimators: `[50, 100, 200, 400, 800]`
    * learning_rate: `[0.005, 0.01, 0.05, 0.1, 0.5, 1.0]`
  * Nombre de plis pour la validation croisée : **5**
  * Nombre total d'entrainement : **600**

* Résultats : 
  * Meilleurs hyperparamètres : 
    * estimator: `DecisionTreeClassifier(max_depth=3)`
    * n_estimators: `800`
    * learning_rate: `0.5`

  * Performances en entraintement : 
   * Accuracy : **80.12%**
   * Temps de calcul : **4566s (1h16)**
   * Matrice de Confusion : <br>
    `65285`  `13233` <br>
    `13215`  `41319`

  * Performance en test : 
   * Accuracy : **79.84%**
   * Temps de calcul : **4566s (1h16)**
   * Matrice de Confusion : <br>
    `16270`  `3324` <br>
    `3381`  `10288`

  On a effectué **600** fit ce qui nous a pris environ **1h16** et qui nous a permis de passer seulement de **79.23%** à **79.84%** malgré le temps d'entrainement. On a choisit de prendre des plages de valeurs assez larges pour nos hyperparamètres en profitant du fait que l'on en exploite que 3 sans prendre en compte trop de cas intermédiaires pour autant. Le temps d'exécution du GridSearch est quand même conséquent pour AdaBoost malgré le nombre d'entrainement.


### XGBOOST
* Processus d'entrainement : 
  * Recherche des hyperparamètres
   * Listes des hyperparamètres testés et valeurs : 
    * n_estimators: `[100, 200, 400]`
    * learning_rate: `[0.01, 0.05, 0.1]`
    * max_depth: `[2, 3, 4]`
    * min_samples_split: `[2, 5, 10]`
    * subsample: `[0.6, 0.8, 1.0]`
  * Nombre de plis pour la validation croisée : **5**
  * Nombre total d'entrainement : **1215**

* Résultats : 
  * Meilleurs hyperparamètres : 
    * n_estimators: `400`
    * learning_rate: `0.1`
    * max_depth: `4`
    * min_samples_split: `2`
    * subsample: `0.6`

  * Performances en entraintement : 
   * Accuracy : **80.89%**
   * Temps de calcul : **1849s (30min)**
   * Matrice de Confusion : <br>
    `65546`  `12972` <br>
    `12453`  `42081`

  * Performance en test : 
   * Accuracy : **80.06%**
   * Temps de calcul : **1849s (30min)**
   * Matrice de Confusion : <br>
    `16215`  `3379` <br>
    `3254`  `10415`

  On a effectué **1215** fit ce qui nous a pris environ **30min** et qui nous a permis de passer seulement de **79.79%** à **80.06%** malgré le temps d'entrainement. Nous avons choisit d'utiliser des plages de paramètres assez large pour pouvoir couvrir plusieurs cas mais en utilisant peu de valeurs intermédiaires pour accélérer l'exécution des entrainements en raison du nombre d'hyperparamètres utilisés.

## Expérimentation 3 : Comparaison des "meilleurs modèles

* Jeux de données utilisé : 
  * Taille ensemble d'entrainement : **133052** lignes et **9** colonnes
  * Taille ensemble de test : **33263** lignes et **9** colonnes

* Résultats des meilleurs modèles obtenus dans Expe 2

|  Evaluation en train | Random Forest |   XGBoost   |   Adaboost  |
|----------------------|---------------|-------------|-------------|
|  Accuracy            |     83.04%    |    80.89%   |    80.12%   |
|----------------------|---------------|-------------|-------------|
|  Temps calcul        |     4578s     |    1849s    |    4566s    |
|----------------------|---------------|-------------|-------------|
|  Matrice confusion   | 67009   11509 | 65546 12972 | 65285 13233 |
|                      | 11058   43476 | 12453 42081 | 13215 41319 |
|----------------------|---------------|-------------|-------------|

|  Evaluation en test  | Random Forest |   XGBoost   |   Adaboost  |
|----------------------|---------------|-------------|-------------|
|  Accuracy            |     80.00%    |    80.06%   |    79.84%   |
|----------------------|---------------|-------------|-------------|
|  Matrice confusion   |  16271  3323  | 16215  3379 | 16270  3324 |
|                      |  3329  10340  | 3254  10415 | 3381  10288 |
|----------------------|---------------|-------------|-------------|

* Commentaires et Analyse : 
  On remarque que les 3 modèles ont un niveau d'accuracy très proche sur le jeu de test (seulement quelques dixièmes de pourcentage d'écart). On a une légère amélioration en comparaison au modèle entrainés avec les paramètres par défaut. Cependant le temps d'exécution nécessaire pour obtenir les hyperparamètres plus pertinent est très élevé. <br>
  On peut considérer dans notre cas que le Gradiet Boosting est le modèle avec la meilleure accuracy mais aussi celui qui a nécessité le moins de temps de calcul. On peut considérer que le temps de calcul pour obtenir le meilleur modèle de Random Forest n'est pas tout a fait pertienent à comparer aux 2 autres car on a fait beaucoup plus d'entrainements cependant le résultat en terme d'accuracy est du même ordre. De plus AdaBoost **600 fit** a été bien plus long à exécuter que le Gradient Boosting **1215** alors que l'on a doublé les nombre d'entrainements.

## Expérimentation 4 : inférence sur un autre jeu de données (optionnel)
Résultats / Commentaires / Analyses : 

## Expérimentation 5 : impact de la taille du jeu de données
Résultats / Commentaires / Analyses : 

| Evaluation (test/total)||    0.1    |    0.2    |    0.3    |    0.4    |    0.5    |    0.6    |    0.7    |    0.8    |    0.9    |
|---------------|---------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Random Forest | train   |  82.54%   |  82.72%   |  82.80%   |  82.94%   |  83.15%   |  83.38%   |  83.83%   |  84.42%   |  85.45%   |
|   Accuracy    | test    |  79.97%   |  79.94%   |  80.06%   |  80.05%   |  80.10%   |  79.92%   |  79.80%   |  79.65%   |  79.29%   |
|               |         |           |           |           |           |           |           |           |           |           |
|   AdaBoost    | train   |  80.16%   |  80.10%   |  79.98%   |  79.92%   |  79.88%   |  79.74%   |  79.83%   |  79.64%   |  79.70%   |
|   Accuracy    | test    |  80.19%   |  80.00%   |  79.85%   |  80.10%   |  80.06%   |  80.09%   |  79.85%   |  79.67%   |  79.58%   |
|               |         |           |           |           |           |           |           |           |           |           |
|    XGBoost    | train   |  80.77%   |  80.86%   |  80.82%   |  80.92%   |  80.94%   |  81.05%   |  81.34%   |  81.94%   |  83.06%   |
|   Accuracy    | test    |  80.11%   |  80.05%   |  80.19%   |  80.28%   |  80.21%   |  80.19%   |  79.99%   |  79.75%   |  79.32%   |
|               |         |           |           |           |           |           |           |           |           |           |

Comme attendu, moins il y a de données d'entrainement, moins le modèle est précis. <br>
Cependant, on remarque que la précision sur le jeu de test ne diminue que très légèrement lorsque l'on réduit la taille du jeu d'entrainement. Cela suggère que le modèle est capable de généraliser assez bien même avec un nombre limité de données d'entrainement.

## Modèle choisi pour la suite : 
On choisit le modèle GradientBoosting avec les hyperparamètres optimaux que l'on a récupéré avec la méthode GridSearch. <br>
C'est le modèle qui nous donne la meilleure précision sur le jeu de test quel que soit sa taille. C'est aussi le seul modèle supporté par LIME et SHAP sans traitement supplémentaire.

## Explicabilité : "permutation feature importance"

* Explication de la méthode : pour chacune des colonnes, on mélange aléatoirement les valeurs des différentes lignes. On compare ensuite la précision obtenue avant et après mélange. Plus la précision diminue, plus la feature est importante, car utiliser la mauvaise valeur aura eu un gros impact sur la prédiction. Inversement, si utiliser une valeur aléatoire n'a que très peu d'impact sur la prédiction, cela signifie que la feature n'est pas importante.

* Résultats obtenus : 

|  Feature | Importance Mean | Importance Std |
|----------|-----------------|----------------|
|   SCHL   |    0.088920     |    0.001626    |
|   WKHP   |    0.080542     |    0.001388    |
|   AGEP   |    0.041883     |    0.001552    |
|   RELP   |    0.025041     |    0.001122    |
|    COW   |    0.009148     |    0.000893    |
|    SEX   |    0.008330     |    0.000915    |
|   POBP   |    0.007689     |    0.000846    |
|    MAR   |    0.003458     |    0.000656    |
|  RAC1P   |    0.003446     |    0.000703    |
|          |                 |                |

On remarque qu'à partir de la feature COW, l'importance des features devient très faible (diminue d'un facteur 10). Les features les plus importantes sont SCHL (niveau d'éducation) et WKHP (nombre d'heures travaillées par semaine), AGEP (âge de la personne) et RELP (relation familiale) on aussi un impact important. <br>
Cela est cohérent avec le fait que le niveau d'éducation et le nombre d'heures travaillées sont des facteurs déterminants du revenu annuel. L'âge peut aussi influencer le revenu, car les personnes plus âgées ont souvent plus d'expérience professionnelle et peuvent donc occuper des postes mieux rémunérés. La relation familiale peut aussi jouer un rôle, par exemple les personnes de référence peuvent avoir des revenus plus élevés que les autres membres du ménage.

## Explicabilité : avec LIME et SHAP

* Méthode LIME
  * Exemple(s) choisi(s) : 4932 (plot/lime/explanation_idx4932.png)
  * Résultats : <br>

|         | Feature |          |   Weight  |
|---------|---------|----------|-----------|
|  1.00 < |   RELP  | <= 2.00  | -0.115436 |
|         |   POBP  | <= 4.00  |  0.085403 |
|         |   SEX   | <= 1.00  |  0.078005 |
| 42.00 < |   AGEP  | <= 55.00 |  0.075710 |
| 16.00 < |   SCHL  | <= 19.00 | -0.073660 |
|  1.00 < |   RAC1P | <= 6.00  | -0.040634 |
| 32.00 < |   WKHP  | <= 40.00 |  0.039521 |
|  1.00 < |   MAR   | <= 5.00  | -0.037421 |
|         |   COW   | <= 1.00  | -0.018505 |
|         |         |          |           |

  * Commentaires / analyses
* Méthode SHAP
  * Exemple(s) choisi(s) : 4932 (plot/shap/waterfall_idx4932.png)
  * Résultats : <br>

  | Feature | Weight |
  |---------|--------|
  | RELP    | -0.79  |
  | AGEP    | +0.52  |
  | SCHL    | -0.47  |
  | WKHP    | +0.45  |
  | MAR     | -0.23  |
  | POBP    | +0.18  |
  | SEX     | +0.17  |
  | RAC1P   | +0.06  |
  | COW     | -0.06  |
  |         |        |

  * Commentaires / analyses : 
   On remarque que les features les plus importantes selon SHAP sont RELP, AGEP, SCHL et WKHP. MAR, POBP et SEX ont une importance modérée, tandis que RAC1P et COW ont une importance faible.
* Comparaison LIME et SHAP
* Analyse summary-plot de SHAP

## Explicabilité : contrefactuelle
Résultats / Commentaires / Analyses : 


