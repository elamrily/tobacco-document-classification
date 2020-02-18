
# Classification des documents du procès des groupes américains du tabac

## Contexte

Le gouvernement américain a attaqué en justice cinq grands groupes américains du tabac pour avoir amassé d'importants bénéfices en mentant sur les dangers de la cigarette. Le cigarettiers  se sont entendus dès 1953, pour "mener ensemble une vaste campagne de relations publiques afin de contrer les preuves de plus en plus manifestes d'un lien entre la consommation de tabac et des maladies graves".
Dans ce procès 14 millions de documents ont été collectés et numérisés. Afin de faciliter l'exploitation de ces documents par les avocats, nous allons mettre en place une classification automatique des types de documents.
Un échantillon aléatoire des documents a été collecté et des opérateurs ont classé les documents dans des répertoires correspondant aux classes de documents : lettres, rapports, notes, email, etc. Nous avons à notre disposition :

- les images de documents : http://data.teklia.com/Images/Tobacco3482.tar.gz
- le texte contenu dans les documents obtenu par OCR (reconnaissance automatique) : Tobacco3482-OCR.tar.gz  (dans ce git)
- les classes des documents définies par des opérateurs : Tobacco3482.csv (dans ce git)

## Bibliothèques nécessaires

Nous aurons besoin des bibliothèques requises pour exécuter ce code - installation à leurs liens officiels individuels.
- pandas
- numpy
- sklearn
- keras

## Préparation des données

- Dans notre cas, les données ne sont pas disponibles au format CSV (il y a que les labels). Nous avons 10 dossiers dans le répértoire de données, chaque dossier correspond à une classe de documents et contient les fichiers textes de la classe correspondante. Nous allons donc d'abord parcourir la structure de répertoires et créer un ensemble de données puis préparer un DataFrame.
   
- Ensuite, nous allons séparer le jeu de données en ensembles d'apprentissage et de validation afin de pouvoir former et tester les classifieurs.
- Nous allons coder notre colonne de labels afin qu'elle puisse être utilisée dans des modèles d'apprentissage automatique.
    
    - Nombre de donnees:  3482
    - X_train:  (2819,)
    - X_test:  (349,)
    - X_val:  (314,)
    

## Feature Engineering

Dans cette étape, les données de texte brutes seront transformées en vecteurs d'entités et de nouvelles entités seront créées à l'aide du jeu de données existant. Nous allons implémenter les différentes idées suivantes afin d’obtenir des caractéristiques pertinentes de notre jeu de données.

- Count Vectors as features : Count Vector est une notation matricielle du jeu de données dans laquelle chaque ligne représente un document du corpus, chaque colonne représente un terme du corpus et chaque cellule représente le nombre de fréquences d'un terme particulier dans un document particulier.

- TF-IDF Vectors as features : Le score TF-IDF représente l'importance relative d'un terme dans le document et de l'ensemble du corpus. Le score TF-IDF est composé de deux termes: le premier calcule la fréquence de terme normalisée (TF) normalisée, le second terme est la fréquence de document inverse (IDF), calculée comme le logarithme du nombre de documents dans le corpus divisé par le nombre des documents où le terme spécifique apparaît.

    - TF (t) = (Nombre de fois que le terme t apparaît dans un document) / (Nombre total de termes dans le document)
    - IDF (t) = log_e (Nombre total de documents / Nombre de documents contenant le terme t)



## Model Building 

La dernière étape du cadre de classification de texte consiste à former un classificateur à l'aide des fonctionnalités créées à l'étape précédente. Il existe de nombreux choix de modèles d'apprentissage automatique pouvant être utilisés pour former un modèle final. Nous allons implémenter différents classificateurs suivants:

- Naive Bayes Classifier
- Linear Classifier
- Bagging Models
- Convolutional Neural Network (CNN)

#### Naive Bayes Classifier

Naive Bayes est une technique de classification basée sur le théorème de Bayes avec une hypothèse d’indépendance parmi les prédicteurs. Un classificateur Naive Bayes suppose que la présence d'une caractéristique particulière dans une classe n'est pas liée à la présence d'une autre caractéristique.

- Avec Count Vectors representation  

    - Score de validation :  0.73
    
    - Prediction des classes et test sur les donnees de test: 
     
                 precision    recall  f1-score   support
    
              0       0.70      0.70      0.70        23
              1       0.98      0.94      0.96        63
              2       0.74      0.90      0.81        48
              3       0.81      0.68      0.74        65
              4       0.66      0.78      0.71        51
              5       0.71      0.83      0.77        18
              6       0.38      0.33      0.36        15
              7       0.60      0.46      0.52        26
              8       1.00      1.00      1.00        15
              9       0.67      0.64      0.65        25
              avg / total       0.76      0.76      0.76       349
              
              matrice de confusion : 
              [[16  0  0  0  2  0  4  1  0  0]
              [ 0 59  0  1  2  1  0  0  0  0]
              [ 2  0 43  1  0  0  1  0  0  1]
              [ 0  1  5 44  5  4  1  4  0  1]
              [ 0  0  2  6 40  0  1  2  0  0]
              [ 2  0  0  0  0 15  1  0  0  0]
              [ 2  0  3  0  5  0  5  0  0  0]
              [ 0  0  1  2  4  1  0 12  0  6]
              [ 0  0  0  0  0  0  0  0 15  0]
              [ 1  0  4  0  3  0  0  1  0 16]]
    

- Avec TF-IDF Vectors representation
    
    - Score de validation :  0.72
    
    - Prediction des classes et test sur les donnees de test: 
    
                 precision    recall  f1-score   support
    
              0       0.84      0.70      0.76        23
              1       0.97      0.95      0.96        63
              2       0.65      0.96      0.77        48
              3       0.80      0.74      0.77        65
              4       0.64      0.80      0.71        51
              5       0.79      0.83      0.81        18
              6       1.00      0.07      0.12        15
              7       0.58      0.27      0.37        26
              8       1.00      1.00      1.00        15
              9       0.65      0.68      0.67        25
              avg / total       0.78      0.76      0.74       349
              
              matrice de confusion : 
              [[16  0  3  0  4  0  0  0  0  0]
              [ 0 60  0  1  1  1  0  0  0  0]
              [ 1  0 46  1  0  0  0  0  0  0]
              [ 0  1  4 48  6  0  0  3  0  3]
              [ 0  0  3  6 41  0  0  1  0  0]
              [ 2  0  1  0  0 15  0  0  0  0]
              [ 0  0  8  1  5  0  1  0  0  0]
              [ 0  1  1  3  5  3  0  7  0  6]
              [ 0  0  0  0  0  0  0  0 15  0]
              [ 0  0  5  0  2  0  0  1  0 17]]
    

#### Linear Classifier

La régression logistique mesure la relation entre la variable dépendante catégorielle et une ou plusieurs variables indépendantes en estimant les probabilités à l'aide d'une fonction logistique / sigmoïde.

- Avec Count Vectors representation
    
    - Score de validation :  0.79

    - Prediction des classes et test sur les donnees de test: 
    
                 precision    recall  f1-score   support
    
              0       0.79      0.65      0.71        23
              1       0.94      0.98      0.96        63
              2       0.83      0.92      0.87        48
              3       0.84      0.78      0.81        65
              4       0.75      0.84      0.80        51
              5       0.88      0.83      0.86        18
              6       0.57      0.80      0.67        15
              7       0.75      0.46      0.57        26
              8       1.00      1.00      1.00        15
              9       0.67      0.64      0.65        25
              avg / total       0.82      0.82      0.81       349
              
              matrice de confusion : 
              [[15  1  1  0  1  0  5  0  0  0]
              [ 0 62  0  1  0  0  0  0  0  0]
              [ 1  0 44  1  0  0  2  0  0  0]
              [ 1  1  1 51  7  1  0  1  0  2]
              [ 1  0  0  6 43  0  1  0  0  0]
              [ 1  1  0  0  0 15  0  0  0  1]
              [ 0  0  2  0  1  0 12  0  0  0]
              [ 0  1  2  2  3  1  0 12  0  5]
              [ 0  0  0  0  0  0  0  0 15  0]
              [ 0  0  3  0  2  0  1  3  0 16]]
    

- Avec TF-IDF Vectors representation

    - Score de validation :  0.77
    
    - Prediction des classes et test sur les donnees de test:
    
                 precision    recall  f1-score   support
    
              0       0.76      0.57      0.65        23
              1       0.95      0.95      0.95        63
              2       0.72      0.88      0.79        48
              3       0.87      0.74      0.80        65
              4       0.72      0.84      0.77        51
              5       0.76      0.89      0.82        18
              6       0.56      0.67      0.61        15
              7       0.64      0.35      0.45        26
              8       1.00      1.00      1.00        15
              9       0.61      0.68      0.64        25
              avg / total       0.79      0.78      0.78       349
              
              matrice de confusion : 
              [[13  0  2  0  3  1  4  0  0  0]
              [ 0 60  0  1  1  1  0  0  0  0]
              [ 2  0 42  0  0  0  3  0  0  1]
              [ 1  1  4 48  5  1  0  3  0  2]
              [ 0  1  1  5 43  0  0  1  0  0]
              [ 1  0  1  0  0 16  0  0  0  0]
              [ 0  0  3  0  2  0 10  0  0  0]
              [ 0  1  1  1  4  2  0  9  0  8]
              [ 0  0  0  0  0  0  0  0 15  0]
              [ 0  0  4  0  2  0  1  1  0 17]]
    

##### Bagging Models

Les modèles Random Forest sont un type de modèle d'ensemble, en particulier les modèles du bagging. Ils font partie de la famille de modèles basés sur des arbres.
- Avec Count Vectors representation

    - Score de validation :  0.74
    
    - Prediction des classes et test sur les donnees de test:
    
                 precision    recall  f1-score   support
    
              0       0.70      0.70      0.70        23
              1       0.91      0.97      0.94        63
              2       0.65      0.83      0.73        48
              3       0.80      0.78      0.79        65
              4       0.81      0.76      0.79        51
              5       0.82      0.78      0.80        18
              6       0.57      0.53      0.55        15
              7       0.44      0.27      0.33        26
              8       1.00      1.00      1.00        15
              9       0.65      0.60      0.63        25
              avg / total       0.76      0.76      0.76       349
              matrice de confusion : 
              [[16  0  2  1  0  0  3  1  0  0]
              [ 0 61  0  1  0  0  0  1  0  0]
              [ 1  2 40  0  1  0  1  2  0  1]
              [ 0  1  4 51  5  0  1  2  0  1]
              [ 1  2  2  6 39  0  0  1  0  0]
              [ 2  0  2  0  0 14  0  0  0  0]
              [ 2  1  3  1  0  0  8  0  0  0]
              [ 0  0  5  3  2  3  0  7  0  6]
              [ 0  0  0  0  0  0  0  0 15  0]
              [ 1  0  4  1  1  0  1  2  0 15]]
     

Selon les résultats obtenus on remarque que l'extraction des caractéristique à l'aide du count vectors est mieux que le tf-idf.

#### Reseau de neuronnes convoltionnel (CNN)

Dans les réseaux neuronaux convolutifs, les convolutions sur la couche d'entrée sont utilisées pour calculer la sortie. Il en résulte des connexions locales, où chaque région de l’entrée est connectée à un neurone dans la sortie. Chaque couche applique des filtres différents et combine leurs résultats.

Model utilisé :

    _________________________________________________________________
     Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 500)               0         
    _________________________________________________________________
    embedding_1 (Embedding)      (None, 500, 32)           16000     
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 500, 64)           14400     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 166, 64)           0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 166, 64)           664       
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 166, 64)           0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 166, 128)          41088     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 33, 128)           0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 33, 128)           132       
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 33, 128)           0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 4224)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              4326400   
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                10250     
    =================================================================
    
    Test Accuracy: 0.76
    
                 precision    recall  f1-score   support
    
              0       0.69      0.48      0.56        23
              1       0.98      0.94      0.96        63
              2       0.73      0.85      0.79        48
              3       0.84      0.66      0.74        65
              4       0.71      0.90      0.79        51
              5       0.79      0.83      0.81        18
              6       0.55      0.73      0.63        15
              7       0.42      0.42      0.42        26
              8       1.00      0.93      0.97        15
              9       0.68      0.60      0.64        25
    
    avg / total       0.77      0.76      0.76       349
    
    
## Amélioration 

Pour obtenir une bonne précision, certaines améliorations peuvent être apportées au cadre général. Par exemple, voici quelques methodes pour améliorer les performances des modèles de classification de texte.

1. Nettoyage de texte: le nettoyage de texte peut aider à réduire le bruit présent dans les données de texte sous forme de mots vides, de signes de ponctuation, de variations de suffixe, etc.

2. Optimisation du paramétrage Hyperparamter dans la modélisation: Le réglage des paramètres est une étape importante. Plusieurs paramètres tels que la longueur de l'arbre, les feuilles, les paramètres réseau, etc. peuvent être ajustés pour obtenir le meilleur modèle.

3. Modèles d'ensemble: empiler différents modèles et mélanger leurs sorties peut aider à améliorer les résultats.