#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:34:55 2018

@author: elamrily
"""

import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Embedding, Input
from keras.layers import Dropout, MaxPooling1D, Conv1D, Flatten, BatchNormalization
from keras.models import Model

from keras.preprocessing import text, sequence

#%% Dans ce fichier il y a les différentes fonctions utiles pour la réalisation du projet 

def data_preper(PATH_data, del_saut_line):
    
    """ Inputs : * PATH_data --> type str, chemin du dossier contenant les 10 dossiers de donnees.
                 * del_saut_line --> type bool, si True, on récupère nos donnees texte sans le saut de ligne '\n'.
        Outputs: * Data frame, Contenant deux colonnes, une correspond au textes, et l'autre au labels. """
    
    classes = []
    filenames = []
    X = []
    y = []
    
    # Parcours du dossier contenant les 10 dossiers correspondants au classes
    for i, classe in enumerate (os.listdir(PATH_data)):
        
        inputFilepath = classe
        filename_w_ext = os.path.basename(inputFilepath)
        filename, file_extension = os.path.splitext(filename_w_ext)
        classes.append(filename)
        path = PATH_data + '/' + filename

        # Parcours de chaque dossier pour extraire le texte de chaque fichier
        for j, element in enumerate (os.listdir(path)):
            inputFilepath = element
            filename_w_ext = os.path.basename(inputFilepath)
            filename, file_extension = os.path.splitext(filename_w_ext)
            filenames.append(filename)
            path_file = path + '/' + element
            
            file = open(path_file,'r',encoding='utf8').read()
            if del_saut_line == True:
                text = ""
                for k, line in enumerate(file.split("\n")):
                    content = line.split()
                    text += (" ".join(content[1:]))
                X.append(text)
                y.append(classes[i])
                
            if del_saut_line == False:
                X.append(file)
                y.append(classes[i])

    y = np.asarray(y)
    X = np.asarray(X)
    
    # create a dataframe using texts and lables
    df = pd.DataFrame()
    df['text'] = X
    df['label'] = y
    
    return(df)

def BoW(df,X_train,X_val,X_test):
    
    """ Inputs : * Donnees entrainement, validation et test
        Outputs: * Document vectors des Donnees """
    
    # Create document vectors
    vectorizer = CountVectorizer(max_features=3000,analyzer='word', token_pattern=r'\w{1,}')
    vectorizer.fit(df['text'])
    
    X_train_counts = vectorizer.transform(X_train)
    X_val_counts = vectorizer.transform(X_val)
    X_test_counts = vectorizer.transform(X_test)
    
    return(X_train_counts,X_val_counts,X_test_counts)

def tfidf(df,X_train,X_val,X_test):
    
     """ Inputs : * Donnees entrainement, validation et test
         Outputs: * tfidf transformation des Donnees """
        
    # N-gram level tf-idf:
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,4), max_features=1000)
    tfidf_vect.fit(df['text'])
    
    X_train_tf =  tfidf_vect.transform(X_train)
    X_val_tf =  tfidf_vect.transform(X_val)
    X_test_tf = tfidf_vect.transform(X_test)
    
    return(X_train_tf,X_val_tf,X_test_tf)
    
def naive_bayes(X_train,X_val,X_test,y_train,y_val,y_test):
    
     """ Inputs : * Donnees entrainement, validation et test avec leurs labels correspondants
         Outputs: * score de validation et prediction des classes des donnees de test """
        
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    score_val = clf.score(X_val,y_val)
    print(' - Validation du classifieur: \n')
    print('Score : ',score_val)
    
    # predict test classes:
    print('\n - Prediction des classes et test sur les donnees de test: \n')
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('matrice de confusion naive bayes: \n',confusion_matrix(y_test, y_pred))
    
    return(score_val,y_pred)

def linear_classifier(X_train,X_val,X_test,y_train,y_val,y_test):
    
    """ Inputs : * Donnees entrainement, validation et test avec leurs labels correspondants
        Outputs: * score de validation et prediction des classes des donnees de test """
         
    lc = linear_model.LogisticRegression()
    lc.fit(X_train,y_train)
    score_val = lc.score(X_val,y_val)
    print(' - Validation du classifieur: \n')
    print('Score : ',score_val)
    
    # predict test classes:
    print('\n - Prediction des classes et test sur les donnees de test: \n')
    y_pred = lc.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('matrice de confusion linear classifier: \n',confusion_matrix(y_test, y_pred))
    
    return(score_val,y_pred)
    
def bagging(X_train,X_val,X_test,y_train,y_val,y_test):
    
    """ Inputs : * Donnees entrainement, validation et test avec leurs labels correspondants
        Outputs: * score de validation et prediction des classes des donnees de test """
         
    # train a bagging model:
    clf = tree.DecisionTreeClassifier(max_depth=100)      # 100 arbres
    bagging = BaggingClassifier(clf,max_samples=0.5, max_features=0.5)
    bagging.fit(X_train,y_train)
    score_val = bagging.score(X_val,y_val)
    print('score : ',score_val)
    
    # predict test classes:
    y_pred = bagging.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('matrice de confusion : \n',confusion_matrix(y_test, y_pred))
    
    return(score_val,y_pred)
    
def get_train_test(train_raw_text, test_raw_text):
    
    """ transformation des donnees de test et entrainement pour etre conforme à la classification cnn """
    
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES, )

    tokenizer.fit_on_texts(list(train_raw_text))
    train_tokenized = tokenizer.texts_to_sequences(train_raw_text)
    test_tokenized = tokenizer.texts_to_sequences(test_raw_text)
    return sequence.pad_sequences(train_tokenized, maxlen=MAX_TEXT_LENGTH), \
           sequence.pad_sequences(test_tokenized, maxlen=MAX_TEXT_LENGTH)

def get_model():

    """ Definition du model cnn utilise """
         
    inp = Input(shape=(MAX_TEXT_LENGTH,))
    model = Embedding(MAX_TEXT_LENGTH, EMBED_SIZE)(inp)
    
    model = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(model)
    model = MaxPooling1D(pool_size=3)(model)
    model = BatchNormalization(axis=1)(model)
    model = Dropout(0.25)(model)
    
    model = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(model)
    model = MaxPooling1D(pool_size=5)(model)
    model = BatchNormalization(axis=1)(model)
    model = Dropout(0.3)(model)
    
    model = Flatten()(model)
    model = Dense(1024, activation="relu")(model)
    model = Dense(10, activation="softmax")(model)
    
    model = Model(inputs=inp, outputs=model)
    
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    
    return model
