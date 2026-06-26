#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:39:12 2018

@author: slowking
"""

import pandas as pd
import numpy as np

import prince

import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 

import tensorflow as tf

from tensorflow.python.keras import models

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout


Pokedex_Master = pd.read_csv("pokedex.csv")
Combat_Master = pd.read_csv("combats_test.csv")
Stats_Master = pd.read_csv("pokemon.csv")
Attacker = pd.read_csv("chart.csv")
Evolution = pd.read_csv("pokemon_species.csv")


Attacker['None'] = 1

Attacker_Transpose = Attacker.T
Attacker_Transpose.columns = Attacker_Transpose.iloc[0]
Attacker_Transpose = Attacker_Transpose.drop('Attacking')
Attacker_Transpose['None'] = 1

Stats_Master.columns = [c.replace(' ', '_') for c in Stats_Master.columns]
Stats_Master.columns = [c.replace('.', '') for c in Stats_Master.columns]
Stats_Master[['Type_1','Type_2']] = Stats_Master[['Type_1','Type_2']].fillna(value='None')

Stats_Master.loc[62,'Name'] = 'Primeape'
Stats_Master['Name'] = Stats_Master['Name'].str.lower()


def Attacker_Expand(Attacker_Transpose, Round, Count):
    colname1 = Attacker_Transpose.columns[Round]
    colname2 = Attacker_Transpose.columns[Count]
    
    name = colname1 + colname2
    
    multi = Attacker_Transpose[colname1].multiply(Attacker_Transpose[colname2], axis='index') 
    column = multi.to_frame(name)
    
    return column

for x in range(0, 19):
    for y in range(0, 19):
        column_add = Attacker_Expand(Attacker_Transpose, x, y)
        Attacker_Transpose = pd.concat([Attacker_Transpose, column_add], axis=1)
        

def Type_Conversion(Stats_Master):
    
    PCA_Master = Stats_Master.copy()
    
    PCA_Type = PCA_Master[['Type_1','Type_2']].fillna(value='None')

    mca = prince.MCA(n_components=15, n_iter=100, copy=False, engine='auto', random_state=42)    
    Typed_MCA = mca.fit(PCA_Type)    
    print(np.sum(Typed_MCA.explained_inertia_))
    
    mca_Components = Typed_MCA.U_
    df = pd.DataFrame(mca_Components,index=mca_Components[:,0]).reset_index()
    df = df.drop(columns=['index'])
    
    return df

PCA_Types = Type_Conversion(Stats_Master)
Stats_Expand_Types = pd.concat([PCA_Types, Stats_Master], axis=1)

def Mega_Stage(Stats_Master, Evolution):
    Stats_Master_Mega = Stats_Master.copy()
    is_mega = Stats_Master_Mega.Name.str.startswith('Mega ')
    Stats_Master_Mega['is_Mega'] = np.where(is_mega == True, 1, 0)
    Stats_Master_Mega = pd.merge(Stats_Master_Mega, Evolution, left_on='Name', right_on='identifier', how='left')
    Stas_Evo = Stats_Master_Mega.sort_values(by=['evolution_chain_id', '#']).reset_index(drop=True)
                                                 
    previous = list()
    evolution = list()
    previous.append(1)
    previous.append(1)
    evolution.append(1)
    evolution.append(2)
                                                 
    for r in range (2,800):
        e = 1
        bb = Stas_Evo.is_baby[r]
        p = Stas_Evo.evolution_chain_id[r]
        previous.append(p)
        if previous[r] == previous[r-1]:
            e = e + 1
        if previous[r] == previous[r-2]:
            e = e + 1
        if bb == 1:
            e = 0
        evolution.append(e)  
    Stas_Evo['Stage'] = evolution
    Stas_Evo.fillna(0, inplace=True)      
    Stas_Evo = Stas_Evo[['Name','is_Mega','is_baby', 'Stage']]  
    return Stas_Evo

Evo = Mega_Stage(Stats_Master, Evolution)
Stats_Expand_Types = pd.merge(Stats_Expand_Types, Evo, left_on='Name', right_on='Name', how='left')


def Pokedex_Combat_Data_Cleaning(Combat_Master, Stats_Expand_Types, Attacker):
    
    Stats_Master_Mod = Stats_Expand_Types.copy()
    Stats_Master_Mod['MType'] = Stats_Master_Mod['Type_1'] + Stats_Master_Mod['Type_2']
    Stats_Master_Mod['is_Legendary'] = np.where(Stats_Master_Mod['Legendary'] == True, 1, 0)
    Stats_Master_Mod = Stats_Master_Mod.drop(columns=['Name', 'Generation', 'Legendary'])
    
    is_Combat_Eff = Combat_Master.copy()
    is_Combat_Eff['First_Winner'] = np.where(Combat_Master['First_pokemon'] == Combat_Master['Winner'], 1, 0)
    is_Combat_Eff = pd.merge(is_Combat_Eff, Stats_Master_Mod, left_on='First_pokemon', right_on='#', how='inner')
    is_Combat_Eff = pd.merge(is_Combat_Eff, Stats_Master_Mod, left_on='Second_pokemon', right_on='#', how='inner')
    
    First_Attack_Effect = list()
               
    for i in range (0, 52080):
              
        A_MT = is_Combat_Eff.MType_x[i]
        D_T1 = is_Combat_Eff.Type_1_y[i]
        D_T2 = is_Combat_Eff.Type_2_y[i]
    
        Attacker1_Combat1 = Attacker_Transpose.at[D_T1, A_MT]  
        Attacker1_Combat2 = Attacker_Transpose.at[D_T2,A_MT]
        First_Attack = Attacker1_Combat1 * Attacker1_Combat2
    
        First_Attack_Effect.append(First_Attack)
    
    Second_Attack_Effect = list()
               
    for i in range (0, 52080):
              
        A_MT = is_Combat_Eff.MType_y[i]
        D_T1 = is_Combat_Eff.Type_1_x[i]
        D_T2 = is_Combat_Eff.Type_2_x[i]
    
        Attacker2_Combat1 = Attacker_Transpose.at[D_T1, A_MT]  
        Attacker2_Combat2 = Attacker_Transpose.at[D_T2,A_MT]
        Second_Attack = Attacker2_Combat1 * Attacker2_Combat2
    
        Second_Attack_Effect.append(Second_Attack)    
    
    is_Combat_Eff['First_Attacker_Eff'] = First_Attack_Effect
    is_Combat_Eff['Second_Attacker_Eff'] = Second_Attack_Effect
    
    return is_Combat_Eff
                       
#This should be a function
      
Combatted = Pokedex_Combat_Data_Cleaning(Combat_Master, Stats_Expand_Types, Attacker)

Combatted = Combatted.drop(columns=['Winner','#_x','Type_1_x', 'Type_2_x','MType_x','#_y','Type_1_y', 'Type_2_y','MType_y'])
Combatted['HP_Delta'] = Combatted['HP_x'] - Combatted['HP_y']
Combatted['Attack_Delta'] = Combatted['Attack_x'] - Combatted['Attack_y']
Combatted['Defense_Delta'] = Combatted['Defense_x'] - Combatted['Defense_y']
Combatted['Sp_Atk_Delta'] = Combatted['Sp_Atk_x'] - Combatted['Sp_Atk_y']
Combatted['Sp_Def_Delta'] = Combatted['Sp_Def_x'] - Combatted['Sp_Def_y']
Combatted['Speed_Delta'] = Combatted['Speed_x'] - Combatted['Speed_y']
Combatted['First_Speed'] = np.where(Combatted['Speed_Delta'] >= 0, 1, 0)

Combatted = Combatted.drop(columns=['HP_x','HP_y','Attack_x', 'Attack_y','Defense_x',
'Defense_y','Sp_Atk_x', 'Sp_Atk_y','Sp_Def_x','Sp_Def_y','Speed_x','Speed_y','Speed_Delta'])

Combatted.isnull().values.any()
Combatted.isnull().sum()

Test_Data_Set = Combatted.loc[Combatted['Test_Set'] == 1].reset_index()
y_pokemon = Test_Data_Set[['First_pokemon','Second_pokemon']]
Test_Data_Set = Test_Data_Set.drop(columns=['index','First_Winner','Test_Set','First_pokemon','Second_pokemon'])
Combatted = Combatted.loc[Combatted['Test_Set'] == 0].reset_index()
Combatted = Combatted.drop(columns=['Test_Set', 'index','First_pokemon','Second_pokemon'])


X = Combatted.drop("First_Winner", axis=1)
y = Combatted["First_Winner"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
acc_random_forest    

X_train.to_csv('Train_Test_Data/X_train.csv')
X_test.to_csv('Train_Test_Data/X_test.csv')
y_train.to_csv('Train_Test_Data/y_train.csv')
y_test.to_csv('Train_Test_Data/y_test.csv')
Test_Data_Set.to_csv('Train_Test_Data/X_predict.csv')

#This should be a function

y_predict = random_forest.predict(Test_Data_Set)
y_pokemon['First_Winner'] = y_predict

y_pokemon['Second_Winner'] = np.where(y_pokemon['First_Winner'] == 1, 0, 1)
y_pokemon_final = y_pokemon[['First_pokemon','First_Winner']]
y_pokemon_final.rename(columns={'First_pokemon':'Pokemon',
                          'First_Winner':'Win'}, inplace=True)
y_pokemon_final2 =y_pokemon[['Second_pokemon','Second_Winner']]
y_pokemon_final2.rename(columns={'Second_pokemon':'Pokemon',
                          'Second_Winner':'Win'}, inplace=True)
y_pokemon_final = y_pokemon_final.append(y_pokemon_final2)
y_pokemon_group = y_pokemon_final.groupby(['Pokemon'])['Win'].sum().reset_index()
y_pokemon_group = y_pokemon_group.sort_values('Win')

y_pokemon_group.to_csv('Leg_Rankings.csv')




logreg = LogisticRegression()
logreg.fit(X_train, y_train)
acc_log = round(logreg.score(X_test, y_test)*100, 2)
acc_log


sorted_features = sorted(zip(random_forest.feature_importances_, X_train.columns), reverse=True)
unzip_sorted_features = list(zip(*sorted_features))
labels = unzip_sorted_features[1]
scores = unzip_sorted_features[0]

fig1, ax1 = plt.subplots()
fig1.suptitle("Importance of features in Random Forest algorithm")
fig1.set_figheight(10)
fig1.set_figwidth(10)
ax1.pie(scores, labels=labels, autopct='%1.2f%%',
        shadow=False, startangle=0)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


def mlp_model(layers, units, dropout_rate, input_shape):
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers-1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=1, activation='sigmoid'))
    return model

def train_ngram_model(X_train, X_test, y_train, y_test,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):

    # Create model instance.
    model = mlp_model(layers=layers,
                                  units=64,
                                  dropout_rate=dropout_rate,
                                  input_shape=X_train.shape[1:])


    loss = 'binary_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_test, y_test),
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    model.save('IMDb_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

result = train_ngram_model(X_train, X_test, y_train, y_test)

