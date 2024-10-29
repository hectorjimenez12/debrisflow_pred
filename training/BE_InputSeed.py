#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import pdb
import copy
import matplotlib.pyplot as plt
import joblib
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # naive bayes
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import random
#SEED = random.randrange(1,100000000) 
import sys


#os.chdir("G:/Mi unidad/Investigacion/2024/Otoño/FONDEF_AMTC_Probability_Of_DebrisFlow/01_Entrenamiento_Modelos/agosto2024/")

lseeds = [13235730, 48661991, 53835526, 69992231, 71453695, 76190774, 8573278,
          99650253, 49060501, 29266638, 22423211, 65008066, 44408396, 54140912,
          19053566, 75511212, 70175406, 63671815, 57427568, 68853935, 14631439,
          65306196, 80596791, 56728913, 11958782, 55182937, 66511785,  85377243,
          37146310, 71993930, 64165577, 32964420, 90891883, 14666363, 91237853,
          51442036, 75222985, 92637250, 90856233,   33950726 , 37378654, 40329400,
          70594870, 62070266, 9223258, 89732966, 72291926, 64957950, 75437863,
          60633730, 90388823, 17412684, 58914594, 68930398, 72934849, 70002947,
          69359784, 94958466, 33950726, 27533747,90188262, 90781274,59668061,
          91968778,93764107, 64896412, 73939202,94958466, 46685014, 83656397, 23534286 ] 


# First try supporting commands formatted like: script.py example.txt
if len(sys.argv) > 1:
    #with open(sys.argv[1]) as f:
    contents = sys.argv[1]
# Now try supporting: script.py < example.txt
elif sys.stdin:
    contents = ''.join(sys.stdin)
# If both methods failed, throw a user friendly error
else:
    raise Exception('Please supply an input')



SEED = lseeds[int(contents)]
random.seed(SEED)
np.random.seed(SEED)
print(SEED)

X = pd.read_csv("X.csv")
features = list(X.columns)
y = pd.read_csv("Y__DP_MP.csv")


''' Split data stratifyng by storms '''
#X2015, X2017e, X2017m, X2020 = X[ storm_ids == 0 ], X[ storm_ids == 1 ], X[ storm_ids == 2 ], X[ storm_ids == 3 ]
#Y2015, Y2017e, Y2017m, Y2020 = y[ storm_ids == 0 ], y[ storm_ids == 1 ], y[ storm_ids == 2 ], y[ storm_ids == 3 ]
#Xstorms, Ystorms = [ X2015, X2017e, X2017m, X2020 ], [ Y2015, Y2017e, Y2017m, Y2020 ] 
#Xtrains, Xvals, Xtests = [], [], []
#Ytrains, Yvals, Ytests = [], [], []

#for istorm in range(4):
#    X_train, X_test, y_train, y_test = train_test_split(Xstorms[istorm], Ystorms[istorm] , test_size = 0.2, random_state=SEED, stratify = Ystorms[istorm])
#    X_train, X_val , y_train, y_val  = train_test_split(X_train, y_train ,test_size = 10/80 ,random_state=SEED,stratify = y_train)
#    Xtrains.append(X_train)
#    Xvals.append(X_val)
#    Xtests.append(X_test)
#    Ytrains.append(y_train)
#    Yvals.append(y_val)
#    Ytests.append(y_test)

#X_train, X_val ,X_test = pd.concat( Xtrains , axis = 0), pd.concat( Xvals , axis = 0), pd.concat( Xtests , axis = 0) 
#y_train, y_val ,y_test = pd.concat( Ytrains , axis = 0), pd.concat( Yvals , axis = 0), pd.concat( Ytests , axis = 0)    

X_train80, X_test, y_train80, y_test = train_test_split(X, y ,test_size = 0.2,random_state=SEED,stratify=y)
X_train,   X_val , y_train,   y_val  = train_test_split(X_train80, y_train80 ,test_size = 20/80 ,random_state=SEED,stratify = y_train80)

#X_train, X_val ,X_test = X_train.reset_index(drop=True) , X_val.reset_index(drop=True) ,X_test.reset_index(drop=True)
#y_train, y_val ,y_test = y_train.reset_index(drop=True) , y_val.reset_index(drop=True) ,y_test.reset_index(drop=True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score

score='roc_auc'
name_models = ['LogisticRegression',#'RandomForest',
               'RandomForest',
               #'KNeighbors',
               #'GaussianNBayes',
               'SupportVectorClassifier',
               'MultiLayerPerceptron']

grid_values = [{'C': [0.01,0.1,1,2]}, 
               {'criterion': ['gini', 'entropy'], 'max_depth': [3,5],'min_samples_split':[10,50,100]},
               #{'n_neighbors': [3, 5, 10, 20] ,'weights': ['distance','uniform']},
               #{'var_smoothing': [1e-9] },
               {'kernel':['rbf','linear','sigmoid'], 'C':[0.01, 0.1, 1, 2] },
               {'hidden_layer_sizes': [50,100,150],'activation':['tanh','relu'] } ]  


models = [LogisticRegression(random_state= 7465,max_iter=500), # ,solver = 'liblinear' ),
          RandomForestClassifier(random_state=7465),
          #KNeighborsClassifier(),
          #GaussianNB(),
          SVC(random_state=7465, probability=(True)),
          MLPClassifier(random_state=7465, learning_rate_init = 0.001, solver = 'adam',max_iter=300)]

"""**Clase Modelo Backward Feature Selection**"""

class BFS_model:
    """Initialition"""
    def __init__(self, name, dgrid_values, dmodels, dXscaled, dY, metric_train = 'roc_auc'):
        self.name         = name                                       # name of the model
        self.grid_values  = dgrid_values[self.name]                     # dictionary with hyperparameters
        self.dmodels      = dmodels
        self.model        = copy.deepcopy(dmodels[self.name])          # dictionary with base model
        self.metric_train = metric_train                               # name of the metric use to train
    
        self.X = dXscaled
        self.Y = dY
        #if train_test_split:
        #    self.X['train'] = pd.concat([self.X['train'],self.X['val']], axis = 0, ignore_index = True) 
        #    self.Y['train'] = pd.concat([self.Y['train'],self.Y['val']], axis = 0, ignore_index = True)
    
        self.RobustScaling()
        self.Oversampling()
        self.features = self.X['features']
    
        cvmax = np.unique(self.Y['train'],return_counts=True)
        self.grid_model = GridSearchCV(copy.deepcopy(self.model), param_grid = self.grid_values,
                                       cv = 100 , scoring = self.metric_train, n_jobs = 30)
    
        self.mhf_hystory     = {} # metrics, hyperparameters and features
        self.select_model    = {}
        self.best_metric_model  = 0
    
    def get_model(self):
        return self.model
    def get_metrics(self):
        return self.metrics_history
    def get_hyperpar(self):
        return self.metrics_history
    
    def RobustScaling(self):
        scaler = MinMaxScaler()
        self.X['train'] = pd.DataFrame(scaler.fit_transform(self.X['train']) , columns = self.X['train'].columns, index = self.X['train'].index)
        self.X['val']   = pd.DataFrame(scaler.transform(self.X['val'])       , columns = self.X['val'].columns  , index = self.X['val'].index)
        self.X['test']  = pd.DataFrame(scaler.transform(self.X['test'])      , columns = self.X['test'].columns , index = self.X['test'].index)
    
    def Oversampling(self):
        y_train, X_train = copy.deepcopy(self.Y['train']), copy.deepcopy(self.X['train']) 
        vc =  y_train.value_counts(dropna=False)
        index_concat = np.random.choice(a = y_train[y_train['y'].isin([1])].index ,size = int( max(vc) - min(vc) ) ,replace = True)
        y_train = pd.concat([y_train,y_train.loc[index_concat,:]], axis = 0) 
        X_train = pd.concat([X_train,X_train.loc[index_concat,:]], axis = 0)
        self.Y['train'], self.Y['val'], self.Y['test'] = np.array(y_train).reshape(-1) , np.array(self.Y['val']).reshape(-1) , np.array(self.Y['test']).reshape(-1)
        self.X['train'], self.X['val'], self.X['test'] = np.array(X_train)             , np.array(self.X['val'])             , np.array(self.X['test'])

    def get_X(self,features_model):
        indexes    = [k for k in range(len( self.features )) if self.features[k] in features_model ]
        return self.X['train'][:,indexes], self.X['val'][:,indexes] ,self.X['test'][:,indexes]
    
    '''Seleccion de hiperparametros optimos de un modelo con features fijas'''
    def tune_hyper(self, features_train = None):
        grid_model_train = copy.deepcopy(self.grid_model)
        if features_train is None:
            features_train = self.features
    
        Xtrain_feat, Xval_feat, Xtest_feat = self.get_X(features_train)
    
        grid_model_train.fit(Xtrain_feat, self.Y['train'] )
        yp_train, ybin_train     = grid_model_train.predict_proba(Xtrain_feat)[:,1], grid_model_train.predict(Xtrain_feat)
        yp_val,  ybin_val        = grid_model_train.predict_proba(Xval_feat)[:,1]  , grid_model_train.predict(Xval_feat)
        yp_test, ybin_test       = grid_model_train.predict_proba(Xtest_feat)[:,1] , grid_model_train.predict(Xtest_feat)
    
        auc_train, auc_val , auc_test              = roc_auc_score(self.Y['train'], yp_train), roc_auc_score(self.Y['val'], yp_val) , roc_auc_score(self.Y['test'], yp_test)
        dresults = {'cmetrics_train': classification_report(self.Y['train'], ybin_train,output_dict=True),
                    'cmetrics_test': classification_report(self.Y['test'], ybin_test,output_dict=True), 'hyperparams':grid_model_train.best_params_,
                    'train_metric': [auc_train, auc_val , auc_test] ,'features': features_train ,'features_selection': self.features,
                    'model': grid_model_train}
        return dresults
    
    def get_dictionary_result(self,dresults,feat_del):
     
      d = {'train_metric':dresults['train_metric'] ,'metrics':dresults['cmetrics_test'],
           'hyperparams': dresults['hyperparams'], 'features':dresults['features'],
           'model':dresults ['model'],'feat_del' :feat_del}
      return d

    ''' Backward selection of features '''
    #ind_sel: indice de seleccion (0 --> training, 1 --> validacion)
    def backward_selection(self, features_train = None, iter = 1, ind_sel = 1, t = 0.05):
        if features_train is None:
            features = self.features
        else:
            features = features_train
      
        if len(features) == 1:
            return self.mhf_hystory
    
        if len(features) == len(self.features) :
            res_all_feat = self.tune_hyper(features)
            
            if isinstance(ind_sel,int):
                self.best_metric_model = res_all_feat['train_metric'][ind_sel]
            else:
                self.best_metric_model = np.mean( [res_all_feat['train_metric'][z] for z in ind_sel] )	
            self.mhf_hystory[0] = {**self.get_dictionary_result(res_all_feat,feat_del = ''), **{'type':'init'} }
            self.select_model = {**copy.deepcopy( self.mhf_hystory[0]  ), **{'iter':0} }
            print('all features: ',self.best_metric_model)
    
        best_improv, best_mhf = 0, None
        for i, feat in enumerate(features):
            train_feats = [k for k in features if k != feat ]
            res_train = self.tune_hyper(train_feats) #use all features
            
            if isinstance(ind_sel,int):  
                auc_val   = res_train['train_metric'][ind_sel]
            else:
                auc_val   =  np.mean( [res_train['train_metric'][z] for z in ind_sel] )
            
            mhf_dict = self.get_dictionary_result(res_train, feat_del = feat)
            if auc_val > best_improv:
                best_improv = auc_val
                best_mhf    = mhf_dict
      
        print(iter, best_improv)
     
        types_previous_models = [j['type'] for j in self.mhf_hystory.values()]
    
        print(types_previous_models)
    
        dif = self.best_metric_model - best_improv    
        print(dif)
        
        if best_improv >= self.best_metric_model:
            self.best_metric_model = best_improv
            self.select_model = {**copy.deepcopy(best_mhf), **{'iter':iter} }
            self.mhf_hystory[iter] = {**best_mhf, **{'type':'improv'} }
            print('New best: ',self.best_metric_model)
            return self.backward_selection(features_train = best_mhf['features'] , iter = iter + 1 ,  ind_sel = ind_sel, t = t )
        elif  dif > t or types_previous_models[-1] == 'noimprov' :
            self.mhf_hystory[iter] = {**best_mhf, **{'type':'noimprov'} }
            return self.backward_selection(features_train =best_mhf['features'] , iter = iter + 1 ,  ind_sel = ind_sel, t = t )
        else:
            self.select_model = {**copy.deepcopy(best_mhf), **{'iter':iter} }
            self.mhf_hystory[iter] = {**best_mhf, **{'type':'improv'} }
            return self.backward_selection(features_train = best_mhf['features'] , iter = iter + 1 ,  ind_sel = ind_sel, t = t )
      
          




################################    
results_models = {}
print('TRAIN')
for imodel in name_models:
    print(imodel)
    bsmodelTT = BFS_model(name         = imodel,
                          dgrid_values = {i:j for i,j in zip(name_models,grid_values)},
                          dmodels      = {i:j for i,j in zip(name_models,models)}     ,
                          dXscaled = {'train':X_train80,'val':X_train80,'test':X_test,'features':features},
                          dY       = {'train': y_train80, 'val': y_train80,  'test':y_test } )
    bsmodelTVT = BFS_model(name         = imodel,
                           dgrid_values = {i:j for i,j in zip(name_models,grid_values)},
                           dmodels      = {i:j for i,j in zip(name_models,models)}     ,
                           dXscaled = {'train':X_train,'val':X_val,'test':X_test,'features':features},
                           dY       = {'train': y_train, 'val':y_val,  'test':y_test } )
    dtt  = bsmodelTT.backward_selection(ind_sel = 0 ,t= 0.025)
    dtvt = bsmodelTVT.backward_selection(ind_sel = 1 ,t= 0.025)
    
    
    results_models[imodel] = [ [bsmodelTT, dtt], [ bsmodelTVT, dtvt ] ]
  
   


#joblib.dump( results_models , imodel + str(SEED) + "joblib.pkl") 
  

import pickle
with open('ModelResults'+ str(SEED) +'.pickle', 'wb') as handle:
    pickle.dump(results_models, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""# Figura 2 (Numero de predictores vs Rendimiento de modelo)

"""
'''


def getplotresults(d):
  test     = [i['train_metric'][2] for i in d.values()]
  training = [i['train_metric'][0] for i in d.values()]
  val      = [i['train_metric'][1] for i in d.values()]

  return training, val, test

import matplotlib.pyplot as plt
import string

name_models = ['LR',#'RandomForest',
               'RF',#'KNN',
               'GNB','SVC', 'MLP']

letters = string.ascii_letters
feats_letters = { features[k] : letters[k] for k in range(len(features)) }
letters_feats = { letters[k]  : features[k] for k in range(len(features)) }

props_orange   = dict(boxstyle='round', facecolor='orange', alpha=0.2)
props_gray   = dict(boxstyle='round', facecolor='gray', alpha=0.2)

fig, axs = plt.subplots(nrows = len(results_models) , ncols = 2, figsize = (8,12), sharey = True,sharex=True)
dres = list(results_models.values())
k = 0
plt.subplots_adjust(wspace=0.1,hspace=0.1)
for row in range(len(results_models)):
  if k < len(dres):

    dTT, dTVT = dres[k][0][1], dres[k][1][1]
    objTT, objTVT = dres[k][0][0], dres[k][1][0]

    trainTT, valTT, testTT = getplotresults(dTT)
    trainTVT, valTVT, testTVT = getplotresults(dTVT)

    axs[row,0].plot( trainTT , linestyle='--', marker='x', color='#000000', label='Train' )
    axs[row,0].plot( testTT , linestyle='--', marker='o', color='#0072B2', alpha=0.4)

    axs[row,1].plot( trainTVT , linestyle='--', marker='x', color='#000000', label='Train' )
    axs[row,1].plot( valTVT , linestyle='--', marker='s', color='#D55E00', label='Val (BE)' )
    axs[row,1].plot( testTVT , linestyle='--', marker='o', color='#0072B2', alpha=0.4, label='Test')

    plt.xticks(range(0, len(dTT)) )
    axs[row,1].set_xticklabels( [str(j) for j in list(range(1, len(dTT)+1 ))[::-1]]  , rotation = 0)
    axs[row,0].set_xticklabels( [str(j) for j in list(range(1, len(dTT)+1 ))[::-1]]  , rotation = 0)

    best_modelTT, best_modelTVT  = objTT.select_model, objTVT.select_model

    xvertTT, xvertTVT = best_modelTT['iter'], best_modelTVT['iter']
    axs[row,0].axvline(x = xvertTT , color = 'r', linestyle = '--',alpha=0.5)
    axs[row,1].axvline(x = xvertTVT , color = 'r', linestyle = '--',alpha=0.5)

    #axs[row,0].plot( xvertTT, testTT[xvertTT] ,markersize=8 ,linestyle = '',marker='o', color='#0072B2', label='Test',zorder=5)
    #axs[row,1].plot( xvertTVT, testTVT[xvertTVT] ,markersize=8 ,linestyle = '',marker='o', color='#0072B2', label='Test',zorder=5)

    if name_models[k] == 'RF':
      axs[row,0].text( 0.15 , 0.85, name_models[k] + '-TT', transform=axs[row,0].transAxes, fontsize= 10.5, ha="center",  verticalalignment='top', bbox=props_orange)
      axs[row,1].text( 0.15 , 0.85, name_models[k] + '-TVT' , transform=axs[row,1].transAxes, fontsize= 10.5, ha="center",  verticalalignment='top', bbox=props_orange)
    else:
      axs[row,0].text( 0.15 , 0.95, name_models[k] + '-TT', transform = axs[row,0].transAxes , fontsize= 10.5, ha="center",  verticalalignment='top', bbox=props_orange)
      axs[row,1].text( 0.15 , 0.95, name_models[k] + '-TVT', transform=axs[row,1].transAxes, fontsize= 10.5, ha="center",  verticalalignment='top', bbox=props_orange)
    axs[row,0].grid( axis='y', linestyle='--')
    axs[row,1].grid( axis='y', linestyle='--')

    del_featsTT  = [i['feat_del'] for i in dTT.values()][1:]
    del_featsTVT = [i['feat_del'] for i in dTVT.values()][1:]
    for j, feature in enumerate(del_featsTT):
      axs[row,0].text( j -0.05 , 0.72, feats_letters[feature] , fontsize= 9.5, ha="center",  verticalalignment='top', bbox = props_gray,zorder=10)
      axs[row,1].text( j -0.05 , 0.72, feats_letters[del_featsTVT[j]] , fontsize= 9.5, ha="center",  verticalalignment='top', bbox = props_gray,zorder=10)
    k = k + 1
  if k == 1:
    axs[row,1].legend( loc = 'upper right')

axs[row,0].set_ylabel(' Selection metric (AUC) [-]',fontsize=12.5, y = 0 )
plt.xlabel('Number of Predictors', x = 0, fontsize=12.5 )
plt.ylim([0.7,1])

yposition = 2.5
t = ""
k = 0
for letter,predictor in zip(feats_letters.values(),feats_letters.keys()):
  if k == 4:
    t = t + letter + '(' + predictor + ')\n'
  else:
    t = t  + letter + '(' + predictor + '), '
  k += 1

axs[row,1].text( -0.05, -0.45, t, transform=axs[row,1].transAxes, fontsize= 9.5, ha="center", verticalalignment="top", bbox=props_gray)
plt.savefig('figure'+ str(SEED) +'.pdf' ,dpi = 1500, bbox_inches='tight')


'''
