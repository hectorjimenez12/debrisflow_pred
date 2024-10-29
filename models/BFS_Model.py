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
      
          


