import os
import numpy as np
import pandas as pd
import pdb
#import geopandas as gpd
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




def get_model_and_results_byIndex(modelList, index, model = "MultiLayerPerceptron"):
  jobdata =  modelList[index] #rootfolderBE + fileseedsBE[1]
  mlptt   = jobdata[model][0]
  classModel   = mlptt[0]
  resultsModel = mlptt[1]
  return classModel, resultsModel



def Scaling(X,y,seed, features):
  X_train80, X_test, y_train80, y_test = train_test_split(X, y ,test_size = 0.2,
                                                          random_state=seed,stratify=y)
  X_train,   X_val , y_train,   y_val  = train_test_split(X_train80, y_train80,
                                                          test_size = 20/80 ,random_state=seed,
                                                          stratify = y_train80)
  Xmap = {'train':X_train80,'val':X_train80,'test':X_test,'features':features}
  Ymap = {'train': y_train80, 'val': y_train80,  'test':y_test }

  scaler = MinMaxScaler()
  #print(scaler)
  Xmap['train'] = pd.DataFrame(scaler.fit_transform(Xmap['train']) , columns = Xmap['train'].columns, index = Xmap['train'].index)
  Xmap['val']   = pd.DataFrame(scaler.transform(Xmap['val'])       , columns = Xmap['val'].columns  , index = Xmap['val'].index)
  Xmap['test']  = pd.DataFrame(scaler.transform(Xmap['test'])      , columns = Xmap['test'].columns , index = Xmap['test'].index)

  return Xmap, Ymap, scaler




def Scaling_NewData(X,scaler,features):
  Xmap = {'train':X,'val':X,'test':X,'features':features}
  
  Xmap['test']  = pd.DataFrame(scaler.transform(Xmap['test'])      ,
                               columns = Xmap['test'].columns ,
                               index = Xmap['test'].index)

  return Xmap['test']




def getplotresults(d):
  test     = [i['train_metric'][2] for i in d.values()]
  training = [i['train_metric'][0] for i in d.values()]
  val      = [i['train_metric'][1] for i in d.values()]

  return [training, val, test]




def get_select_model(dict_model, ind_sel, treshold):
  tvtmetrics      = getplotresults(dict_model)
  metrics_select  = np.array(tvtmetrics[ind_sel])
  id_max          = np.where(metrics_select == max(metrics_select))[0][-1]
  max_metric      = metrics_select[id_max]
  id_pass_filt    = np.where( max_metric - metrics_select <= treshold )[0]
  id_select       = [i for i in id_pass_filt if i >= id_max][-1]
  features_select = dict_model[id_select]['features']
  return tvtmetrics[2][id_select], features_select, id_select





def Prediction(model, features_model, XvaluesDF):
  #Npredictions = XvaluesDF.shape[0]
  Xvalues = XvaluesDF.loc[:,features_model]
  Xvalues = np.array(Xvalues)
  predictions = model.predict_proba(Xvalues)[:,1] #probability of ocurrence of debris flow
  return predictions
