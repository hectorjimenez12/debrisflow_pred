#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import pdb
#import geopandas as gpd
import copy
import matplotlib.pyplot as plt
import joblib


os.chdir("G:/Mi unidad/Investigacion/2024/Otoño/FONDEF_AMTC_Probability_Of_DebrisFlow/00_Generacion_Dataset/")
X = pd.read_csv('Output/00_Predictor_Generation/PredictorX_notAllStorms.csv')

storm_ids = X['storm']
X = X.drop(labels = ['storm'], axis = 1 )

X.columns
X = X.rename(columns = {'pr_max':'max_precip','pr_sum':'total_precip',
                        #'sc_max':'snow_cover_max','sm_prev':'soil_moisture_prev',
                        #'sm_max':'max_soil_moisture',
                        'MltnInd':'melton_index','Tf_hr':'tf_hr',
                        'Dd_km-1':'drainage_density','SlopeMn':'mean_slope',
                        'Tc_Clfr':'california_tc', 'Area_m2':'area',
                        'nStrahler1':'nstra','LRiverNetwork_km':'long_river',
                        'L_toOutlet_km':'long_outlet','V_m_s':'vel_ms',
                        'RvrsSlD':'river_slope','MnRvrSD':'main_river_slope'})

#X = X.drop(labels = ['snow_cover_max','max_soil_moisture'], axis = 1 )
X['area'] = X['area']/1000000

X['max_precip'] = X['max_precip'].values*np.sqrt(1/X['california_tc'].values)

#plt.scatter(range(len(X['max_precip'])) ,X['max_precip'].values)
#plt.scatter(range(len(X['max_precip'])) ,X['max_precip'].values*np.sqrt(1/X['california_tc'].values))


features = list(X.columns)
print(features)

def delete_cor_features(features,datadf,treshold):
    Mcor = datadf.corr()
    for i in range(Mcor.shape[0]):
        Mcor.iloc[i,i] = 0
    #print(Mcor)
    if np.max(np.array(Mcor)) > treshold:
        feat_max, ncors, max_cor = '', 0, 0
        for i, feat in enumerate(features):
          cors_values = Mcor[feat].values
          Ncor_over   = len( cors_values[cors_values > treshold] )
          if Ncor_over > ncors:
              feat_max, ncors, max_cor = feat, Ncor_over, np.max(cors_values)
              #print(feat_max, ncors, max_cor)
          elif Ncor_over == ncors:
              if max_cor < np.max(cors_values):
                  feat_max, ncors, max_cor = feat, Ncor_over, np.max(cors_values)
    
        datadfnew = datadf.drop(labels = [feat_max], axis = 1 )
        return delete_cor_features(features = list(datadfnew.columns), datadf = datadfnew, treshold= treshold)
    else:
        return datadf

Xpost = delete_cor_features(features = features, datadf = X, treshold = 0.8)
featuresPos = list(Xpost.columns)
print(featuresPos)

X, features = Xpost, featuresPos
X.to_csv('Output/01_Predictor_Filtered/X90_notAllStorms.csv',index=False)
