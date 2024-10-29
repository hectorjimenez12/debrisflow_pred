# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:12:05 2023

@author: hector J
"""

import HydroErr as he
import os
import numpy as np
import pandas as pd
import pdb
import geopandas as gpd
import copy
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import haversine as hs
from geopy import distance
import math

'''
# IMPORT CLASSES AND FUNCTIONS
'''
os.chdir("G:/Mi unidad/Investigacion/2024/Otoño/FONDEF_AMTC_Probability_Of_DebrisFlow/00_Generacion_Dataset")
#import Source.Functions as fgeo
#import Source.Class_timeserie as ts #import all functions that are required



#%%
''' Shapes of analysis'''  
creeks = gpd.read_file( os.getcwd()+ '\\Data\\Shapefiles\\DF_Shapefile.shp')
creeks.plot()
creeks = creeks.to_crs("epsg:4326")
points_centroid = creeks.centroid.values
ymax,ymin  = np.max([i.y for i in points_centroid]) , np.min([i.y for i in points_centroid])
xmax, xmin = np.max([i.x for i in points_centroid]) , np.min([i.x for i in points_centroid])




#%% Obtener datos de pr y t desde series de tiempo interpoladas en R


datap1   = pd.read_csv("Data/Predictores_DP_MP/p1IDW.csv")
datapt   = pd.read_csv("Data/Predictores_DP_MP/ptIDW.csv")
datageo2  = pd.read_csv("Data/Predictores_DP_MP/df.Basins_v2.csv")
datageo1  = pd.read_csv("Data/Predictores_DP_MP/df.Basins.csv")

datapt.columns = ['ID','may2013','mar2015','jul2015','ene2017','may2017','ene2020']
datap1.columns = ['ID','may2013','mar2015','jul2015','ene2017','may2017','ene2020']

datapt = datapt.drop(columns = ['ene2017','ene2020'])
datap1 = datap1.drop(columns = ['ene2017','ene2020'])


datap1.iloc[:,1:].boxplot()
datapt.iloc[:,1:].boxplot()

#smpolygons = [ pd.read_csv('Data/ERA5SM/SeriesSM_Marzo_2015.csv'),
#               pd.read_csv('Data/ERA5SM/SeriesSM_Enero_2017.csv'),
#               pd.read_csv('Data/ERA5SM/SeriesSM_Mayo_2017.csv'),
#               pd.read_csv('Data/ERA5SM/SeriesSM_Enero_2020.csv') ]
#SCdata = pd.read_csv('Data/Modis/SnowCover.csv')
#SCpolygons = [ SCdata.iloc[0:11,:],SCdata.iloc[11:31,:],
#               SCdata.iloc[31:52,:],SCdata.iloc[52:,:]   ]

#%% Generar inputs para modelos clasificadores

XYdata = {}
geo_features1 = ['MltnInd','SlopeMn','Tc_Clfr','Area_m2','RvrsSlD','MnRvrSD','ab2015','ab2017j','ab2017m', 'ab2020','ID']
geo_features2 = ['nStrahler1','Dd_km-1' , 'Ct_km','Tf_hr','L_toOutlet_km','V_m_s','LRiverNetwork_km']



datageo = pd.concat( [datageo1.loc[:,geo_features1], datageo2.loc[:,geo_features2] ], axis = 1)
#geo_features = geo_features1 + geo_features2
datageo['ab2013'] = 0
datageo['ab2015j'] = 0

geo_features = ['MltnInd','SlopeMn','Tc_Clfr','Area_m2','RvrsSlD','MnRvrSD',
                'nStrahler1','Dd_km-1' , 'Ct_km','Tf_hr','L_toOutlet_km','V_m_s','LRiverNetwork_km']

met_features = ['pr_max','pr_sum'] #prmax de   (linea de nieves modis)}
#sm_features  = ['prev_sm','max_sm']
y = []

list_creek = [pd.DataFrame( datageo.iloc[i]) for i in list(datageo.index)]
names_event = ['ab2013','ab2015','ab2015j','ab2017m'] #'ab2020' 'ab2017j' 
list_creek[0]


for i, dfcreek in enumerate( list_creek ):
    print(i)
    geodata = {ig: dfcreek.loc[ig,:].values[0] for ig in geo_features} 
        
    tf_nan  = [ ~ np.isnan(dfcreek.loc[j,:].values[0])  for j in names_event ]
    
    id_bas = dfcreek.loc['ID'].values[0]
    
    dataprsum = datapt[ datapt.iloc[:,0].isin([id_bas]) ]
    dataprmax = datap1[ datap1.iloc[:,0].isin([id_bas]) ]
    
    prmax_bas = { names_event[k] : mag_pr for k,mag_pr in enumerate( dataprmax.values[0][1:] ) }
    prsum_bas = { names_event[k] : mag_pr for k,mag_pr in enumerate( dataprsum.values[0][1:] ) }
    
    #smbas   = { names_event[k]:{'sm':data.loc[:,str(creeks.ID[i])].values } for k,data in enumerate(smpolygons) }
    #SCbas   = { names_event[k]:{'SC':data.loc[:,str(creeks.ID[i])].values } for k,data in enumerate(SCpolygons) }
    
    #sc_max  = [ np.nanmax(SCserie['SC']) for SCserie in SCbas.values()]
    #sc_max  = [ k if ~np.isnan(k)  else 0 for k in sc_max ]
    
    #sm_prev = [smserie['sm'][0] for smserie in smbas.values()]
    #sm_max = [ smserie['sm'].max() for smserie in smbas.values()]
    
    XYdata = dict(XYdata, **{ str(i)+'_'+ikey: dict({ 'storm':k ,
                                                     'pr_max': prmax_bas[ikey],                                                     
                                                     'pr_sum':prsum_bas[ikey]
                                                     #'sc_max':sc_max[k],
                                                     #'sm_prev':sm_prev[k],
                                                     #'sm_max':sm_max[k] 
                                                     } ,**geodata)  for k,ikey in enumerate(names_event) if tf_nan[k] } )
    y += [ dfcreek.loc[j,:].values[0] for k,j in enumerate(names_event) if  tf_nan[k] ]


X = pd.DataFrame.from_dict( data=XYdata, orient='index')
#X = X.drop(columns=['Melton','Dd','Smean','Sstream','Tc_Med'])
y = pd.DataFrame(y).astype({0:'int32'}).rename(columns =  {0:'y'})
np.unique(y,return_counts=True)
X = X.reset_index(level = 0, drop = True)

X.to_csv('Output/00_Predictor_Generation/PredictorX_notAllStorms.csv',index=False)
y.to_csv('Output/00_Predictor_Generation/Y_notAllStorms.csv',index=False)



#mesilla_ind = [i for i in X.index if '13_' in i][0:4]
#xmes = X.loc[ mesilla_ind ,:]
#xmes.to_csv('PredictoresMesilla.csv')
