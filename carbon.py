#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import math
import scipy as sp
import netCDF4
import gsw # this will be VERY useful https://github.com/TEOS-10/GSW-Python


# # 1. Download synthetic argo file (= S-file)

# In[29]:


# import requests
# import urllib 
# import ftplib
import wget
import os

def download_float(dac, wmo):
    """
    dac : data center, e.g. 'coriolis'
    wmo : world id of the float (unique), e.g. 6901866
    """
    
    # check if float folder exists
    if(os.path.isdir('./floats') == False):
        print("Directory 'floats' does not exist")
        print("It's being created in your current path")
        os.mkdir(os.path.join(os.getcwd(),'floats'))
    # check if file already exists in that folder
    if(os.path.isfile(os.getcwd()+'/floats/'+str(wmo)+'_Sprof.nc') == False):
        print("File does not exist")
        print("File is being downloaded")
        print("...")
        base_url = 'ftp://ftp.ifremer.fr/ifremer/argo/dac/'
        full_url = base_url+dac+'/'+str(wmo)+'/'
        link = full_url+str(wmo)+'_Sprof.nc'
        wget.download(link, out = os.getcwd()+'/floats/')
        print("Done !")
    else:
        print("File already exists, moving on.")


# In[30]:


download_float('coriolis',6900798)
download_float('coriolis',6901866)


# # 2. Extract data from NetCDF

# In[31]:


# Read data

def read_data(ncfile):
    
    # see also the doc for SYNTHETIC BGC-ARGO files: https://archimer.ifremer.fr/doc/00445/55637/75359.pdf
    nc = netCDF4.Dataset(os.getcwd()+'/floats/'+ncfile, mode='r') # add a get url download the file in a folder then read it
    # get coordinates variables
    lat = nc.variables['LATITUDE'][:].data
    lon = nc.variables['LONGITUDE'][:].data
    time = nc.variables['JULD'][:].data
    depth = nc.variables['PRES'][:].data
    depth_qc = nc.variables['PRES_QC'][:].data
    
    #other parameters
    bbp = nc.variables['BBP700'][:].data
    bbp_qc = nc.variables['BBP700_QC'][:].data
    chla = nc.variables['CHLA'][:].data
    chla_qc = nc.variables['CHLA_QC'][:].data
    temp = nc.variables['TEMP'][:].data
    temp_qc = nc.variables['TEMP_QC'][:].data
    psal = nc.variables['PSAL'][:].data
    psal_qc = nc.variables['PSAL_QC'][:].data
    
    # close netcdf
    nc.close()
    
    # put data into a Dataframe
    data = pd.DataFrame({'depth':np.concatenate(depth), 'depth_qc':np.concatenate(depth_qc), 'temp':np.concatenate(temp),
              'temp_qc':np.concatenate(temp_qc), 'psal':np.concatenate(psal), 'psal_qc':np.concatenate(psal_qc),
              'chla':np.concatenate(chla), 'chla_qc':np.concatenate(chla_qc), 'bbp':np.concatenate(bbp), 
              'bbp_qc':np.concatenate(bbp_qc)})
    
    # number of profiles in the file
    n_prof = len(time)
    
    # repeat metadata to fit data length
    points_per_profile = data.shape[0]/n_prof
    time = np.repeat(time, points_per_profile)
    lat = np.repeat(lat, points_per_profile)
    lon = np.repeat(lon, points_per_profile)
    data['time'] = time
    data['lat'] = lat
    data['lon'] = lon

    # add profile IDs in a similar way, assuming the hypothesis of constant length per profile is correct (thanks to S profiles?)
    ids = np.array(range(1,n_prof+1))
    data['id'] = np.repeat(ids, points_per_profile)
    
    # some additional cleaning
    FillValue = 99999.0
    data = data.replace(FillValue, 'NaN')
    
    return(data)


# In[48]:


data = read_data('6901866_Sprof.nc')
#data = read_data('6900798_Sprof.nc')


# In[49]:


data


# # 3. Some data cleaning
# - convert QC bytes into integers
# - remove depth where we don't have BBP data ==> this needs to be validated and check the order with the additional QCs to be applied

# In[50]:


# convert QC bytes into integers
def bytes_to_int(x):
    try:
        x = int(x)
    except:
        x = 'NaN'
    return(x)


# In[51]:


# apply that function where it is needed
data['temp_qc'] = data['temp_qc'].apply(bytes_to_int)
data['psal_qc'] = data['psal_qc'].apply(bytes_to_int)
data['depth_qc'] = data['depth_qc'].apply(bytes_to_int)
data['chla_qc'] = data['chla_qc'].apply(bytes_to_int)
data['bbp_qc'] = data['bbp_qc'].apply(bytes_to_int)


# In[52]:


# remove depth where we don't have BBP data
# NOTE : is that correct? If not, we may have troubles with the median filter. OR we can let NaN be present in
# BBP data but then write a median filter that does not take them into account? Which is probably the case? 
# To check with Giorgio
data = data[data.bbp != 'NaN']
data


# # 4. Compute density

# In[53]:


# compute density
# see https://teos-10.github.io/GSW-Python/
psal = gsw.SA_from_SP(np.array(data['psal']), np.array(data['depth']), np.array(data['lon']), np.array(data['lat']))
temp = gsw.CT_from_t(psal, np.array(data['temp']), np.array(data['depth']))
sigma = gsw.sigma0(psal, temp)
data['sigma'] = sigma


# # 5. Additional QCs (to be done)

# In[54]:


# to be done : apply some additionnal QC on BBP data
# QUESTION: before of after removing the NaN (if needed) => see below


# # 6. Some plots

# In[55]:


data.plot.scatter(x = 'bbp', y = 'depth')


# In[56]:


# # plot some data
# from plotnine import ggplot, aes, geom_point, geom_line
# tmp = data[data.id == 1][:10]
# tmp.reset_index(inplace=True)#, drop=True)
# (ggplot(data = tmp) + aes(x = 'temp', y = 'depth') + geom_point()) #+ scale_y_reverse()

# ==> for an unknown reason, plotnine does not behave like it used to..


# # 7. Remove dark offset  ==> remark maybe it's just ONE value per float and not per profile

# In[57]:


def remove_dark_offset(group):
    min_bbp = np.nanmin(group['bbp'])
    group['bbp'] = group['bbp'] - min_bbp
    return(group)


# In[58]:


data = data.groupby('id').apply(remove_dark_offset)


# # 8. Apply median filter on BBP data

# In[59]:


# median filter on BBP data ==> this needs to be done for EACH profile individually
from scipy import signal

# def medfilt (x, k):
#     """Apply a length-k median filter to a 1D array x.
#     Boundaries are extended by repeating endpoints.
#     """
#     assert k % 2 == 1, "Median filter length must be odd."
#     assert x.ndim == 1, "Input must be one-dimensional."
#     k2 = (k - 1) // 2
#     y = np.zeros ((len (x), k), dtype=x.dtype)
#     y[:,k2] = x
#     for i in range (k2):
#         j = k2 - i
#         y[j:,i] = x[:-j]
#         y[:j,i] = x[0]
#         y[:-j,-(i+1)] = x[j:]
#         y[-j:,-(i+1)] = x[-1]
#     return np.median (y, axis=1)

# def medianfilter(group):
#     smoothed = signal.medfilt(group['bbp'],5) # kernel size = 5
#     return(pd.Series(smoothed)) # apply on dataframe MUST return a dataframe, a series or a scaler, not a numpy array

# # not a pretty code but it works ..
# # grouped['bbp'].apply(lambda x: signal.medfilt(x, kernel_size = 5))
# tmp = data.groupby('id').apply(medianfilter)
# tmp = tmp.reset_index()
# data['bbp'] = np.array(tmp[0])

def medianfilter(group):
    smoothed = signal.medfilt(group['bbp'],5) # kernel size = 5
    group['bbp'] = smoothed
    return(group)

tmp = data.groupby('id').apply(medianfilter)


# # 9. Computation of the Mixed Layer Depth (MLD)
# - sigma criteria : 0.03 kg/m³ instead of 0.1
# - depth ref = 10 m
# - See Kara et al., 2003:  https://doi.org/10.1029/2000JC000736)

# In[60]:


sigma_criteria = 0.03
depth_ref = 10
from scipy.interpolate import interp1d

def compute_MLD(group): # this will probably need to be adapted after QC application (for NaN for instance, or if sigma data between the surface and 10m are bad)
    # approx sigma et 10m
    f = interp1d(group['depth'], group['sigma']) # linear interp
    # we define sigma_surface as sigma at 10m
    sigma_surface = float(f(depth_ref))
    MLD = np.min(group[group['sigma'] >= sigma_surface + sigma_criteria]['depth'])
    group['MLD'] = np.repeat(MLD, group.shape[0])
    return(group)

data = data.groupby('id').apply(compute_MLD)


# # 10. Convert BBP to POC

# In[45]:


# some criteria (NB: empirical factors)
withinMLD = 37.530
belowMLD = 31.620

def BBP_to_POC(group):
    MLD = np.max(group['MLD'])
    # POC from BBP
    # within MLD
    tmp1 = group[group.depth <= MLD]['bbp']*withinMLD
    # below MLD
    tmp2 = group[group.depth > MLD]['bbp']*belowMLD
    # concat data
    group['poc'] = pd.concat([tmp1, tmp2])
    return(group)


# In[46]:


data = data.groupby('id').apply(BBP_to_POC)


# In[47]:


data


# # TO DO
# - compute the depth of the bottom of the euphotic zone (3 différents ways possible but two to really compare)
# - then go to the integration -> use scipy for this
# - check the right order of every section
# - QC the data (with the QC of 2012 first (see Dall'Olmo) but then check w/ Raphaelle Sauzede
# - convert julian day and human time

# In[ ]:


tmp

