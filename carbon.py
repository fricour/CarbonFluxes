#!/usr/bin/env python
# coding: utf-8

# In[382]:


import numpy as np
import pandas as pd
import math
import scipy as sp
import netCDF4
import gsw # this will be VERY useful https://github.com/TEOS-10/GSW-Python


# # 1. Download synthetic argo file (= S-file)

# In[383]:


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


# In[384]:


download_float('coriolis',6900798)
#download_float('coriolis',6901866) # Black Sea, just for testing
download_float('coriolis',6900799)


# # 2. Extract data from NetCDF

# In[385]:


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


# In[386]:


#data = read_data('6901866_Sprof.nc')
data = read_data('6900798_Sprof.nc') # here you choose which float you'll analyze.
#data = read_data('6900799_Sprof.nc')


# In[387]:


data


# # 3. Some data cleaning
# - convert QC bytes into integers
# - remove depth where we don't have BBP data ==> this needs to be validated and check the order with the additional QCs to be applied

# In[388]:


# convert QC bytes into integers
def bytes_to_int(x):
    try:
        x = int(x)
    except:
        x = 'NaN'
    return(x)


# In[389]:


# apply that function where it is needed
data['temp_qc'] = data['temp_qc'].apply(bytes_to_int)
data['psal_qc'] = data['psal_qc'].apply(bytes_to_int)
data['depth_qc'] = data['depth_qc'].apply(bytes_to_int)
data['chla_qc'] = data['chla_qc'].apply(bytes_to_int)
data['bbp_qc'] = data['bbp_qc'].apply(bytes_to_int)


# In[390]:


# remove negative NaN depth and negative depths
data['depth_qc'].replace(3, 'NaN', inplace=True)
data['depth_qc'].replace(4, 'NaN', inplace=True)
data = data[data.depth != 'NaN']
data = data[data.depth >= 0]


# In[391]:


data


# # 4. Quality controls (to be done) => couldn't find any tests in Dall'Olmo 2012

# In[392]:


# to be done : apply some additionnal QC on BBP data
# QUESTION: before of after removing the NaN (if needed) => see below

# QC from https://archimer.ifremer.fr/doc/00491/60262/63668.pdf

# let's apply those from ARGO
# replace QC 3 & 4 with NaN
data['temp_qc'].replace(3, 'NaN', inplace=True)
data['temp_qc'].replace(4, 'NaN', inplace=True)
data['psal_qc'].replace(3, 'NaN', inplace=True)
data['psal_qc'].replace(4, 'NaN', inplace=True)
data['bbp_qc'].replace(3, 'NaN', inplace=True)
data['bbp_qc'].replace(4, 'NaN', inplace=True)
data['chla_qc'].replace(3, 'NaN', inplace=True)
data['chla_qc'].replace(4, 'NaN', inplace=True)


# In[393]:


data = data[data.temp_qc != 'NaN']
data = data[data.psal_qc != 'NaN']
#data = data[data.bbp_qc != 'NaN'] # will be applied AFTER the computation of the MLD
# I am not applying the QC on chla on purpose because I know that Chla QC is very tricky .. we could lose entire profiles
data


# # 5. Compute density

# In[394]:


# compute density
# see https://teos-10.github.io/GSW-Python/
psal = gsw.SA_from_SP(np.array(data['psal']), np.array(data['depth']), np.array(data['lon']), np.array(data['lat']))
temp = gsw.CT_from_t(psal, np.array(data['temp']), np.array(data['depth']))
sigma = gsw.sigma0(psal, temp)
data['sigma'] = sigma


# # 6. Computation of the Mixed Layer Depth (MLD)
# - sigma criteria : 0.03 kg/m³ instead of 0.1
# - depth ref = 10 m
# - See Kara et al., 2003:  https://doi.org/10.1029/2000JC000736)
# 
# 
# - /!\ some issues are still present with MLD computation .. check ID = 20 for e.g. => better check for a diff of 0.03 in both directions .. looks like they are density inversions at some point.. => well maybe not actually: see http://bora.uib.no/bitstream/handle/1956/1722/paper_mld_BORA.pdf?sequence=1&isAllowed=y and https://www.researchgate.net/figure/Month-of-maximum-MLD-reached-in-the-North-Atlantic-for-a-the-DT-02C-climatology_fig3_229010378

# In[395]:


# mmmh need to adress two issues, the presence of NaN and when we don't have enough data before 10?

sigma_criteria = 0.03
depth_ref = 10 # need to be adapted ?? otherwise there could be issues with bad PSAL and TEMP data
from scipy.interpolate import interp1d

def compute_MLD(group): # this will probably need to be adapted after QC application (for NaN for instance, or if sigma data between the surface and 10m are bad)
    
    # subset group to avoid NaN issue (i.e. TypeError: '<' not supported between instances of 'str' and 'float')
    tmp = group[['depth', 'sigma']]
    tmp = tmp[tmp.sigma != 'NaN']
    #tmp = tmp[tmp.depth != 'NaN']
    
    # check if first depth is below depth_ref
    if(np.min(tmp['depth'] >= depth_ref)):
        group['MLD'] = np.repeat('NaN', group.shape[0])
    else:
        # approx sigma et 10m
        f = interp1d(tmp['depth'], tmp['sigma']) # linear interp
        # we define sigma_surface as sigma at 10m
        sigma_surface = float(f(depth_ref))
        MLD = np.min(tmp[tmp['sigma'] >= sigma_surface + sigma_criteria]['depth'])
        group['MLD'] = np.repeat(MLD, group.shape[0])
    return(group)

data = data.groupby('id').apply(compute_MLD)

# only keep data for which we have a MLD
data = data[data.MLD != 'NaN']


# In[396]:


np.unique(data['MLD'])


# In[397]:


# check MLD range
print(np.min(np.unique(data['MLD'])))
print(np.max(np.unique(data['MLD'])))


# # Now that the MLD has been computed, we can remove data where bbp is NaN (core-argo vs b-argo)

# In[398]:


# remove depth where we don't have BBP data
# NOTE : is that correct? If not, we may have troubles with the median filter. OR we can let NaN be present in
# BBP data but then write a median filter that does not take them into account? Which is probably the case? 
# To check with Giorgio
data = data[data.bbp != 'NaN']
data


# # 6. Some plots

# In[399]:


data.plot.scatter(x = 'bbp', y = 'depth')


# In[400]:


# # plot some data
# from plotnine import ggplot, aes, geom_point, geom_line
# tmp = data[data.id == 1][:10]
# tmp.reset_index(inplace=True)#, drop=True)
# (ggplot(data = tmp) + aes(x = 'temp', y = 'depth') + geom_point()) #+ scale_y_reverse()

# ==> for an unknown reason, plotnine does not behave like it used to..


# # 7. Remove dark offset  ==> remark maybe it's just ONE value per float and not per profile

# In[401]:


def remove_dark_offset(group):
    min_bbp = np.nanmin(group['bbp'])
    group['bbp'] = group['bbp'] - min_bbp
    return(group)


# In[402]:


data = data.groupby('id').apply(remove_dark_offset)


# # 8. Apply median filter on BBP data

# In[403]:


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


# # 9. Convert BBP to POC

# In[404]:


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


# In[405]:


data = data.groupby('id').apply(BBP_to_POC)


# In[406]:


data


# # 10. Convert julian time to human time

# In[407]:


import datetime
def julian_to_human(group):
    gregorian_time = datetime.date(1950, 1, 1) + datetime.timedelta(days=np.min(group['time']))
    #group['human_time'] = np.repeat(gregorian_time.strftime('%d-%m-%Y'), group.shape[0])
    group['human_time'] = np.repeat(gregorian_time, group.shape[0])
    return(group)

data = data.groupby('id').apply(julian_to_human)


# # 11. Bottom of the euphotic zone
# 3 (4) methods:
# - With PAR data but let's forget that, it's not on all BGC float
# - Chla empirical (I like this one, it's so easy to use)
# - Based on Chla relative values (shape of the profile and not absolute values)
# - script from Lionel about a paper from Morel 2001

# ### A) Based on Chla relative values

# In[408]:


# then in that case, we must also apply a median filter on chla data
def medianfilter_chla(group):
    smoothed = signal.medfilt(group['chla'],5) # kernel size = 5
    group['chla'] = smoothed
    return(group)

data = data.groupby('id').apply(medianfilter_chla)


# In[409]:


# Let's start with the shape of the Chla profile
# plot some data
tmp = data[data.id == 20] # choose a profil ID here 
tmp = tmp[tmp.depth < 250]

#tmp.plot.scatter(x = 'chla', y = 'depth')
from matplotlib import pyplot as plt
plt.gca().invert_yaxis()
plt.plot(tmp['chla'], tmp['depth'])


# In[410]:


# based on some profiles, I decided to put the threshold of Chla at 0.1 mg/m³ (pay attention to the units in Morel et al., 2001)
# compare the value of the depth found here with the one of Morel, should be close. Let's also compute the empirical equation to see.
chla_threshold = 0.1

## NOTE : not sure it's the good way to go


# In[411]:


z_eu_chla_threshold = min(tmp[tmp.chla < chla_threshold].depth)
z_eu_chla_threshold


# ### B) Based on empirical relationship

# In[412]:


chla_10m = float(tmp.iloc[(tmp['depth']-10).abs().argsort()[:1]]['chla']) # chla at ~ 10 m (or the closest)
z_eu_empir = 1/(0.012939+(0.018151*chla_10m))
z_eu_empir


# ### C) Morel's method (to be done)

# # Which method is the best to compute a z_eu ? Needs to be discussed

# In[413]:


# compute the z_eu for each profile, I'm choosing the satellite method (empirical)
def compute_z_eu(group):
    chla_10m = float(group.iloc[(group['depth']-10).abs().argsort()[:1]]['chla']) # chla at ~ 10 m (or the closest)
    z_eu_empir = 1/(0.012939+(0.018151*chla_10m))
    group['z_eu'] = z_eu_empir
    return(group)


# In[414]:


data = data.groupby('id').apply(compute_z_eu)


# In[415]:


data


# ## Thickness of the layer where particles can potentially be produced by photosynthesis

# In[416]:


# I am gonna keep the empirical z_eu
def compute_zp(group):
    z_p = max(min(group['MLD']), min(group['z_eu']))
    group['z_p'] = z_p
    return(group)


# In[417]:


data = data.groupby('id').apply(compute_zp)


# In[418]:


# np.unique(data[data.MLD > 200]['human_time'])
# np.unique(data[data.MLD > 200]['MLD'])


# # Check that flux is ~ 0 at 1000m (see hypotheses)
# - For those two floats, it is the case but this step needs to be reminded when going global

# # Integration of POC in different layers

# In[419]:


tmp = data[data.id == 20]
tmp


# In[420]:


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simps.html
# https://en.wikipedia.org/wiki/Simpson%27s_rule

# integration over the whole profile (just a check)
x = np.array(tmp['depth'])
y = np.array(tmp['poc'])
from scipy import integrate
integrate.simps(y,x)


# In[421]:


def integrate_poc(depth_min, depth_max, data):
    data = data[(data.depth > depth_min) & (data.depth < depth_max)]
    x = np.array(data.depth)
    y = np.array(data.poc)
    return(integrate.simps(y,x))


# In[422]:


def compute_carbon_stocks(group):
    
    # 0-1000m => check in the paper iPOC_0_1000 
    iPOC_0_1000 = integrate_poc(0, 1000, group)
    group['iPOC_0_1000'] = iPOC_0_1000
    
    # 0 -> zp : 'productive layer' or iPOC_0_zp
    zp = min(group['z_p'])
    iPOC_0_zp = integrate_poc(0, zp, group)
    group['iPOC_0_zp'] = iPOC_0_zp
    
    # zp -> 1000m : 'mesopelagic layer' or iPOC_zp_1000
    iPOC_zp_1000 = integrate_poc(zp, 1000, group)
    group['iPOC_zp_1000'] = iPOC_zp_1000
    
    # other stocks in the meso layer such as iPOC_zp+zi_1000
    # zi = 50, 100, 200, 300, 400
    
    zi = 50 
    try: # needed to be added when MLD is so deep that we are close to z_max (i.e. 1000m) hence flux integration can be buggy
        iPOC_zpzi50_1000 = integrate_poc(zp+zi, 1000, group)
        group['iPOC_zp+zi50_1000'] = iPOC_zpzi50_1000
    except:
        group['iPOC_zp+zi50_1000'] = 'NaN'
    
    zi = 100
    try:
        iPOC_zpzi100_1000 = integrate_poc(zp+zi, 1000, group)
        group['iPOC_zp+zi100_1000'] = iPOC_zpzi100_1000
    except:
        group['iPOC_zp+zi100_1000'] = 'NaN'
    
    zi = 200
    try:
        iPOC_zpzi200_1000 = integrate_poc(zp+zi, 1000, group)
        group['iPOC_zp+zi200_1000'] = iPOC_zpzi200_1000
    except:
        group['iPOC_zp+zi200_1000'] = 'NaN'
    
    zi = 300
    try: 
        iPOC_zpzi300_1000 = integrate_poc(zp+zi, 1000, group)
        group['iPOC_zp+zi300_1000'] = iPOC_zpzi300_1000
    except:
        group['iPOC_zp+zi300_1000'] = 'NaN'
    
    zi = 400
    try:
        iPOC_zpzi400_1000 = integrate_poc(zp+zi, 1000, group)
        group['iPOC_zp+zi400_1000'] = iPOC_zpzi400_1000
    except:
        group['iPOC_zp+zi400_1000'] = 'NaN'
    
    return(group)


# In[423]:


#import ipdb; ipdb.set_trace() # debugging starts here


# In[424]:


data = data.groupby('id').apply(compute_carbon_stocks)
data


# In[425]:


# check stock max 
max(data['iPOC_0_1000'])


# In[426]:


# let's just plot full stock (0-1000) over time to see if the trends are consistent with the paper
test = data[['human_time', 'iPOC_0_1000']].drop_duplicates()
test


# In[427]:


# attention, total stock has been SMOOTHED so the pic was surely erased in Fig. 3
test.plot()


# # Let's (try to) replicate Fig. 2

# In[428]:


from pylab import *
from matplotlib import pyplot as plt
from matplotlib import dates

from datetime import datetime, timedelta
from datetime import date

import cmocean
import matplotlib.dates as mdates


# In[429]:


# data to plot
x = np.array(data['human_time'])
y = np.array(data['depth'])
z = np.array(data['bbp'])

# compute log10 BBP
# import math
# z = math.log10(z)
# see https://stackoverflow.com/questions/47208473/attributeerror-numpy-float64-object-has-no-attribute-log10/47208873#47208873
z = np.log10(z.astype(np.float64))
# np.log10(list(z)) # also work but np.log10(z) does NOT work


# In[430]:


# cmap = cmocean.cm.tempo_r
# cmap = mpl.cm.RdYlBu
# cmap = mpl.cm.RdYlBu_r
# cmap = mpl.cm.jet
# cmap = mpl.cm.magma
cmap = mpl.cm.hot


vmin = np.percentile(z, 15) # "min color range"
vmax = np.percentile(z, 95) 
ymax = 1000 # depth max to plot
title = '$\log_{10}$(bbp$_{700}$) [m$^{-1}$]'

# see matplotlib tuto at https://matplotlib.org/3.3.2/tutorials/introductory/usage.html

fig, ax = plt.subplots(figsize=(20, 4)) # Create a figure containing a single axes.
ax.set_facecolor('darkgrey')
# fig = plt.figure(1, figsize=(20, 6))
# ax = fig.add_subplot(111, axisbg='#E6E6E6')
ax.scatter(x, y, s=64, c=z, marker='o', edgecolors='none', vmin=vmin, vmax=vmax, cmap=cmap) # Plot data on the axes.
ax.set_ylabel('depth [m]', fontsize = 20)
#ax.set_title(title, fontsize = 20)
ylim([0, ymax])
plt.gca().invert_yaxis()

#####   set ticks and  labels
months = mdates.MonthLocator(interval=1)
years = mdates.YearLocator() 

datemin = datetime.strptime('2010-12-01', '%Y-%m-%d').date()
datemin = datemin.toordinal()
datemax = datetime.strptime('2012-10-01', '%Y-%m-%d').date()
datemax = datemax.toordinal()

ax.set_xlim(datemin, datemax)

ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_locator(years)
fmt = mdates.DateFormatter('%m')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_formatter(fmt)

ax.tick_params(axis='x', which='major', direction='out', pad=25, labelsize=20)
ax.tick_params(axis='x', which='minor', direction='out', pad=0.5, labelsize=20)
ax.tick_params(axis='both', which='both', labelsize=20)

# add mld data
mld = np.array(data[['MLD', 'human_time']].drop_duplicates().MLD)
time_mld = np.array(data[['MLD', 'human_time']].drop_duplicates().human_time)
ax.plot(time_mld, mld, 'w-', lw = 2)

# add colorbar
ax1 = fig.add_axes([0.15, 1, .7, .1]) # left (+ for left), up (+ for up), length, thickness

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                       norm=norm,
                       orientation='horizontal')

cb1.ax.set_title(title, fontsize = 20)
cb1.ax.tick_params(labelsize=20) 

imaxes = gca()
axes(cb1.ax)
axes(imaxes)


# # What about Fig. 3?

# In[431]:


data.columns


# In[432]:


data2 = data[['human_time', 'iPOC_0_1000', 'iPOC_0_zp','iPOC_zp_1000','iPOC_zp+zi50_1000',
             'iPOC_zp+zi100_1000', 'iPOC_zp+zi200_1000', 'iPOC_zp+zi300_1000', 'iPOC_zp+zi400_1000',
             'MLD']].drop_duplicates()
data2


# In[433]:


# remove NaN otherwise matplotlib will not work. It should only concerns stocks that are NaN which is the same as 0
data2 = data2.replace('NaN', 0)

# data to plot
x = np.array(data2['human_time'])
iPOC_0_1000 = np.array(data2['iPOC_0_1000'])
iPOC_0_zp = np.array(data2['iPOC_0_zp'])
iPOC_zp_1000 = np.array(data2['iPOC_zp_1000'])
iPOC_zpzi50_1000 = np.array(data2['iPOC_zp+zi50_1000'])
iPOC_zpzi100_1000 = np.array(data2['iPOC_zp+zi100_1000'])
iPOC_zpzi200_1000 = np.array(data2['iPOC_zp+zi200_1000'])
iPOC_zpzi300_1000 = np.array(data2['iPOC_zp+zi300_1000'])
iPOC_zpzi400_1000 = np.array(data2['iPOC_zp+zi400_1000'])

# add MLD data
mld = np.array(data2['MLD'])
time_mld = np.array(data2['human_time'])


# In[434]:


# sns.palplot(sns.light_palette("grey", reverse=True))


# In[435]:


import seaborn as sns
pal = sns.light_palette("grey", reverse=True)
pal = sns.dark_palette("green", reverse=True)


# In[436]:


fig, ax = plt.subplots(figsize = (20,6))
ax2 = ax.twinx()
w = 5

#ax.plot(x, iPOC_0_1000) # Plot data on the axes.
# ax.bar(x, iPOC_zp_1000, width = w, color = pal[0])
# ax.bar(x, iPOC_zpzi50_1000, width = w, color = pal[1])
# ax.bar(x, iPOC_zpzi100_1000, width = w, color = pal[2])
# ax.bar(x, iPOC_zpzi200_1000, width = w, color = pal[3])
# ax.bar(x, iPOC_zpzi300_1000, width = w, color = pal[4])
# ax.bar(x, iPOC_zpzi400_1000, width = w, color = pal[5])

ax.bar(x, iPOC_zp_1000, width = w, color = pal[5])
ax.bar(x, iPOC_zpzi50_1000, width = w, color = pal[4])
ax.bar(x, iPOC_zpzi100_1000, width = w, color = pal[3])
ax.bar(x, iPOC_zpzi200_1000, width = w, color = pal[2])
ax.bar(x, iPOC_zpzi300_1000, width = w, color = pal[1])
ax.bar(x, iPOC_zpzi400_1000, width = w, color = pal[0])

# iPOC in the upper layer
ax.plot(x, iPOC_0_zp, color = 'red')

ax.set_ylabel('iPOC [g POC m$^{-2}$]', fontsize = 20)

# add (not smoothed) iPOC_0_1000
# Shade the area between y1 and line y=0
ax.fill_between(x, y1 = iPOC_0_1000, facecolor="yellow", alpha=.5) # The outline color

# import seaborn as sns
# seq_col_brew = sns.color_palette("gray", 5)
# sns.set_palette(seq_col_brew)

# ax.set_title(title, fontsize = 20)
# ylim([0, ymax])
# plt.gca().invert_yaxis()

#####   set ticks and  labels
months = mdates.MonthLocator(interval=1)
years = mdates.YearLocator() 

datemin = datetime.strptime('2010-12-01', '%Y-%m-%d').date()
datemin = datemin.toordinal()
datemax = datetime.strptime('2012-10-01', '%Y-%m-%d').date()
datemax = datemax.toordinal()

ax.set_xlim(datemin, datemax)

ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_locator(years)
fmt = mdates.DateFormatter('%m')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_formatter(fmt)

ax.tick_params(axis='x', which='major', direction='out', pad=25, labelsize=20)
ax.tick_params(axis='x', which='minor', direction='out', pad=0.5, labelsize=20)
ax.tick_params(axis='both', which='both', labelsize=20)

# add mld data (on another axes)
ax2.plot(time_mld, mld, 'b--', lw = 2)
ax2.set_ylabel('MLD [m]', fontsize = 20)
plt.gca().invert_yaxis()


# In[437]:


data2.head(5)


# # Compute instantaneous fluxes (reminder : hyp that flux is 0 at 1000 m)

# In[438]:


def compute_flux(dataset, var_name):
    fluxes = []
    
    for i in range(dataset.shape[0]):
        if(i == 0): 
            fluxes.append(0)
        else:
            deltaIPOC = dataset.iloc[i].loc[var_name] - dataset.iloc[i-1].loc[var_name]
            deltaT = relativedelta(dataset.iloc[i].loc['human_time'], dataset.iloc[i-1].loc['human_time'])
            deltaT = deltaT.days
            flux = (deltaIPOC/deltaT)*100 # *100 to get mg instead of g
            fluxes.append(flux)
    return(fluxes)


# In[439]:


data2['flux_zpzi50_1000'] = compute_flux(data2, 'iPOC_zp+zi50_1000')
data2['flux_zpzi100_1000'] = compute_flux(data2, 'iPOC_zp+zi100_1000')
data2['flux_zpzi200_1000'] = compute_flux(data2, 'iPOC_zp+zi200_1000')
data2['flux_zpzi300_1000'] = compute_flux(data2, 'iPOC_zp+zi300_1000')
data2['flux_zpzi400_1000'] = compute_flux(data2, 'iPOC_zp+zi400_1000')
data2['flux_0_zp'] = compute_flux(data2, 'iPOC_0_zp')


# In[440]:


data2.head()


# In[441]:


# data to plot
x = np.array(data2['human_time'])
flux_zpzi50_1000 = np.array(data2['flux_zpzi50_1000'])
flux_zpzi100_1000 = np.array(data2['flux_zpzi100_1000'])
flux_zpzi200_1000 = np.array(data2['flux_zpzi200_1000'])
flux_zpzi300_1000 = np.array(data2['flux_zpzi300_1000'])
flux_zpzi400_1000 = np.array(data2['flux_zpzi400_1000'])
flux_0_zp = np.array(data2['flux_0_zp'])


# In[442]:


fig, ax = plt.subplots(figsize = (20,6))
ax2 = ax.twinx()

ax.plot(x, flux_zpzi50_1000, color = pal[4])
ax.plot(x, flux_zpzi100_1000, color = pal[3])
ax.plot(x, flux_zpzi200_1000, color = pal[2])
ax.plot(x, flux_zpzi300_1000, color = pal[1])
ax.plot(x, flux_zpzi400_1000, color = pal[0])
ax.plot(x, flux_0_zp, color = 'red')
ax.set_ylabel('E$_{zi}$ [mg POC m$^{-2}$]', fontsize = 20)

#####   set ticks and  labels
months = mdates.MonthLocator(interval=1)
years = mdates.YearLocator() 

datemin = datetime.strptime('2010-12-01', '%Y-%m-%d').date()
datemin = datemin.toordinal()
datemax = datetime.strptime('2012-10-01', '%Y-%m-%d').date()
datemax = datemax.toordinal()

ax.set_xlim(datemin, datemax)

ax.xaxis.set_minor_locator(months)
ax.xaxis.set_major_locator(years)
fmt = mdates.DateFormatter('%m')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_formatter(fmt)

ax.tick_params(axis='x', which='major', direction='out', pad=25, labelsize=20)
ax.tick_params(axis='x', which='minor', direction='out', pad=0.5, labelsize=20)
ax.tick_params(axis='both', which='both', labelsize=20)

# add mld data (on another axes)
ax2.plot(time_mld, mld, 'b--', lw = 2)
ax2.set_ylabel('MLD [m]', fontsize = 20)
plt.gca().invert_yaxis()


# # TO DO
# (- 1. QC the data (with the QC of 2012 first (see Dall'Olmo) but then check w/ Raphaelle Sauzede (couldn't find test in Dall'Olmo 2012))
# - 1. compute the depth of the bottom of the euphotic zone (3 différents ways possible but two to really compare) => needs to be discussed and tested with the method of Lionel for the determination of z_eu
# - 2. finalize the check for no POC at 1000m

# In[ ]:




