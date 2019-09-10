#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:24:20 2019

@author: mgermano
"""

#    import our libraires
import numpy as np
import pwlf
import external_functions
import time
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing 

start_time = time.time()

def checkdate(i_time,k_lat,l_lon):

    data = external_functions.load_era5(k_lat,l_lon)
    _,z,thv = external_functions.calculategeoh(data['sgeo'][i_time],
                              data['lnsp'][i_time],
                              data['T'][i_time,:],
                              data['q'][i_time,:],
                              data['level'])
    time_display =  data['time'][i_time]  
    
    #Post processing
    blh = data['blh'][i_time]
    T2  = data['T2'][i_time]
    u   = data['u'][i_time,:]
    v   = data['v'][i_time,:]
    th  = thv/(1.+0.609133*data['q'][i_time,:])
    T   = data['T'][i_time,:]
    zs=z[::-1]
    ths=thv[::-1]
    zCI = zs[zs<5000]
    temp = ths[zs<5000]
    
    
    return zCI,temp,time_display


def piecewise_regression(time,k_lat,l_lon,n_lines):
    
    time_ERA =time
    #2 and 4
    k_lat=k_lat
    l_lon= l_lon
    
    data = checkdate(time_ERA,k_lat,l_lon)
        
    x=data[0] #ERA 5 model layer heights zlevels
    y=data[1] #pottemp
    
    #   initialize piecwise linear fit with your x and y data
    myPWLF = pwlf.PiecewiseLinFit(x,y)
    
    #   fit the data for n line segments. if starting points needed (x_c & y_c: x_c=[x[0]], y_c=[y[0]])
    z = myPWLF.fit(n_lines)
    #calculate slopes
    slopes = myPWLF.calc_slopes()
    print("Z values:",z)
    print("a1,a2,a3 values:",slopes)
    
    #   predict for the determined points
    xHat = x#np.linspace(min(x), max(x), num=len(x))
    yHat = myPWLF.predict(xHat)
    print("--- day processed:" + str(time)+"---")
    
    return xHat,yHat,slopes,z

def graphic(x,y,xHat,yHat,time,k_lat,l_lon):
    """
 Parameters
        ----------
        x: float
             ERA5 array Height
        y: float
            array potential temperature profile ERA5
    """
    
    data = checkdate(time,k_lat,l_lon)
    # plot the results
    plt.figure()
    plt.plot(y, x,color="blue",label="ERA5",linestyle="-")
    plt.plot(yHat, x, color = "red"
             ,label="Fitting",linestyle="--")
    plt.xlabel('Theta') 
    plt.ylabel('Height (m)') 
    plt.title(data[2])
    plt.legend()
    plt.show()
    
def compute_regression(time_final):
    k_lat = 2
    l_lon = 4
    
    
        
    result = piecewise_regression(time_final,k_lat,l_lon,2)
    
    result_theta = result[1]
    result_slope = result[2]
    result_z = result[3]

    theta_pred = np.array(result_theta)
    slope = np.array(result_slope)
    z_points = np.array(result_z)
    return theta_pred,slope,z_points


#export(0,10,2,4)

def compute_parallel(np=4): #np=multiprocessing.cpu_count() to use all the cores
    """ np is number of processes to fork """

    p = multiprocessing.Pool(np)
    output = p.map(compute_regression, [i for i in range(8784)]) #8784

    
    return output
def post_process():
    """ post-processing """    
    output = compute_parallel()

    theta_pred, slope, z_points = zip(*output)
    #merge into one single array
    theta_pred = np.array(theta_pred)
    slope = np.array(slope)
    z_points = np.array(z_points)
    if len(theta_pred) and len(slope) and len(z_points)>0:
        np.save('/Users/mgermano/Documents/PhD/clustering/meeting_04:5/curving_fit/results/theta_pred.npy', theta_pred)
        np.save('/Users/mgermano/Documents/PhD/clustering/meeting_04:5/curving_fit/results/slope.npy', slope)
        np.save('/Users/mgermano/Documents/PhD/clustering/meeting_04:5/curving_fit/results/z_points.npy', z_points)   
    
    return print("Process Finished!")

#post_process()

########## plot graph
time_ERA =7435 
k_lat=2
l_lon= 4

data = checkdate(time_ERA,k_lat,l_lon)
    
zlevels=data[0] #ERA 5 model layer heights
pottemp=data[1]
result = piecewise_regression(time_ERA,k_lat,l_lon,3)
graphic(zlevels,pottemp,result[0],result[1],time_ERA,k_lat,l_lon)

print("--- That took %s seconds ---" % (time.time() - start_time))

