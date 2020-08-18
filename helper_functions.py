import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.integrate import quad, trapz, simps
from scipy.stats import chisquare
from scipy.special import erfc

import camb
from camb import model, initialpower

import scipy
from scipy.optimize import curve_fit

import copy

from tqdm import tqdm

import itertools 



def plot_fsiher_2(F_param, color_, label, ax_, p_fid = p_fid, calib = False):
    
    F = 0.7 * tri2Full(F_param, p_fid)

    if calib == True:
        
        F[6, 6] += 1/0.05**2
        F[7, 7] += 1/1.0**2
        F[8, 8] += 1/0.1**2
        F[9, 9] += 1/1.0**2
    
    F_inv = np.linalg.inv(F)

    F_inv_46 = np.array([[F_inv[2, 2], F_inv[2, 5]], 
                         [F_inv[5, 2], F_inv[5, 5]]])

    Fisher_2nd(F_inv_46, p_fid_fnl['Omegam0'], p_fid_fnl['sigma_8'], r'$\Omega_m$', r'$\sigma_8$',
           color_, label, ax)
    plt.margins(0.5)
    

def sigma_fnl(F_param, p_fid = p_fid, bPlot = False):
    
    F = 0.7 * tri2Full(F_param, p_fid)
    F_inv = np.linalg.inv(F)

    F_inv_46 = np.array([[F_inv[5, 5], F_inv[5, 6]], 
                         [F_inv[6, 5], F_inv[6, 6]]])
    
    if bPlot:
        
        fig, ax = plt.subplots()

        Fisher(F_inv_46, p_fid_fnl['sigma_8'], p_fid_fnl['fnl'], r'$\sigma_8$', r'$f_{nl}$', ax)
        
        plt.margins(0.5)

    return np.sqrt(F_inv[6, 6])


def S_simps(N):

    """

    """
    S0 = [1, 1]

    for i in range(1, (N-2) + 1):

        if i % 2 == 0:

            S0.insert(i, 2)

        else:

            S0.insert(i, 4)

    S0 = np.array(S0)

    S = np.zeros((N, N))

    for i in range(N):

        S[i, :] = S0

    for i in range(1, N-1):

        if i % 2 == 1:

            S[i, :] = 4 * S[i, :]

        else:

            S[i, :] = 2 * S[i, :]

    return S


def simps_2d(fxy, x, y):

    """

    """

    I_simps = 0.0

    N = x.size

    h_x = (x.max() - x.min())/(N - 1)
    h_y = (y.max() - y.min())/(N - 1)

    S = S_simps(N)

    for i in range(x.size):

        for j in range(y.size):

            I_simps += 1.0/9 * h_x * h_y * S[i, j] * fxy[i, j]

    return I_simps


def tri2Full(M, params):
    
    """
    Helper function to generate Full matrix from upper triangular
    """
    n = len(params)

    F = np.zeros((n, n)) 
    triu = np.triu_indices(n)           # Find upper right indices of a triangular nxn matrix
    tril = np.tril_indices(n, -1)       # Find lower left indices of a triangular nxn matrix

    F[triu] = np.array(list(M.values()))    # Assign list values to upper right matrix
    F[tril] = F.T[tril]                     # Make the matrix symmetric

    return F


def ELLIPSE(x0, y0, a, b, theta, n = 1000):

    theta = np.deg2rad(theta)
    t = np.linspace(0, 2*np.pi, n)
    XX = x0 + a*np.cos(theta)*np.cos(t) - b*np.sin(theta)*np.sin(t)
    YY = y0 + a*np.sin(theta)*np.cos(t) + b*np.cos(theta)*np.sin(t)
    e = (XX, YY)
    
    return e


def Fisher(F, x_fiducial, y_fiducial, xlabel, ylabel, ax):

    """
    F is actually F^-1, the covariance matrix
    """

    #.figure(figsize = (12, 8))

    #sigma_X, sigma_Y = np.sqrt(np.absolute(np.linalg.inv(F)[0,0])), np.sqrt(np.absolute(np.linalg.inv(F)[1, 1]))
    #sigma_XY = np.sqrt(np.absolute(np.linalg.inv(F)[0, 1])) 
    
    #sigma_X, sigma_Y = np.sqrt(np.abs(F[0, 0])), np.sqrt(np.abs(F[1, 1]))
    #sigma_XY = np.sqrt(np.abs(F[0, 1]))
    # Change above
    sigma_X2, sigma_Y2 = F[0, 0], F[1, 1]
    sigma_XY = F[0, 1]

    a2 = 0.5 * (sigma_X2 + sigma_Y2) + np.sqrt(0.25 * (sigma_X2 - sigma_Y2)**2 + sigma_XY**2) # taking np.absolute( out
    b2 = 0.5 * (sigma_X2 + sigma_Y2) - np.sqrt(0.25 * (sigma_X2 - sigma_Y2)**2 + sigma_XY**2)
    theta = np.rad2deg(0.5 * np.arctan2((2*sigma_XY),(sigma_X2 - sigma_Y2)))  # np.arctan wont work
    
    
    COLORS = ['#1f77b4', 'r', 'royalblue']
    COLORS0 = ['turquoise', 'c', 'teal']
    COLORS1 = ['orangered', 'orange', 'tomato']
    COLORS2 = ['dodgerblue', 'royalblue']
    
    for i, Alpha in enumerate([1.52]): # 2,3 - sigma 2.48, 3.44
        
       	E = ELLIPSE(x_fiducial, y_fiducial, Alpha*np.sqrt(a2/2), Alpha*np.sqrt(b2/2), theta) #Alpha*1/2
        
        if i == 1:
            
            ax.plot(E[0], E[1], label = str(i+1) + r'$\sigma$', lw=3.0, alpha = 1, zorder = i, ls = "--")
            
        else:
            
        
            ax.plot(E[0], E[1], label = str(i+1) + r'$\sigma$', lw=3.0, alpha = 1, zorder = i)
            
        
        ax.set_xlabel(xlabel, size = 16)
        ax.set_ylabel(ylabel, size = 16)
        ax.legend(prop={'size': 12}, loc = 'best')
        #plt.xlim([x_fiducial - x_fiducial*0.0001, x_fiducial + x_fiducial*0.0001])
        #plt.ylim([y_fiducial - y_fiducial*0.0001, y_fiducial + y_fiducial*0.0001])
        ax.margins(0.35);

        
def Fisher_2nd(F, x_fiducial, y_fiducial, xlabel, ylabel, color_, label, ax):

    """
    F is actually F^-1, the covariance matrix
    """
    
    Alpha = 1.52
    
    sigma_X2, sigma_Y2 = F[0, 0], F[1, 1]
    sigma_XY = F[0, 1]

    a2 = 0.5 * (sigma_X2 + sigma_Y2) + np.sqrt(0.25 * (sigma_X2 - sigma_Y2)**2 + sigma_XY**2) # taking np.absolute( out
    b2 = 0.5 * (sigma_X2 + sigma_Y2) - np.sqrt(0.25 * (sigma_X2 - sigma_Y2)**2 + sigma_XY**2)
    theta = np.rad2deg(0.5 * np.arctan2((2*sigma_XY),(sigma_X2 - sigma_Y2)))  # np.arctan wont work
    
        
    E = ELLIPSE(x_fiducial, y_fiducial, Alpha*np.sqrt(a2/2), Alpha*np.sqrt(b2/2), theta) #Alpha*1/2
        
    ax.plot(E[0], E[1], label = label, lw = 2.6, alpha = 1,
                     color = color_)
    
    ax.set_xlabel(xlabel, size = 19)
    ax.set_ylabel(ylabel, size = 19)
    ax.legend(prop = {'size': 12}, loc = 'best', ncol = 1)
    ax.margins(0.35);

    
def Fisher_compare(F0, F1, x_fiducial, y_fiducial, xlabel, ylabel, Alpha, size = (9, 6)):
    plt.figure(figsize=size)

    sigma_X0, sigma_Y0 = np.sqrt(np.absolute(np.linalg.inv(F0)[0,0])), np.sqrt(np.absolute(np.linalg.inv(F0)[1, 1]))
    sigma_XY0 = np.sqrt(np.absolute(np.linalg.inv(F0)[0, 1])) 
    
    sigma_X1, sigma_Y1 = np.sqrt(np.absolute(np.linalg.inv(F1)[0,0])), np.sqrt(np.absolute(np.linalg.inv(F1)[1, 1]))
    sigma_XY1 = np.sqrt(np.absolute(np.linalg.inv(F1)[0, 1])) 
    
    a2_0 = np.absolute(0.5 * (sigma_X0**2 + sigma_Y0**2) + np.sqrt(0.25 * (sigma_X0**2 - sigma_Y0**2)**2 + sigma_XY0**2))
    b2_0 = np.absolute(0.5 * (sigma_X0**2 + sigma_Y0**2) - np.sqrt(0.25 * (sigma_X0**2 - sigma_Y0**2)**2 + sigma_XY0**2))
    theta_0 = np.rad2deg(0.5 * np.arctan((2*sigma_XY0)/(sigma_X0**2 - sigma_Y0**2)))
    
    a2_1 = np.absolute(0.5 * (sigma_X1**2 + sigma_Y1**2) + np.sqrt(0.25 * (sigma_X1**2 - sigma_Y1**2)**2 + sigma_XY1**2))
    b2_1 = np.absolute(0.5 * (sigma_X1**2 + sigma_Y1**2) - np.sqrt(0.25 * (sigma_X1**2 - sigma_Y1**2)**2 + sigma_XY1**2))
    theta_1 = np.rad2deg(0.5 * np.arctan((2*sigma_XY1)/(sigma_X1**2 - sigma_Y1**2)))
    
    E0 = ELLIPSE(x_fiducial, y_fiducial, Alpha*np.sqrt(a2_0)/2., Alpha*np.sqrt(b2_0)/2., theta_0)
    E1 = ELLIPSE(x_fiducial, y_fiducial, Alpha*np.sqrt(a2_1)/2., Alpha*np.sqrt(b2_1)/2., theta_1)

    plt.plot(E0[0], E0[1], c = 'C0', label = '50 Mpc cut', lw = 2.5)
    plt.plot(E1[0], E1[1], c = 'C1', label = '10 Mpc cut', lw = 2.5)
    plt.scatter(x_fiducial, y_fiducial, label = 'Fiducial Value', c = 'k', lw = 0.25)
    plt.xlabel(xlabel, size = 22)
    plt.ylabel(ylabel, size = 22)
    plt.legend(prop={'size':13})
    
