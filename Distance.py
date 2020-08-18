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

delta_crit = np.sqrt(0.75) * 1.686    # Note: sqrt(0.75)
c = scipy.constants.c/1e3    # km/s
h = 0.697
rho_0 = 42285353995.6  # Msun / Mpc**3
delta_vir = 200

# Parameters
p_fid = {'H0': 73.000000000,
        'Omegab0': 0.045974855,
         'Omegam0': 0.268343029,
         'ns': 0.963000000,
         'w0': -1.000000000,
         'sigma_8': 0.809000000}


p_fid_fnl = {'H0': 73.000000000,
             'Omegab0': 0.045974855,
             'Omegam0': 0.268343029,
             'ns': 0.963000000,
             'w0': -1.000000000,
             'sigma_8': 0.809000000,
             'fnl': 0.00}

# ==============================================================

class Distance():

    def __init__(self, z, H0 = 67, Omegab0 = 0.02256/0.67**2, Omegam0 = (0.1142+0.02256)/0.67**2,
                 ns = 0.962, w0 = -1.0, wa = 0.0,
                 Tcmb0 = 2.75):

        self.z = z
        self.Tcmb0 = Tcmb0
        self.H0 = H0
        self.Omegab0 = Omegab0
        self.Omegam0 = Omegam0
        self.ns = ns
        self.w0 = w0
        self.wa = wa



    def E(self, z = None):

        """
        Eq. 2.3 from 1312.4430
        Assuming flat Universe.

        """

        if z == None:

            Om = self.Omegam0
            Ode = 1.0 - Om
            de_int = lambda z: (1.0 + self.w0 + self.wa * z/(1. + z)) / (1 + z)

            return np.sqrt(Om * (1 + self.z)**3 + Ode**(np.exp(3 * quad(de_int, 0, self.z)[0])))

        else:

            Om = self.Omegam0
            Ode = 1.0 - Om

            return np.sqrt(Om * (1 + z)**3 + Ode**(3*(1+self.w0)))


    def dL(self, z = None):

        """
        Luminosity distance; Based on Eq. 4 from 1110.2310
        """

        if z == None:

            return c * (1 + self.z)/(self.H0) * quad(lambda z: 1/self.E(), 0, self.z)[0]

        else:

            return c * (1 + z)/(self.H0) * quad(lambda z: 1/self.E(z), 0, z)[0]


    def Xi(self):

        """
        Comoving radial distance
        """

        return (c/self.H0) * quad(lambda z: 1/self.E(), 0, self.z)[0]


    def dVdz(self, dOmega = 4 * np.pi):

        """
        Based on Eq. 3 from 1110.2310
        """

        return dOmega * c * self.dL()**2 / ((1 + self.z)**2 * self.E() * self.H0)


    def V_comoving(self, dOmega = 4 * np.pi):

        """
        Comoving Volume; Eq. 3 from 1110.2310
        """

        return 1/0.7**3 * quad(lambda z: self.dVdz(), 0, self.z)[0]


    def DA(self):

        """
        Angular diameter distance
        """

        return (1 + self.z)**(-1) * self.Xi()
