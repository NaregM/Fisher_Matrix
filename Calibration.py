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

class Calibration:

    def __init__(self, z, BM0 = 0.0, alpha = 0.0, sLnM0 = 0.1, beta = 0.0):

        self.z = z
        self.BM0 = BM0
        self.alpha = alpha
        self.sLnM0 = sLnM0
        self.beta = beta


    def BM(self):

        """
        Calibration parameter
        """
        return self.BM0 * (1 + self.z)**self.alpha


    def sigmaLnM(self):

        """
        Calibration parameter
        """
        return self.sLnM0 * (1 + self.z)**self.beta


    def xm(self, M, Mthr, m):

        """
        Eq. 18 from 1210.7276
        """
        DlnM = 0.3

        return (np.log10(Mthr) + m * DlnM - self.BM() - np.log10(M))/np.sqrt(2 * self.sigmaLnM()**2)
