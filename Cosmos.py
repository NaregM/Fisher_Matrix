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
from Distance import Distance
from Calibration import Calibration

# ====================================================================

delta_crit = np.sqrt(0.75) * 1.686    # Note: sqrt(0.75)
c = scipy.constants.c/1e3    # km/s
h = 0.697
rho_0 = 42285353995.6  # Msun / Mpc**3
delta_vir = 200

# ======================================================================

class Cosmos(Distance, Calibration):

    def __init__(self, z, kmin, kmax, N, Y200rhoc, sigma_8,
                 BM0 = 0.0, alpha = 0.0, sLnM0 = 0.1, beta = 0.0,
                 fnl = 0, bfNL = False, bfNL_Ph = False,
                 H0 = 67., Omegab0 = 0.02256/0.67**2, Omegam0 = (0.1142+0.02256)/0.67**2,
                 ns = 0.962, w0 = -1.0, wa = 0.0, Tcmb0 = 2.75,
                 Ps = None, Mlim = 1e14, Mcut = 4e14):

        Distance.__init__(self, z, H0, Omegab0, Omegam0, ns, w0, wa, Tcmb0)
        Calibration.__init__(self, z, BM0 = BM0, alpha = alpha, sLnM0 = sLnM0, beta = beta)

        self.kmin = kmin
        self.kmax = kmax
        self.N = N
        self.Mlim = Mlim
        self.Y200rhoc = Y200rhoc
        self.h = H0/100

        self.fnl = fnl
        self.bfNL = bfNL
        self.bfNL_Ph = bfNL_Ph

        self.sigma_8 = sigma_8
        self.Ps = Ps


    def Omz(self):

        return self.Omegam0 * (1 + self.z)**3 / (self.Omegam0 * (1 + self.z)**3 + (1 - self.Omegam0) * (1 + self.z)**(3 * (1 + self.w0)))

    def rho_m(self):
        """
        return the matter density for the current cosmology
        """
        mpc_to_cm = 3.0856e24
        crit_dens = 1.8791e-29*self.h*self.h*pow(mpc_to_cm, 3.0) # in grams Mpc^{-3}
        M_sun = 1.989e33 # in grams
        return crit_dens*self.Omz()/(M_sun) # in M_sun Mpc^{-3}

    def Ks(self):

        return np.linspace(self.kmin, self.kmax, self.N)#np.arange(self.kmin, self.kmax, 0.01)#


    def kd(self):

        DeltaK = 0.017

        kd = [1e-3]
        kd_ = 1e-3

        for i in range(8):

            kd_ += DeltaK
            kd.append(kd_)

        return np.array(kd)


    @classmethod
    def A_norm(cls, H0, Omegab0, Omegam0, ns, sigma_8, fnl):

        c = cls(z = 0.0, H0 = H0, Omegab0 = Omegab0, Omegam0 = Omegam0, ns = ns, fnl = fnl,
                kmin = 1e-5, kmax = 10, N = 1500, Y200rhoc = 1e-3, sigma_8 = sigma_8,
                BM0 = 0.0, alpha = 0.0, sLnM0 = 0.1, beta = 0.0)

        numerator = c.sigma_8**2           # sigma_8
        denominator = trapz((1/(2*np.pi**2)) * c.Ks()**(c.ns + 2) * c.T_we()**2 * c.W(c.R2M(8))**2, c.Ks())

        return numerator/denominator


    def T_we(self, k = None, bSingleK = False):

        """
        Transfer Function, Hu-E
        """
        if bSingleK == False:

            k = self.Ks()

        else:

            k = k

        omb = self.Omegab0
        om0 = self.Omegam0
        h = self.H0/100.
        theta2p7 = self.Tcmb0 / 2.7

        s = 44.5 * np.log(9.83/(om0*pow(h,2))) / np.sqrt(1.0 + 10.0 * pow(omb * h * h, 0.75))
        alphaGamma = 1.0 - 0.328 * np.log(431.0 * om0 * h * h) * omb / om0 + 0.38 * np.log(22.3 * om0 * h * h) * (omb / om0) * (omb / om0)
        Gamma = om0 * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + pow(0.43 * k* h * s, 4.0)))
        q = k * theta2p7 * theta2p7 / Gamma
        C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
        L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)

        Tk = L0 / (L0 + C0 * q * q)

        return Tk


    def setPower(self):

        T = self.T_we()
        Pk = self.Ks()**self.ns * T**2
        norm = self.A_norm(self.H0, self.Omegab0, self.Omegam0, self.ns, self.sigma_8, self.fnl)

        self.Ps = Pk * norm * self.d()**2


    def P_k(self, k):

        T = self.T_we(k, bSingleK=True)
        Pk = k**self.ns * T**2
        norm = self.A_norm(self.H0, self.Omegab0, self.Omegam0, self.ns, self.sigma_8, self.fnl)

        return Pk * norm * self.d()**2


    def Md(self):

        """
        Mass range for integration
        """

        DeltaM = 10**0.3 #0.07# 0.03

        Md = []
        Md_ = (1e11)/DeltaM

        for i in range(27):  # 200

            Md_ *= DeltaM
            Md.append(Md_)

        return np.array(Md)


    def Mh2M200(self, M200):

        """
        Assuming h = 200, and virial radius is 500 (v is 500)
        """

        f = lambda x: x**3 * (np.log(1.0 + 1.0/x) - 1.0/(1.0 + x))

        d = self.Omegam0 * (1 + self.z)**3 /(self.Omegam0 * (1 + self.z)**3 + (1 - self.Omegam0)) - 1

        # I think (1+d) instead of Omega0 is the right choice but 200/Omega0 makes it same as the paper
        DELTA_V = (200)/self.Omegam0 #

        a = [0.5116, -0.4283, -3.13e-3, -3.52e-5]

        c = 9.0/(1.0 + self.z) * pow(M200/(0.8e13), -0.13)

        f_h = DELTA_V/200 * f(1.0/c)

        p = a[1] + a[2] * np.log(f_h) * a[3] * (np.log(f_h))**2

        x_of_f = 1.0/np.sqrt(a[0] * f_h**(2*p) + (3.0/4)**2) + 2.0 * f_h

        return M200 / (DELTA_V/200) * (c * x_of_f)**3


    def setMlim(self):

        """
        Eq.1, from 1210.7276|

        """

        Mcut = (0.8e14) # 0.8e14

        if self.z == 0.025:

            self.Mlim = Mcut

        else:

            Y = (self.Y200rhoc/(60.)**2) / (57.296)**2  # arcmin to degree and degree to radian
            M200crit = ((self.DA())**2 * self.E()**(-2/3) * (Y/2.5e-4))**0.533 * (1e15)

            if M200crit <= Mcut:

                M200crit = Mcut


            self.Mlim = self.Mh2M200(M200crit)# M_solar


    def D_plus(self):

        """
        HMF paper
        """

        integrand = lambda z_: (1 + z_)/(np.sqrt(self.Omegam0 * (1 + z_)**3 + 1.0 - self.Omegam0))**3
        I = quad(integrand, self.z, np.inf, limit = 1500)[0]

        return 5./2 * self.Omegam0 * self.E() * I


    def D_plus_0(self):

        """
        HMF paper
        """
        Om = self.Omegam0
        integrand = lambda z_: (1 + z_)/(np.sqrt(self.Omegam0 * (1 + z_)**3 + 1.0 - self.Omegam0))**3
        I = quad(integrand, 0.0, np.inf, limit = 1500)[0]

        return 5./2 * self.Omegam0 * np.sqrt(self.Omegam0 * (1 + 0)**3 + 1.0 - self.Omegam0) * I


    def d(self):

        """
        Eq.11, HMF paper
        """

        return self.D_plus()/self.D_plus_0()


    def dOMEGA(self, fsky):

        """
        Returns the solid angle covered by cluster survey given the sky-coverage

        """
        return fsky * 41253 * (np.pi/180)**2


    def M2R(self, M):

        """
        Radius of virialized mass
        """
        d = self.Omegam0 * (1 + self.z)**3 /(self.Omegam0 * (1 + self.z)**3 + (1 - self.Omegam0)) - 1

        DELTA_V = (200)/self.Omegam0 #(1+d)

        return ((3.0 * M)/(4.0 * np.pi * self.rho_m()*DELTA_V))**0.33


    def R2M(self, R):

        d = self.Omegam0 * (1 + self.z)**3 /(self.Omegam0 * (1 + self.z)**3 + (1 - self.Omegam0)) - 1

        DELTA_V = (200)/self.Omegam0#(1+d)

        return (4.0/3 * np.pi * self.rho_m()*DELTA_V) * R**3


    def W(self, M, k = None, bSingleK = False):

        """
        Top-hat Window function, M in units of M_sun
        """

        if bSingleK:

            k = k

        else:

            k = self.Ks()

        return 3.0 * (np.sin(k * self.M2R(M)) - (k*self.M2R(M)) * np.cos(k*self.M2R(M)))/(k*self.M2R(M))**3.0


    def sigma_squared(self, M, bR = False, j = 0):

        """
        Mass variance. arXiv:0712.0034v4 eq.1
        """
        #if bR == True:

        #R_ = self.M2R(M)
        K_ = self.Ks()
        win = self.W(M)

        I = (K_)**(2. + 2.*j) * self.Ps * win**2

        return 1.0/(2.0*np.pi**2) * trapz(I, x = K_)

        #elif bR == False:

         #   K_ = self.Ks()
        #    win = self.W(M)

           # I = (K_)**(2. + 2.*j) * self.Ps * win**2
    #
         #   return 1.0/(2.0*np.pi**2) * trapz(I, x = K_)


    def nu(self, M):

        """
        Peak height
        """

        return (delta_crit)/np.sqrt(self.sigma_squared(M))


    def dW2dM(self, M):

        """
        From HMF paper
        """
        k = self.Ks()
        return (np.sin(k * self.M2R(M)) - k * self.M2R(M) * np.cos(k*self.M2R(M))) * (np.sin(k*self.M2R(M)) * (1 - 3./(k*self.M2R(M))**2)
                + 3 * np.cos(k*self.M2R(M))/(k*self.M2R(M)))


    def S3(self, M):

        """
        sigma_R and sigma_M should get clarified
        """
        #R_ = self.M2R(M)
        return 3.15 * 1e-4 * self.fnl / (np.sqrt(self.sigma_squared(M)))**0.838


    def R_nl(self, M):

        """

        Equation 8

        """

        dc = delta_crit#/self.d()
        sm2 = self.sigma_squared(M)
        # sigma to the -1.838 changged to -0.838
        return 1 + 1./6 * (sm2/dc) * (self.S3(M) * (dc**4 / sm2**2 - 2 * dc**2 / sm2 - 1) + \
                                    -3.15e-4 * self.fnl * 0.838 * sm2**0.5 * np.sqrt(sm2)**(-0.838) * (dc**2/sm2 - 1))


    def dlnsdlnM(self, M):

        """
        dlnSigma/dlnM from HMF paper
        """
        K_ = self.Ks()
        I = self.Ps/(K_**2) * self.dW2dM(M)

        return (3./(2 * np.pi**2 * self.M2R(M)**4 * self.sigma_squared(M))) * simps(I, K_)

    # ====================================================================
    # fnl relaed functions
    # ====================================================================

    def M_R(self, k, M):

        """
        Premordil power eq. from
        """

        return 2.0/3 * (self.T_we(k, bSingleK=True) * k**2)/(((self.H0/3e5)**2) * self.Omegam0) * self.W(M, k, bSingleK = True)


    def P_phi(self, k):

        """
        Newtonian potential from MG paper
        """
        norm = self.A_norm(self.H0, self.Omegab0, self.Omegam0, self.ns, self.sigma_8, self.fnl)

        return 9.0/4 * norm * (self.H0/3e5)**4 * self.Omegam0**2 * k**(self.ns - 4)


    def F_R_K(self, k, M):

        """
        From Verde, Mattarasee for a single k-value

        """

        fnl = self.fnl
        #M = self.Mlim

        sR2 = self.sigma_squared(M)
        #print(sR2)
        ksi = np.arange(1e-6, 1.5, 0.03)#np.linspace(0.0, 150, 480)#np.arange(1e-3, 5, 0.01)#np.arange(1e-6, 10, 0.01)   #np.linspace(self.Ks().min(), self.Ks().max(), self.Ks().size//4)# - 1e-9
        mu = np.linspace(-1.00+1e-2, 1.00-1e-2, len(ksi))  #1e-3?

        kksi, mmu = np.meshgrid(ksi, mu, indexing = 'ij', sparse = True, copy = False)

        alpha = np.sqrt(kksi**2 + k**2 + 2 * mmu * k * kksi)

        I = kksi**2 * self.M_R(kksi, M) * self.P_phi(kksi) * self.M_R(alpha, M) * (2.0 + self.P_phi(alpha)/self.P_phi(k))

        #S = S_simps(len(ksi))

        return (2 * fnl/(8. * np.pi**2 * sR2)) * trapz(trapz(I, ksi), mu)#


    def Delta_b_nG(self, M, k):

        """
        Non-Gaussian scale dependent correction to bias
        """

        return self.F_R_K(k, M)/self.M_R(k, M)


    def b_L(self, M, k = None):

        """
        Linear bias
        """
        a, p = 0.75, 0.3

        nu_ = self.nu(M)

        bL = 1 + (a * nu_**2 - 1)/delta_crit + 2 * p/(delta_crit * (1 + (a * nu_**2)**p))

        if self.bfNL_Ph == True:

            return  bL + (bL - 1) * delta_crit/self.d() * self.Delta_b_nG(M, k)

        elif self.bfNL_Ph == False:

            return bL


    def f_J(self, M):

        """
        Jenkins fitting function (There is a typo in HMF?)
        """
        s = np.sqrt(self.sigma_squared(M))

        return 0.315 * np.exp(-np.absolute(0.61 + np.log(s**(-1.)))**(3.8))  # *self.d() took out redundant


    def f_T(self, M):

        """
        Tinker fitting function
        """

        return None


    def dndlnM(self, M):

        """
        n(M,z)
        """
        if self.bfNL == False:

            return (self.rho_m()/ M) * self.f_J(M) * np.absolute(self.dlnsdlnM(M)) #self.rho_m() *(1+self.z)**3

        elif self.bfNL == True:

            return (self.dndM_PS(M)/self.dndM_PSng(M)) * (self.rho_m()*(1+self.z)**3 / M) * self.f_J(M) * np.absolute(self.dlnsdlnM(M))


    def dndM(self, M):

        """
        n(M,z)
        """
        if self.bfNL == False:

            return (self.rho_m() / M**2) * self.f_J(M) * np.absolute(self.dlnsdlnM(M)) #self.rho_m() *(1+self.z)**3

        elif self.bfNL == True:

            return (self.dndM_PS(M)/self.dndM_PSng(M)) * (self.rho_m()*(1+self.z)**3 / M**2) * self.f_J(M) * np.absolute(self.dlnsdlnM(M))


    def dndM_ng(self, M):

        """

        """
        self.R_nl()


    def dndM_PS(self, M):

        """
        Press-Schechter mass function. Eq. 4.17 from 0711.4126

        """
        dc = delta_crit#/self.d()
        sm2 = self.sigma_squared(M)
        norm = -np.sqrt(2/np.pi)

        return norm * (self.rho_m() * (1+self.z)**3 / M**2) * (dc/np.sqrt(sm2)) * self.dlnsdlnM(M) * np.exp(-dc**2 / (2*sm2))


    def dndM_PSng(self, M):

        """
        Non-Gaussian PS mass function. Eq. 4.19 from 0711.4126

        """
        dc = delta_crit#/self.d()
        sm2 = self.sigma_squared(M)

        R = self.M2R(M)

        prefac = -np.sqrt(2/np.pi) * (self.rho_m()*(1+self.z)**3 /M) * np.exp(-dc**2 / (2*sm2))
        term1 = 1/M * self.dlnsdlnM(M) * (dc/sm2**0.5 + 1/6 * sm2**0.5 * self.S3(M) * ((dc/sm2)**2 \
                                                                                  - 2 * dc**2 / sm2 - 1))

        #term2 = 1/6 * sm2**0.5 * ((self.S3(M+1e-2) - self.S3(M-1e-2))/(2e-2)) * (dc**2 / sm2 - 1)

        term2 = 1/6 * sm2**0.5 * ((self.S3(M*(1+1e-2)) - self.S3(M*(1-1e-2)))/(M*2e-2)) * (dc**2 / sm2 - 1)

        return prefac * (term1 + term2)


    def b_eff(self, m, k = None):

        """

        """
        Md = self.Md()

        nM = np.array([self.dndM(M_) for M_ in Md])
        nM = np.where(np.isnan(nM) == True, np.nanmin(nM), nM)

        ERF = np.array([erfc(self.xm(M_, self.Mlim, m)) - erfc(self.xm(M_, self.Mlim, m + 1)) for M_ in Md])


        if self.bfNL_Ph == False:

            bL = np.array([self.b_L(M_) for M_ in Md])
            bias = trapz(nM * bL * ERF, x = Md)/ trapz(nM * ERF, x = Md)

            return bias

        elif self.bfNL_Ph == True:

            #bb = lambda k, m: self.b_L(m, k)

            bL = np.array([self.b_L(M_, k) for M_ in Md], dtype = np.float32)
            bias = trapz(nM * bL * ERF, x = Md)/ trapz(nM * ERF, x = Md)

            return bias


    def Ph_mn(self, m, n):

        """
        Galaxy power spectrum
        """
        kmin = 1e-3
        kmax = 0.136
        Psk = np.array([self.P_k(k) for k in self.kd()])
        if self.bfNL_Ph == False:


            bm = self.b_eff(m)
            bn = self.b_eff(n)

            return bm * bn * Psk

        elif self.bfNL_Ph == True:

            #ks = np.linspace(kmin, kmax, len(self.Ks()))
            bm = np.array([self.b_eff(m, k) for k in self.kd()], dtype = np.float32)
            bn = np.array([self.b_eff(n, k) for k in self.kd()], dtype = np.float32)

            return bm * bn * Psk


    def Veff_mn(self, m, n):

        """

        """

        Ps = np.array([self.P_k(k) for k in self.kd()])
        beff_m = self.b_eff(m)
        beff_n = self.b_eff(n)

        Pmn = beff_m * beff_n * Ps
        Pnm = beff_n * beff_m * Ps
        Pmm = beff_m * beff_m * Ps
        Pnn = beff_n * beff_n * Ps

        nm = self.dndlnM(self.Mlim * 10**(m*0.3))
        nn = self.dndlnM(self.Mlim * 10**(n*0.3))

        n_ = (Pmn)**2 * nm * nn

        delta_nm = 1 if n == m else 0

        d1 = (nm * Pmm + 1) * (nn * Pnn + 1)
        d2 = nm * nn * (Pnm + delta_nm/nm)**2

        return self.V_comoving() * n_/(d1 + d2)


    def VarCrossPower(self, m, n):

        """
        Variance of cross powerspectrum term, eq A1
        """

        Nmod = 1

        Pmn = self.b_eff(m) * self.b_eff(n) * self.Ps
        Pmm = self.b_eff(m) * self.b_eff(m) * self.Ps
        Pnn = self.b_eff(n) * self.b_eff(n) * self.Ps

        Mpl = self.Md()[self.Md() < 1e16]
        nm = self.dndlnM(Mpl[m])            #    >>Is this dndm or dndln<< ???????
        nn = self.dndlnM(Mpl[n])

        delta_nm = 1 if n == m else 0

        return 1/Nmod * ((Pmm + 1/nm) * (Pnn + 1/nm) + (Pmn + delta_nm/nm)**2)
