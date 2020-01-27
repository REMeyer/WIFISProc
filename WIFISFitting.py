import numpy as np
import scipy.optimize as spo
from scipy.special import wofz
from astropy.modeling.models import Voigt1D

#Normal gauss
def gauss(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [A, sigma, mean]'''
    return  p0[0] * np.exp((-1.0/2.0) * ((xs - p0[2])/p0[1])**2.0)

def gaussian(xs, a, sigma, x0):
    second = ((xs - x0)/sigma)**2.0
    full = a*np.exp((-1.0/2.0) * second)
    
    return full

### Gauss with y-offset
def gauss_os(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [A, sigma, mean, offset]'''
    return  p0[3] + p0[0] * np.exp((-1.0/2.0) * ((xs - p0[2])/p0[1])**2.0)

def gaussian_os(xs, a, sigma, x0, b):
    second = ((xs - x0)/sigma)**2.0
    full = b + a*np.exp((-1.0/2.0) * second)
    
    return full

### Natural gauss
def gauss_nat(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [sigma, mean]'''
    return  (1 / (2.*np.pi*p0[0]**2.))* np.exp((-1.0/2.0) * ((xs - p0[1])/p0[0])**2.0)

def gaussian_nat(xs, sigma, x0):
    second = ((xs - x0)/sigma)**2.0
    full = (1/(2*np.pi*sigma**2.))*np.exp((-1.0/2.0) * second)
    
    return full

### Lorentzian
def lorentz(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [sigma, mean]'''
    full = p0[0] * (0.5 * p0[1]) / ((xs - p0[2])**2. + (0.5*p0[1])**2.) + p0[3]
    return full 

def lorentzian(xs, a, sigma, mean, b):
    p = [a,sigma,mean,b]
    full = p[0] * (0.5 * p[1]) / ((xs - p[2])**2. + (0.5*p[1])**2.) + p[3]
    return full

### Voigt
def voigt(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [sigma, mean]'''
    v = Voigt1D(x_0 = p0[0], amplitude_L = p0[1], fwhm_L = p0[2], fwhm_G = p0[3])
    full = p0[4] + v(xs)
    return full 

def voigtfull(xs, mean, aL, sigmaL, sigmaG, offset):
    p = [mean, aL, sigmaL, sigmaG, offset]
    v = Voigt1D(x_0 = p[0], amplitude_L = p[1], fwhm_L = p[2], fwhm_G = p[3])

    full = p[4] + v(xs)
    return full

### VoigtNorm
def voigtnorm(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [sigma, mean]'''
    full = np.real(wofz(xs + 1j*p0[0])) / (p0[1] * np.sqrt(2.*np.pi))

    return full 

def voigtfullnorm(xs, gamma, sigma):
    p = [gamma, sigma]
    full = np.real(wofz((xs + 1j*gamma)/(sigma * np.sqrt(2.))) / (sigma * np.sqrt(2.*np.pi)))

    return full

### Function for performing the fit
def fit_func(xdata, ydata, p0, func=gaussian):
    '''Performs a fit of the function <func> using scipy.optimize.curvefit. Returns 
    the fit parameters and the covariance matrix.
    
    # Inputs #
    xdata:      xdata to be fit
    ydata:      ydata to be fit
    p0:         Initial guess for parameters in order of input for <func>
    '''

    popt, pcov = spo.curve_fit(func, xdata, ydata, p0=p0)
    return [popt, pcov]
