####################
#   Python Class that extracts spectra from WIFIS datacubes, 
#   corrects them with a telluric star spectrum,
#   derives overall uncertainties, and writes solution to file.
#   
#   Author: Elliot Meyer, 
#            Dept Astronomy & Astrophysics University of Toronto
#   Date: 2018-09-17
#   Adapted telluric reduction code developed by Margaret Ikape in 2017/2018
####################

########################################
# READ THE main class __init__ comments
# Look at the bottom of this script for an usage example for these classes.
########################################

import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
import numpy as np
from astropy.io import fits
from sys import exit

import matplotlib.pyplot as mpl
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gridspec

from glob import glob
import gaussfit as gf
import os

from astropy.visualization import (PercentileInterval, LinearStretch,
                                    ImageNormalize, ZScaleInterval)
from astropy import wcs
from astropy import units as u
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import Angle

mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

def norm(spec):
    ''' Quick function to median normalize an array.'''
    return spec/np.median(spec)

def circle_spec(spec, x0, y0, radius, annulus = False):
    '''Function that returns a mask defining the regions within a 
        circular aperature or in an circular annulus

        Inputs:
            x0:      x center
            y0:      y center
            radius:  radius in pixels
            annulus: True or False
    '''

    # Create x,y grid for the mask and define the circular region.
    shape = spec.shape
    xx, yy = np.mgrid[:shape[1],:shape[2]]
    circle = (xx - x0)**2. + (yy - y0)**2.

    if annulus:
        # If annulus, create a mask only within the two annular limits
        whgood = (circle <= (annulus**2.0)) & (circle >= (radius**2.0))
    else:
        # If circle, create a mask within the circle radius
        whgood = circle <= radius**2.0

    # return the mask
    return whgood

def ellipse_region(cube, center, a, ecc, theta, annular = False):
    ''' Function that returns a mask that defines the elements that lie
        within an ellipsoidal region. The ellipse can also be annular.
        
        Inputs:
            Cube:    The data array
            center:  Tuple of the central coordinates (x_0, y_0) (spaxels)
            a:       The semi-major axis length (spaxels).
            ecc:     The eccentricity of the ellipse (0<ecc<1)
            theta:   The rotation angle of the ellipse from vertical (degrees)
                     rotating clockwise. 
            
        Optional Inputs:
            annular: False if just simple ellipse, otherwise is the INNER
                     annular radius (spaxels)
        
    '''
    
    # Define angle, eccentricity term, and semi-minor axis
    an = Angle(theta, 'deg')
    e = np.sqrt(1 - (ecc**2))
    b = a * ecc
    
    # Create outer ellipse
    ell_region = Ellipse2D(amplitude=10, x_0 = center[0], y_0 = center[1],\
                a=a, b=b, theta=an.radian)
    
    if annular:
        # Define inner ellipse parameters and then create inner mask
        a2 = annular
        b2 = a2 * ecc
        ell_region_inner = Ellipse2D(amplitude=10, x_0 = center[0],\
                y_0 = center[1], a=a2, b=b2, theta=an.radian)
        
        # Set region of outer mask and within inner ellipse to zero, leaving
        # an annular elliptical region
        ell_region[ell_region_inner > 0] = 0
    
    # Return the mask
    return ell_region

def fwhm2sigma(fwhm):
    ''' Quick function to convert a gaussian fwhm to a standard deviation.'''
    
    return fwhm / np.sqrt(8 * np.log(2))

def create_vega_con():
    '''One-off function to create a vega spectrum for use in correcting WIFIS 
    telluric spectra'''
    
    vega = pd.read_csv("vega.txt", sep = "\s+", header=None)
    vega.columns = ['a', 'b', 'c']
    n_points = 895880
    x_vals = vega['a']*10#to convert to angstroms
    y_vals = vega['b']

    sigma = fwhm2sigma(7)
    
    # Make Gaussian centered at 13 with given sigma
    x_position = 10049
    kernel_at_pos = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))

    # Make kernel sum to 1
    kernel_at_pos = kernel_at_pos / sum(kernel_at_pos)

    ## Number of kernel points before center (at 0)
    kernel_n_below_0 = int((len(kernel_at_pos) - 5 ) / 2.567)

    convolved_y = np.convolve(y_vals, kernel_at_pos)
    ##print(convolved_y[13+ kernel_n_below_0])

    smoothed_by_convolving = \
        convolved_y[kernel_n_below_0:(n_points+kernel_n_below_0)]
    f1 = interp1d(x_vals, smoothed_by_convolving, kind='cubic')

    #plt.plot(x_vals, smoothed_by_convolving/210000, \
    #label='smoothed vega')#/230000

    #"""Saving the convolved Vegaj to file"""
    c = np.where((x_vals >= 8000) & (x_vals <= 13500))[0]
    var = zip(x_vals[c], (smoothed_by_convolving/210000)[c])
    np.savetxt("vega-con-new.txt", var)
    
def convolvemodels(wlfull, datafull, veldisp):

    reg = (wlfull >= 9500) & (wlfull <= 13500)
    
    wl = wlfull[reg]
    data = datafull[reg]

    c = 299792.458

    #Sigma from description of models
    m_center = 11500
    m_sigma = np.abs((m_center / (1 + 100./c)) - m_center)
    f = m_center + m_sigma
    v = c * ((f/m_center) - 1)
    
    sigma_gal = np.abs((m_center / (veldisp/c + 1.)) - m_center)
    sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    convolvex = np.arange(-5*sigma_conv,5*sigma_conv, 2.0)
    gaussplot = gf.gauss_nat(convolvex, [sigma_conv,0.])

    out = np.convolve(datafull, gaussplot, mode='same')

    return out

###################################

class WIFISSpectrum():
    ''' Class designed to extract WIFIS spectra from reduced datacubes
        for easy post-pipeline processing.''' 
    
    def __init__(self, cubefile, z, limits = False, circle = False, \
                 ellipse = False):
        '''
        ### Inputs ###
        cubefile:   path to datacube
        z:          the redshift of the target object'''

        self.cubefile = cubefile
        cubebase = self.cubefile.split('/')[:-1]
        self.cubebasepath = '/'.join(cubebase) + '/'

        # Extract cube data and header
        cubeff = fits.open(self.cubefile)
        self.cubedata = cubeff[0].data
        self.cubehead = cubeff[0].header
        
        # Create a flux image of the datacube
        self.cubeim = np.nansum(self.cubedata, axis = 0)
        
        # Generate wavelength array from header in Angstroms
        pixel = np.arange(self.cubedata.shape[0]) + 1.0
        cubewl = pixel*self.cubehead['CDELT3'] + self.cubehead['CRVAL3']
        self.cubewl *= 1e10

        self.z = z
        
        # Limits of a rectangle (x0,y0) [bottom left], (x1,y1) [top right] 
        # in the form [x0,x1,y0,y1]
        self.limits = limits  
        # Define a circle with center (x0,y0) in the form [x0,y0,radius, annulus]
        self.circle = circle 
        # Define an ellipse with center (x0,y0), angle theta (deg), 
        # semi-major axis (spaxels), and eccentricity (e) in the form 
        # [x0,y0,theta,a,e, annulus]
        self.ellipse = ellipse
        
        # Some housekeeping parameters to check status of spectral extraction
        self.extracted = False
        self.uncertainties = False
        
    def get_uncertainties(self, mode = 'Direct'):
        '''Function that estimates uncertainties in two possible fashions:
        
        1 - Direct) Directly determines the noise in the extracted region 
                    by simply taking the square root of the signal and adding
                    predetermined sky/thermal noise. 
        2 - SEM)    Calculated the standard error of the mean (SEM) of the set 
                    of individual observations. Takes the SEM between the 
                    extracted regions in each individual frame. 
                    Currently only works on science target frames. This method
                    should likely be used only when the number of individual 
                    observations are 10+. Not sure if this is reliable for 
                    WIFIS data'''
        
        if mode == 'Direct':
            
            # Get cube integration time
            inttime = self.cubehead['INTTIME']
            gain = 1.33            
            datasqrt = np.sqrt(data * inttime * gain)
            
            try:
                skyfls = glob(self.cubebasepath + '*_sky_cube.fits')
                skyff = fits.open(skyfls[0])
                skydata = skyff[0].data
                skyhead = skyff[0].header
                skyflat = np.nanmedian(skydata.reshape(skydata.shape[0],-1),\
                                       axis = 1)
                
                pixel = np.arange(skydata.shape[0]) + 1.0
                skywl = pixel*skyhead['CDELT3'] + skyhead['CRVAL3']
                skywl *= 1e10
                skyinterp = np.interp(self.cubewl, skywl, skydata,\
                                      left = 0.1, right = 0.1)
                
            except:
                print('### Problem finding and processing sky cubes,'+\
                      ' are they in the same directory as the merged cube?')
                skyinterp = np.ones(data.shape) * 8.0
            
            skysqrt = np.sqrt(skyflat * inttime * gain)
            
            #thermalsqrt = np.sqrt(0.22 * inttime * gain)
            thermalsqrt = 0
            
            noise = np.sqrt(datasqrt**2.0 + skysqrt**2.0 + thermalsqrt**2.0)
            
            self.galerr = noise
            self.uncertainties = True

        if mode == 'SEM':
            
            #Checking to see if spectrum is extracted
            if not self.extracted:
                print("Need to extract spectrum to derive uncertainties...")
                return

            #Getting the filepaths -- assumes the individual exposures are 
            #within the merged cube directory as standard by the WIFIS Pipeline
            tarfls = glob(self.cubebasepath + '*_obs_cube.fits')

            #Extracting the spectra
            masterarr = []
            for i,fl in enumerate(tarfls):
                wl, spec, head = self.extract_spectrum(fl, tartype)

                # Interpolating the spectra onto the same wavelength grid as
                # median galaxy
                galinterp = np.interp(self.galwl, wl, spec, left = 0, right = 0)
                masterarr.append(galinterp)

            #Calculating the uncertainties
            masterarr = np.array(masterarr)
            errarr = np.std(masterarr, axis = 0) / np.sqrt(masterarr.shape[0])

            #Setting the class values
            self.cubeerr = errarr
            self.uncertainties = True
            
    def extract_spectrum(self):
        '''Function that extracts a telluric or science spectrum in the
        aperture provided.
        '''

        if (self.limits == False) and (self.circle == False) and \
          (self.ellipse == False):
            print("Extraction limits not set. Please set the "+\
                  "relevant ellipse, circle or limits class variables "+\
                  "to extract the spectra")
            return

        #Slicing telluric star, then taking the mean along the spatial axes.
        if self.limits:
            specslice = self.cubedata[:,limits[0]:limits[1],limits[2]:limits[3]]
            specmean = np.nanmean(specslice, axis=1)
            specmean = np.nanmean(specmean, axis=1)
            specmedian = np.nanmean(specslice, axis=1)
            specmedian = np.nanmean(specmean, axis=1)
            
        elif self.circle:
            whgood = circle_spec(self.cubedata, circle[0],circle[1],\
                                 circle[2], annulus = circle[3])
            flatfull = self.cubedata.reshape(self.cubedata.shape[0],-1)
            whgoodflat = whgood.flatten()
            specslice = flatfull[:,whgoodflat]
                        
            specmean = []
            for i in range(specslice.shape[0]):
                sl = specslice[i,:]
                
                if False not in np.isnan(sl):
                    specmean.append(1.0)
                    continue
                    
                nans = np.isnan(sl)
                sl[nans] = -1
                gd = sl > 0
                #sigclip = stats.sigmaclip(sl[gd], low = 15, high = 10)[0]
                specmean.append(np.mean(sl[gd]))
            
            specmean = np.array(specmean)
            #specmedian = np.nanmedian(specslice,axis=1)
            #specmean = np.nanmean(specmean, axis=1)
        elif self.ellipse:
            whgood = ellipse_region(self.cubedata, ellipse[0], ellipse[1],\
                        ellipse[2], ellipse[3], ellipse[4], \
                        annulus = ellipse[4])
            flatdata = self.cubedata.reshape(self.cubedata.shape[0],-1)
            whgoodflat = whgood.flatten()
            specslice = flatfull[:,whgoodflat]
                        
            specmean = []
            for i in range(specslice.shape[0]):
                sl = specslice[i,:]
                
                if False not in np.isnan(sl):
                    specmean.append(1.0)
                    continue
                
                #masking and excluding NaNs
                nans = np.isnan(sl)
                sl[nans] = -1
                gd = sl > 0
                #sigclip = stats.sigmaclip(sl[gd], low = 15, high = 10)[0]
                specmean.append(np.mean(sl[gd]))
            
            specmean = np.array(specmean)
            #specmedian = np.nanmedian(specslice,axis=1)
            #specmean = np.nanmean(specmean, axis=1)
            
        else:
            specslice = self.cubedata
            specmean = np.nanmean(specslice, axis=1)
            specmean = np.nanmean(specmean, axis=1)
            specmedian = np.nanmean(specslice, axis=1)
            specmedian = np.nanmean(specmean, axis=1)

        self.spectrum = specmean
        self.extracted = True

    def plotSpectra(self):

        if self.extracted:
            fig, axes = mpl.subplots(figsize = (15,10))
            axes.plot(self.cubewl, self.spectrum, label='Mean')
            axes.minorticks_on()
            mpl.show()
        else:
            print("Spectrum not extracted yet")
    
    def plotImage(self, subimage = False):
        '''Produces a plot of the cube image with the defined regions overlaid. Axes should be 
        in celestial coordinates.
        
        Optional parameters:
        
        subimage:   A matplotlib gridspec.GridSpec image element. Use this to make the plot as
                    one axis of a multi-axis image.'''
        
        if not subimage:
            cubewcs = wcs.WCS(self.cubehead)

            fig = mpl.figure(figsize = (12,10))
            gs = gridspec.GridSpec(1,1)

            ax1 = mpl.subplot(gs[0,0], projection = cubewcs)

            norm = ImageNormalize(self.cubeim, interval=ZScaleInterval())
            ax1.imshow(self.cubeim, interpolation = None, origin='lower',norm=norm, \
                          cmap='Greys')

            if self.limits:
                rect = patches.Rectangle((self.limits[0], self.limits[1]), self.limits[2],\
                    linewidth=2, edgecolor='r',facecolor='none')
                ax1.add_patch(rect)
            if self.circle:
                if self.circle[3]:
                    circ = patches.Circle([self.circle[1],self.circle[0]],\
                        radius=self.circle[2], linewidth=2, edgecolor='m',facecolor='none')
                    ax1.add_patch(circ)
                    circ = patches.Circle([self.circle[1],self.circle[0]],\
                        radius=self.circle[3], linewidth=2, edgecolor='r',facecolor='none')
                    ax1.add_patch(circ)
                else:
                    circ = patches.Circle([self.circle[1],self.circle[0]],\
                        radius=self.circle[2], linewidth=2, edgecolor='r',facecolor='none')
                    ax1.add_patch(circ)
            if self.ellipse:
                # [x0,y0,theta,a,e, annulus]
                if self.ellipse[5]:
                    an = Angle(ellipse[2], 'deg')
                    b = ellipse[3] * ellipse[4]
                    
                    el1 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[3], 2*b, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='m', facecolor='none')
                    ax1.add_patch(el1)
                    
                    b2 = ellipse[5] * ellipse[4]
                    el2 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[5], 2*b2, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='m',facecolor='none')
                    ax1.add_patch(cl2)
                else:
                    an = Angle(ellipse[2], 'deg')
                    b = ellipse[3] * ellipse[4]
                    
                    el1 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[3], 2*b, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='m', facecolor='none')
                    ax1.add_patch(el1)

            ax1.grid('both', color='g', alpha=0.5)

            lon, lat = ax1.coords
            lon.set_ticks(spacing= 5 * u.arcsec, size = 5)
            lon.set_ticklabel(size = 13)
            lon.set_ticks_position('lbtr')
            lon.set_ticklabel_position('lb')
            lat.set_ticks(spacing= 10 * u.arcsec, size = 5)
            lat.set_ticklabel(size = 13)
            lat.set_ticks_position('lbtr')
            lat.set_ticklabel_position('lb')
            lat.display_minor_ticks(True)
            lon.display_minor_ticks(True)

            mpl.show() 
        
        else:
            cubewcs = wcs.WCS(self.cubehead)

            self.plotax = mpl.subplot(subimage, projection = cubewcs)

            norm = ImageNormalize(self.cubeim, interval=ZScaleInterval())
            self.plotax.imshow(self.cubeim, interpolation = None, origin='lower',norm=norm, \
                          cmap='Greys')

            if self.limits:
                rect = patches.Rectangle((self.limits[0], self.limits[1]), self.limits[2],\
                    linewidth=2, edgecolor='r',facecolor='none')
                self.plotax.add_patch(rect)
            if self.circle:
                if self.circle[3]:
                    circ = patches.Circle([self.circle[1],self.circle[0]],\
                        radius=self.circle[2], linewidth=2, edgecolor='m',facecolor='none')
                    self.plotax.add_patch(circ)
                    circ = patches.Circle([self.circle[1],self.circle[0]],\
                        radius=self.circle[3], linewidth=2, edgecolor='r',facecolor='none')
                    self.plotax.add_patch(circ)
                else:
                    circ = patches.Circle([self.circle[1],self.circle[0]],\
                        radius=self.circle[2], linewidth=2, edgecolor='r',facecolor='none')
                    self.plotax.add_patch(circ)
            if self.ellipse:
                # [x0,y0,theta,a,e, annulus]
                if self.ellipse[5]:
                    an = Angle(ellipse[2], 'deg')
                    b = ellipse[3] * ellipse[4]
                    
                    el1 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[3], 2*b, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='m', facecolor='none')
                    self.plotax.add_patch(el1)
                    
                    b2 = ellipse[5] * ellipse[4]
                    el2 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[5], 2*b2, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='m',facecolor='none')
                    self.plotax.add_patch(cl2)
                else:
                    an = Angle(ellipse[2], 'deg')
                    b = ellipse[3] * ellipse[4]
                    
                    el1 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[3], 2*b, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='m', facecolor='none')
                    self.plotax.add_patch(el1)

            self.plotax.grid('both', color='g', alpha=0.5)

            lon, lat = self.plotax.coords
            lon.set_ticks(spacing= 5 * u.arcsec, size = 5)
            lon.set_ticklabel(size = 13)
            lon.set_ticks_position('lbtr')
            lon.set_ticklabel_position('lb')
            lat.set_ticks(spacing= 10 * u.arcsec, size = 5)
            lat.set_ticklabel(size = 13)
            lat.set_ticks_position('lbtr')
            lat.set_ticklabel_position('lb')
            lat.display_minor_ticks(True)
            lon.display_minor_ticks(True)
    
##############################################################
class WIFISTelluric():
    '''Class that hanles the telluric reduction procedure for WIFIS spectra.'''

    def __init__(self, target, telluric):
        '''Inputs for the class are:
        galfile:        path to galaxy datacube
        tellfile:       path to telluric star datacube
        galimfile:      path to image file for galaxy datacube
                        Usually is the same path as datacube except has cubeImg in the datacube name
        tellimfile:     path to telluric file for telluric datacube
        z:              an estimate of the galaxy redshift'''

        self.target = target
        self.telluric = telluric 

        galimf = fits.open(self.galfile[:-7]+'Img_1.fits')
        #galimf = fits.open(self.galfile.split('_')[0]+'_obs_cubeImg.fits')
        self.galim = galimf[0].data
        self.galimhdr = galimf[0].header

        tellimf = fits.open(self.tellfile[:-7]+'Img_1.fits')
        #tellimf = fits.open(self.tellfile.split('_')[0]+'_obs_cubeImg.fits')
        self.tellim = tellimf[0].data
        self.tellimhdr = tellimf[0].header

        self.tlimits = False #Limits of a rectangle (x0,y0) [bottom left], (x1,y1) [top right] in the form [x0,x1,y0,y1]
        self.tcircle = False #Limits of a circle with center (x0,y0) in the form [x0,y0,radius]
        self.glimits = False
        self.gcircle = False
        
        #Setting standard telluric fit adjustment values
        self.tscale = 1.0 #Sets the scale of the vega spectrum to fit to the telluric star
        self.tshift = 0.0 #Sets the wl shift of the vega spectrum relative to the telluric spectrum
        self.gscale = 1.0 #Sets the scale of the telluric spectrum to fit to the target
        self.gshift = 0.0 #Sets the scale of the telluirc spectrum to fit to the target

        self.textracted = False
        self.gextracted = False
        self.reducedspectrum = False
        self.uncertainties = False
        self.telluricload = False
                
    def get_uncertainties(self):
        '''Function that estimates uncertainties using the standard error of the mean of the
        set of individual observations. Currently only works on science target frames. This method
        should likely be used only when the number of individual observations are 10+'''

        tartype = 'gal'
        
        #Checking to see if summed galaxy is extracted
        if not self.gextracted:
            print("Need full galaxy extraction to derive uncertainties...")
            return
        
        #Getting the filepaths
        tarbase = self.galfile.split('/')[:-1]
        self.tarbase = '/'.join(tarbase) + '/'
        tarfls = glob(self.tarbase + '*_obs_cube.fits')

        #Extracting the spectra
        masterarr = []
        for i,fl in enumerate(tarfls):
            wl, spec, head = self.extract_spectrum(fl, tartype)

            #Interpolating the spectra onto the same wavelength grid as median galaxy
            galinterp = np.interp(self.galwl, wl, spec, left = 0, right = 0)
            masterarr.append(galinterp)


        #Calculating the uncertainties
        masterarr = np.array(masterarr)
        errarr = np.std(masterarr, axis = 0) / np.sqrt(masterarr.shape[0])

        #Setting the class values
        self.galerr = errarr
        self.uncertainties = True
        
    def extract_telluric(self):
        '''Extracts the telluric spectrum using the specified aperture'''

        self.tellwl, self.tellspec,\
            self.tellheader = self.extract_spectrum(self.tellfile,'telluric')
        self.textracted = True

    def extract_galaxy(self):
        '''Extracts the science spectrum using the specified aperture'''

        self.galwl, self.galspec,\
            self.galheader = self.extract_spectrum(self.galfile,'gal')
        self.gextracted = True

    def extract_both(self):
        '''Convenience function for extracting both telluric and science spectra'''

        self.extract_galaxy()
        self.extract_telluric()
    
    def plotSpectra(self, kind='both'):

        if kind == 'both':
            fig, axes = mpl.subplots(2,1, figsize = (15,10))
            axes[0].plot(self.tellwl, self.tellspec,label='Mean')
            axes[1].plot(self.galwl, self.galspec, label='Mean')

            axes[0].minorticks_on()
            axes[1].minorticks_on()
            mpl.show()
        elif kind == 'telluirc':
            fig, axes = mpl.subplots(figsize = (15,10))
            axes.plot(self.tellwl, self.tellspec,label='Mean')
            axes.minorticks_on()
            mpl.show()
        elif kind == 'galaxy':
            fig, axes = mpl.subplots(figsize = (15,10))
            axes.plot(self.tellwl, self.tellspec,label='Mean')
            axes.minorticks_on()
            mpl.show()

    def open_reduced_telluric(self):
        if os.path.isfile(self.tellfile[:-5]+'_fullreduce.fits'):
            ffile = fits.open(self.tellfile[:-5]+'_fullreduce.fits')
            self.tellwl = ffile[1].data
            self.TellSpecReduced = ffile[0].data
            self.tellspec = ffile[0].data
            self.telluricload = True
        else:
            print("Reduced telluric file doesn't exist...")
            return

    def write_reduced_spectrum(self, suffix='', kind = 'Galaxy'):
        '''Function that writes the final reduced spectrum to file. Must have created a reduced spectrum first.
        
        If there is no reduced spectrum an extracted spectrum will be written instead.
        
        The first extension is the spectrum, second is the wavelength array, third (if calculated) is the 
        uncertainties'''
        if kind == 'Telluric':
            try:
                hdu = fits.PrimaryHDU(self.TellSpecReduced)
                hdu2 = fits.ImageHDU(self.tellwl, name = 'WL')
                hdul = fits.HDUList([hdu,hdu2])
                hdul.writeto(self.tellfile[:-5]+'_fullreduce.fits', overwrite=True)
            except:
                print("Something went wrong with Telluric saving...")
                return
        else:
            if self.reducedspectrum:
                print("Writing reduced final spectrum....")
                hdu = fits.PrimaryHDU(self.FinalSpec)
            elif self.gextracted:
                print("No reduced spectrum, writing extracted spectrum....")
                hdu = fits.PrimaryHDU(self.galspec)
            else:
                print("Science spectrum not extracted...returning")
                return

            hdu2 = fits.ImageHDU(self.galwl, name = 'WL')
            if self.uncertainties:
                hdu3 = fits.ImageHDU(self.galerr, name = 'ERR')
                hdul = fits.HDUList([hdu,hdu2,hdu3])
            else:
                print('NO UNCERTAINTIES CALCULATED...NOT INCLUDING IN FITS')
                hdul = fits.HDUList([hdu,hdu2])

            hdul.writeto(self.galfile[:-5]+'_extracted_'+suffix+'.fits', overwrite=True)
            print("Wrote to "+self.galfile[:-5]+'_extracted_'+suffix+'.fits')

    def extract_spectrum(self, fl, spectype):
        '''Function that extracts a telluric or science spectrum in the aperture provided.

        Inputs:
            fl:         datacube file
            spectype:   'gal' (science target) or 'telluric' 
        '''

        #Determined datacube type
        if spectype == 'gal':
            limits = self.glimits
            circle = self.gcircle
        elif spectype == 'telluric':
            limits = self.tlimits
            circle = self.tcircle

        if (limits == None) and (circle == None):
            print("Extraction limits not set. Please set the relevant circle or limits variables to extract the spectra")
            return

        #Opening the datacube, determining wl array
        f = fits.open(fl)
        full = f[0].data

        header = f[0].header
        pixel = np.arange(full.shape[0]) + 1.0
        wlval = pixel*header['CDELT3'] + header['CRVAL3']
        wlval *= 1e10

        #Slicing telluric star, then taking the mean along the spatial axes.
        if limits:
            specslice = full[:,limits[0]:limits[1],limits[2]:limits[3]]
            specmean = np.nanmean(specslice, axis=1)
            specmean = np.nanmean(specmean, axis=1)
            specmedian = np.nanmean(specslice, axis=1)
            specmedian = np.nanmean(specmean, axis=1)
            
        elif circle:
            whgood = circle_spec(full, circle[0],circle[1],circle[2], annulus = circle[3])
            flatfull = full.reshape(full.shape[0],-1)
            whgoodflat = whgood.flatten()
            specslice = flatfull[:,whgoodflat]
                        
            specmean = []
            for i in range(specslice.shape[0]):
                sl = specslice[i,:]
                
                if False not in np.isnan(sl):
                    specmean.append(1.0)
                    continue
                    
                nans = np.isnan(sl)
                sl[nans] = -1
                gd = sl > 0
                #sigclip = stats.sigmaclip(sl[gd], low = 15, high = 10)[0]
                specmean.append(np.mean(sl[gd]))
            
            specmean = np.array(specmean)
            #specmedian = np.nanmedian(specslice,axis=1)
            #specmean = np.nanmean(specmean, axis=1)
            
        else:
            specslice = full
            specmean = np.nanmean(specslice, axis=1)
            specmean = np.nanmean(specmean, axis=1)
            specmedian = np.nanmean(specslice, axis=1)
            specmedian = np.nanmean(specmean, axis=1)

        return wlval, specmean, header

    def interactive_vega(self, vega_con_Interp, kind = 'Telluric'):
        '''Function that interactively allows for the fitting of either a telluric star and the vega spectrum
        or the telluric spectrum and the science target spectrum.'''

        mpl.ion()
        if kind == 'Telluric':
            target = 'Telluric'
            calib = 'Vega'
            wl = self.tellwl
            tstar = self.tellspec
            shift = self.tshift
            scale = self.tscale
        else:
            target = 'Galaxy'
            calib = 'Telluric'
            wl = self.galwl
            tstar = self.galspec
            shift = self.gshift
            scale = self.gscale

        vegainterp = vega_con_Interp(wl + shift) ** scale
        TellSpec = tstar / (vegainterp/np.nanmedian(vegainterp))

        #Creating the initial plot
        fig, axes = mpl.subplots(2, figsize = (14,8),sharex=True)
        axes[0].plot(wl, tstar/np.nanmedian(tstar), 'r', label='Target')
        axes[0].plot(wl, vegainterp/np.nanmedian(vegainterp), 'b', label='Calibration')
        axes[1].plot(wl, TellSpec, 'g', label='Target')

        axes[0].legend(loc='best')
        axes[1].legend(loc='best')
        axes[0].set_title("No shift or scale")

        fig.canvas.draw()

        #Beginning the iteration
        status1 = True
        status2 = True
        print("Enter N to quit the interactive mode")
        axes[0].set_xlim((8500,13400))
        axes[1].set_xlim((8500,13400))
        #axes[1].set_ylim((0.,3))
        while (status1 != 'N') and (status2 != 'N'):
            status1 = input('Please enter the shift value (%s or N): ' %(str(shift)))
            if status1 == 'N':
                continue
            status2 = input('Please enter the scale value (%s or N): ' %(str(scale)))
            if status2 == 'N':
                continue

            if status1 != '':
                try:
                    shift = float(status1)
                except:
                    print("Input on shift bad, try again")
                    continue
            if status2 != '':
                try:
                    scale = float(status2)
                except:
                    print("Input on scale bad, try again")
                    continue
            
            axes[0].clear()
            axes[1].clear()

            axes[0].tick_params(direction = 'inout', top = True, right = True)
            axes[1].tick_params(direction = 'inout', top = True, right = True)

            vegainterp = vega_con_Interp(wl + shift) ** scale
            TellSpec = tstar / (vegainterp / np.nanmedian(vegainterp))

            axes[0].plot(wl, tstar/np.nanmedian(tstar), 'r', label=target)
            axes[0].plot(wl, vegainterp/np.nanmedian(vegainterp), 'b', label=calib)
            axes[1].plot(wl, TellSpec, 'g', label=target + ' ' + 'Reduced')

            axes[0].legend(loc='best')
            axes[1].legend(loc='best')
            axes[1].set_xlabel('Wavelength ($\AA$)')
            axes[1].set_ylabel('Relative Flux')
            axes[0].set_ylabel('Relative Flux')
            #axes[1].set_ylim((0,3))
            
            mpl.subplots_adjust(wspace=0, hspace=0)
            fig.canvas.draw()

        mpl.close('all')
        mpl.ioff()

        if kind == 'Telluric':
            self.tscale = scale
            self.tshift = shift
        else:
            self.gscale = scale
            self.gshift = shift
            
    def shiftScale(self, vega_con_Interp, kind = 'Telluric'):
        '''Function that interactively allows for the fitting of either a telluric star and the vega spectrum
        or the telluric spectrum and the science target spectrum.'''

        if kind == 'Telluric':
            target = 'Telluric'
            calib = 'Vega'
            wl = self.tellwl
            tstar = self.tellspec
            shift = self.tshift
            scale = self.tscale
        else:
            target = 'Galaxy'
            calib = 'Telluric'
            wl = self.galwl
            tstar = self.galspec
            shift = self.gshift
            scale = self.gscale

        vegainterp = vega_con_Interp(wl + shift) ** scale
        TellSpec = tstar / (vegainterp/np.nanmedian(vegainterp))

        #Creating the initial plot
        fig, axes = mpl.subplots(2, figsize = (14,8),sharex=True)
        axes[0].plot(wl, tstar/np.nanmedian(tstar), 'r', label='Target')
        axes[0].plot(wl, vegainterp/np.nanmedian(vegainterp), 'b', label='Calibration')
        axes[1].plot(wl, TellSpec, 'g', label='Target')

        axes[0].legend(loc='best')
        axes[1].legend(loc='best')
        axes[0].set_title("No shift or scale")

        axes[0].set_xlim((8500,13400))
        axes[1].set_xlim((8500,13400))
            
        axes[0].tick_params(direction = 'inout', top = True, right = True)
        axes[1].tick_params(direction = 'inout', top = True, right = True)

        axes[0].legend(loc='best')
        axes[1].legend(loc='best')
        axes[1].set_xlabel('Wavelength ($\AA$)')
        axes[1].set_ylabel('Relative Flux')
        axes[0].set_ylabel('Relative Flux')
            
        mpl.subplots_adjust(wspace=0, hspace=0)

    def remove_features(self, target, confirm=True, kind='galaxy',\
                       inspect = True):

        #Creating the initial plot
        if kind == 'telluric':
            wl = np.array(self.tellwl)
            spec = np.array(self.TellSpecReduced)
        elif kind == 'galaxy_z':
            wl = np.array(self.galwl / (1. + self.z))
            spec = np.array(self.FinalSpec)
        else:
            wl = np.array(self.galwl)
            spec = np.array(self.FinalSpec)
            
        for vals in target:
            startval = vals[0]
            endval = vals[1]
            
            wh = np.where((wl >= startval) & (wl <= endval))[0]
            whfull = np.where((wl >= (startval - 50)) & (wl <= (endval + 50)))[0]
            
            if len(wh) == 0:
                print("Spectral region, ",vals," doesn't exist, try again")
                continue
            
            fig, axis = mpl.subplots(figsize = (15,10))
            axis.plot(wl[whfull], spec[whfull],'r')
            
                
            start = wl[wh[0]-1]
            end = wl[wh[-1]+1]
            print(start,end, spec[wh][0],spec[wh][-1])
            
            pf = np.polyfit([start,end],[spec[wh][0],spec[wh][-1]], 1)
            contfit = np.poly1d(pf)
            spec[wh] = contfit(wl[wh])
                                         
            axis.plot(wl[whfull], spec[whfull],'b')
            
            mpl.show()
            
            if confirm:
                c = input("Do you want to keep this change? (Y/N): ")
                if c == 'Y':
                    if kind == 'telluric':
                        print("Saving new Telluric spectrum...")
                        self.TellSpecReduced = spec 
                    else:
                        print("Saving new Galaxy spectrum...")
                        self.FinalSpec = spec
                else:
                    continue
            else:
                if kind == 'telluric':
                    print("Saving new Telluric spectrum...")
                    self.TellSpecReduced = spec 
                else:
                    print("Saving new Galaxy spectrum...")
                    self.FinalSpec = spec

    def telluricAdjust(self, region, scale, shift=0.0):
        
        start = region[0]
        end = region[1]

        wh = np.where((self.galwl >= start) & (self.galwl <= end))[0]

        wlslice = self.galwl[wh]
        tellslice = self.TellInterp[wh]
        galslice = self.galspec[wh]

        fig, axes = mpl.subplots(2,1, figsize = (15,10))
        axes[0].plot(wlslice, norm(tellslice), 'r')
        axes[0].plot(wlslice, norm(galslice), 'b')
        axes[1].plot(wlslice, galslice/norm(tellslice**scale),'k')
        mpl.show()
    
    def do_telluric_reduction(self, hlinemode = 'measure',\
                    interactivetelluric = False, plot = True, profile='lorentzian',\
                             telluricmask = [], hlineplot=False):
        '''Function that reduces the science spectrum with a telluric spectrum. 
        First removes h-lines from the telluric spectrum with either a vega spectrum from file,
        or directly fitting the lines. Has interactive modes to manually stretch and shift
        the fitting spectra to match the target spectra. Then corrects the galaxy spectrum by
        the telluric spectrum.'''

        #Check if the relevant spectra are extracted.
        if not self.textracted:
            print("Telluric spectrum not extracted...attempting")
            self.extract_telluric()
        if not self.gextracted:
            print("Science spectrum not extracted...attempting")
            self.extract_galaxy()

        if not self.telluricload: #If a telluric is not already reduced and loaded
            ##Loading convolved Vega
            vega_con = pd.read_csv('/Users/relliotmeyer/WIFIS/TellCorSpec/vega-con-new.txt',\
                                   sep = ' ', header = None)
            vega_con.columns = ['a', 'b']   #Col 1 = wavelength, col 2 = FLux
            vegawl = vega_con['a']
            vegadata = vega_con['b']
            
            #Creating interpolator for Vega spectrum
            if hlinemode == 'measure':
                poly = self.measure_hlines(fittype = 'normal', plot = hlineplot, profile=profile)
                vegawlnew = vegawl + poly(vegawl)
                #vega_Interp = interp1d(vegawl, vegadata, kind='cubic',\
                #                 bounds_error = False)

                vega_con_Interp = interp1d(vegawlnew, vegadata, kind='cubic',\
                                           bounds_error = False)
                
                vegainterp = vega_con_Interp(self.tellwl + self.tshift) ** self.tscale
                
                #Create final vega spectrum using interpolator, then adjust telluric spectrum
                self.TellSpecReduced = self.tellspec / (vegainterp / np.median(vegainterp)) 
                print("SHIFT AND SCALE FOR VEGA IS: ", self.tshift, self.tscale)
                
                if plot:
                    fig, axis = mpl.subplots(2,1,figsize=(15,10), sharex=True)
                    axis[0].plot(self.tellwl, norm(self.tellspec),'b', label='Standard Star Spectrum')
                    axis[0].plot(self.tellwl, norm(vegainterp),'r', label='Vega Spectrum')
                    axis[1].plot(self.tellwl, norm(self.TellSpecReduced),'k', label='Telluric Spectrum')
                    axis[0].set_ylabel('Relative Flux', fontsize = 17)
                    axis[1].set_ylabel('Relative Flux', fontsize = 17)
                    axis[1].set_xlabel(r'Wavelength (\AA)', fontsize = 17)
                    axis[0].tick_params(axis='both', which='major', labelsize=13)
                    axis[1].tick_params(axis='both', which='major', labelsize=13)
                    midguess = np.array([8865, 9017, 9232, 9550, 10052, 10941, 12822])
                    for midwl in midguess:
                        axis[0].axvline(midwl, linestyle='--', color='gray')
                        axis[1].axvline(midwl, linestyle='--', color='gray')
                    
                    axis[1].legend(fontsize = 15)
                    axis[0].legend(fontsize = 15)

                    mpl.minorticks_on()
                    mpl.subplots_adjust(wspace=0, hspace=0)
                    mpl.savefig('/Users/relliotmeyer/Desktop/VegaCorrection.pdf', dpi=500)
                    mpl.show()
                    
            
            elif hlinemode == 'none':
                #poly = self.measure_hlines(fittype = 'normal', plot = True, profile=profile)
                #vegawlnew = vegawl + poly(vegawl)
                #vega_Interp = interp1d(vegawl, vegadata, kind='cubic',\
                #                 bounds_error = False)

                vega_con_Interp = interp1d(vegawl, vegadata, kind='cubic',\
                                           bounds_error = False)
                
                vegainterp = vega_con_Interp(self.tellwl + self.tshift) ** self.tscale
                
                #Create final vega spectrum using interpolator, then adjust telluric spectrum
                self.TellSpecReduced = self.tellspec / norm(vegainterp) 
                
                print("SHIFT AND SCALE FOR VEGA IS: ", self.tshift, self.tscale)
                
                if plot:
                    fig, axis = mpl.subplots(2,1,figsize=(15,10))
                    axis[0].plot(self.tellwl, norm(self.tellspec),'b')
                    axis[0].plot(self.tellwl, norm(vegainterp),'r')
                    axis[1].plot(self.tellwl, self.TellSpecReduced)
                    mpl.show()

            elif hlinemode == 'interactive':
                #If interactive then enter interactive fitting mode
                vega_con_Interp = interp1d(vegawl, vegadata, kind='cubic', bounds_error = False)

                self.interactive_vega(vega_con_Interp)
                vegainterp = vega_con_Interp(self.tellwl+self.tshift) ** self.tscale
                #self.shiftScale(vega_con_Interp)

                #Create final vega spectrum using interpolator, then adjust telluric spectrum
                self.TellSpecReduced = self.tellspec / (vegainterp / np.median(vegainterp)) 
                print("SHIFT AND SCALE FOR VEGA IS: ", self.tshift, self.tscale)
                
            elif hlinemode == 'remove':
                newtelluric = self.measure_hlines(fittype = 'normal', plot=False,\
                                                               remove=True, profile=profile)
                if plot:
                    fig, axes = mpl.subplots(figsize = (15,10))
                    axes.plot(self.tellwl, self.tellspec,'b')
                    axes.plot(self.tellwl, newtelluric,'r')
                    mpl.show()
                
                self.TellSpecReduced = newtelluric
            
            elif hlinemode == 'broaden':
                poly = self.measure_hlines(fittype = 'normal', plot = True, profile=profile)
                vegawlnew = vegawl + poly(vegawl)

                #vega_con_Interp = interp1d(vegawl, vegadata, kind='cubic',\
                #               bounds_error = False)
                
                #vegainterp = vega_con_Interp(self.tellwl) ** self.tscale
                
                xs = np.arange(-50,51, vegawl[5]-vegawl[4])
                #vp = gf.voigtfullnorm(xs, 0.5115, -3.0)
                vp = gf.voigtfullnorm(xs, 15, 0.005)
                out = np.convolve(vegadata, vp, mode='same')
                #out = vegadata
                vega_con_Interp = interp1d(vegawlnew, out, kind='cubic', bounds_error = False)
                vegainterp = vega_con_Interp(self.tellwl+self.tshift) ** self.tscale
                self.TellSpecReduced = self.tellspec / norm(vegainterp) 

                if plot:
                    fig, axis = mpl.subplots(2,1,figsize=(15,10))
                    axis[0].plot(self.tellwl, norm(self.tellspec),'b')
                    axis[0].plot(self.tellwl, norm(vegainterp),'r')
                    axis[1].plot(self.tellwl, self.TellSpecReduced)
                    axes.tick_params(which='minor')
                #axis.plot(vegawl, norm(vegadata), 'b')
                #axis.plot(vegawl, norm(out),'r')
                    mpl.show()
                return
                
            self.write_reduced_spectrum(kind = 'Telluric')

        #Create interpolator of telluric spectrum (using non-NaN values)
        notnan = ~np.isnan(self.TellSpecReduced)
        TelStar_Interp = interp1d(self.tellwl[notnan], self.TellSpecReduced[notnan], \
                                  kind='cubic', bounds_error=False)  

        #If interactive then enter interactive fitting mode
        if interactivetelluric:
            #self.interactive_vega(TelStar_Interp, kind = 'Galaxy')
            self.shiftScale(TelStar_Interp, kind = 'Galaxy')

        print("SHIFT AND SCALE FOR TELLURIC IS: ", self.gshift, self.gscale)
        
        #Create final telluric spectrum using interpolator, then adjust science spectrum
        self.TellInterp = TelStar_Interp(self.galwl+self.gshift) ** self.gscale
        normtell = self.TellInterp/np.nanmedian(self.TellInterp)
        
        if len(telluricmask) > 0:
            for i in range(len(telluricmask)):
                whgd = np.where((self.galwl >= telluricmask[i][0]) & (self.galwl <= telluricmask[i][1]))[0]
                pf = np.polyfit([telluricmask[i][0],telluricmask[i][1]], \
                                [normtell[whgd][0],normtell[whgd][-1]], 1)
                contfit = np.poly1d(pf)
                cont = contfit(self.galwl[whgd])
                normtell[whgd] = cont

        self.FinalSpec = self.galspec / normtell

        self.reducedspectrum = True

        wlval = self.galwl
        wlvalz = self.galwl / (1 + self.z)

        #Plot resulting spectrum
        if plot == True:

            regions = [(9400,9700),(10300,10500),(11000,11500),(11500,11900),\
                       (12200,12500),(12600,13000)]
            for i, region in enumerate(regions):
                fig, axes = mpl.subplots(2,1,figsize = (15,10), sharex=True)
                whreg = (wlvalz >= region[0]) & (wlvalz <= region[1])
                
                axes[0].plot(wlvalz[whreg],norm(normtell[whreg]), 'r')
                axes[0].plot(wlvalz[whreg],norm(self.galspec[whreg]),'b')
                
                axes[1].plot(wlvalz[whreg],norm(self.FinalSpec[whreg]), 'k')
                mpl.tight_layout()
                mpl.minorticks_on()
                mpl.grid(axis='x', which='both')

                mpl.subplots_adjust(wspace=0, hspace=0)
                mpl.show()
            
            wlval = wlval / (1+self.z)
            
            fig, ax = mpl.subplots(figsize = (15,10))
        
            nonnan = ~np.isnan(self.FinalSpec)
            ax.plot(wlval[nonnan][50:-20], norm(self.FinalSpec[nonnan][50:-20]), 'k')
            
            ax.set_title("Reduced and de-Redshifted Spectrum")
            ax.set_xlabel("Wavelength ($\AA$)", fontsize=13)
            ax.set_ylabel("Flux", fontsize = 13)

            ax.xaxis.set_minor_locator(AutoMinorLocator())

            mpl.show()

    def measure_hlines(self, fittype='quadratic', plot=True, remove=False, profile='lorentzian'):
        '''TESTING FUNCTION: To determine the wl offset between the vega and telluric spectrum'''


        #hlinelow =  [8840,8960,9190,9520,10000,10880,12775]
        #hlinehigh = [8925,9050,9275,9600,10120,11020,12880]
        hlinelow =  [8840,8960,9190,9520,9900,10880,12720]
        hlinehigh = [8925,9050,9275,9600,10120,11020,12900]

        midguess = np.array([8865, 9017, 9232, 9550, 10052, 10941, 12822])
        midwl = [np.mean([hlinelow[i],hlinehigh[i]]) for i in range(len(hlinelow))]

        ##Loading convolved Vega
        vega_con = pd.read_csv('/Users/relliotmeyer/WIFIS/TellCorSpec/vega-con-new.txt', sep = ' ', header = None)
        vega_con.columns = ['a', 'b']   #Col 1 = wavelength, col 2 = FLux
        vegawl = vega_con['a']
        vegadata = vega_con['b']
        
        telluriccopy = np.array(self.tellspec) 

        diffs = []
        if plot and fittype == 'normal':
            fig, axes = mpl.subplots(2,4, figsize = (15,7))
            axes = axes.flatten()
        
        for l in range(len(hlinelow)):
            if fittype == 'normal':
                wh = np.where((self.tellwl >= hlinelow[l]) & (self.tellwl <= hlinehigh[l]))[0]
                linewl = self.tellwl[wh]
                linedata = self.tellspec[wh]
                good = np.ones(len(midguess[2:]), dtype=bool)
                
                try:
                    if profile == 'lorentzian':
                        popt,pcov = gf.lorentzian_fit(linewl, linedata, [-150., 30., midguess[l], 7])
                        #print(popt)
                        fitg = gf.lorentz(linewl, popt)
                        midline = popt[2]
                    elif profile == 'voigt':
                        popt,pcov = gf.voigt_fit(linewl, linedata, [midguess[l], -150., 30., 10., 7.])
                        #print(popt)
                        fitg = gf.voigt(linewl, popt)
                        midline = popt[0]

                    if plot:
                        axes[l].plot(linewl, linedata/np.median(linedata),'b:')
                        axes[l].plot(linewl, fitg/np.median(fitg), 'r:')
                        axes[l].axvline(midline, color = 'g', linestyle = ':')
                        #axes[l].plot(linewl, linedata/fitg)

                    vwh = np.where((vegawl >= hlinelow[l]) & (vegawl <= hlinehigh[l]))[0]
                    vwl = vegawl[vwh]
                    vdata = vegadata[vwh]

                    #poptvega,pcov = gf.gaussian_fit_os(linewl,linedata,[-0.5,10.,np.mean([hlinelow[l],hlinehigh[l]]), 7])
                    #if profile == 'lorentzian':
                    #    poptvega,pcov = gf.lorentzian_fit(vwl, vdata, [-150., 30., midguess[l], 7])
                    #    fitvega = gf.lorentz(vwl, popt)
                    #    midline = poptvega[2]
                    #elif profile == 'voigt':
                    
                    try:
                        poptvega,pcov = gf.voigt_fit(vwl, vdata, [midguess[l], -150., 30., 10., 7.])
                    except:
                        poptvega,pcov = gf.voigt_fit(vwl, vdata, [midguess[l], -150., 30., 10., 7.])
                    fitvega = gf.voigt(vwl, poptvega)
                    midline = poptvega[0]


                    if plot:                    
                        axes[l].plot(vwl, 0.2+vdata/np.median(vdata),'b--')
                        axes[l].plot(vwl, 0.2+fitvega/np.median(fitvega), 'r--')
                        axes[4].set_xlabel("Wavelength $(\AA)$")
                        axes[l].axvline(midline, color = 'g', linestyle = '--')

                    if remove:
                        pf = np.polyfit([linewl[0],linewl[-1]], [linedata[0],linedata[-1]], 1)
                        contfit = np.poly1d(pf)
                        cont = contfit(linewl)

                        telluriccopy[wh] = (linedata/fitg) * cont

                    if profile == 'lorentzian':
                        diffs.append(popt[2] - poptvega[2])
                    elif profile == 'voigt':
                        diffs.append(popt[0] - poptvega[0])
                except:
                    print("Couldn't fit line #", l)
                    if l > 1:
                        good[l - 2] = False
                
            elif fittype == 'quadratic':
                wh = np.where((self.tellwl >= hlinelow[l]) & (self.tellwl <= hlinehigh[l]))[0]
                linewl = np.array(self.tellwl[wh])
                linedata = self.tellspec[wh]
                datawl = (linewl[1] - linewl[0])

                minarg = np.argmin(linedata)
                mindata = linedata[minarg]
                minminus = linedata[minarg - 1]
                minplus = linedata[minarg + 1]

                a0 = mindata
                a1 = 0.5 * (minplus - minminus)
                a2 = 0.5 * (minplus + minminus - 2.*mindata)
                cen_val = minarg - (a1 / (2.* a2))
                decimal = cen_val % 1
                wlcen_data = linewl[int(cen_val)] + datawl * decimal

                wh = np.where((vegawl >= hlinelow[l]) & (vegawl <= hlinehigh[l]))[0]
                linewl = np.array(vegawl[wh])
                linedata = np.array(vegadata[wh])
                datawl = linewl[1] - linewl[0]

                minarg = np.argmin(linedata)
                mindata = linedata[minarg]
                minminus = linedata[minarg - 1]
                minplus = linedata[minarg + 1]

                a0 = mindata
                a1 = 0.5 * (minplus - minminus)
                a2 = 0.5 * (minplus + minminus - 2.*mindata)
                cen_val = minarg - (a1 / (2.* a2))
                decimal = cen_val % 1
                wlcen_vega = linewl[int(cen_val)] + datawl * decimal

                print(wlcen_data, wlcen_vega, wlcen_data - wlcen_vega)
                diffs.append(wlcen_data - wlcen_vega)

        pf = np.polyfit(midguess[2:][good], diffs[2:], 2)
        polyfit = np.poly1d(pf)
        cont = polyfit(midwl)

        if plot:
            mpl.show()
            fig, axes = mpl.subplots()
            axes.plot(midwl, diffs)
            axes.plot(midwl, cont)
            axes.set_xlabel("Wavelength $(\AA)$")
            axes.set_ylabel("Offset $(\AA)$")
            mpl.show()

        if remove:
            return telluriccopy
        else:
            return polyfit

    def plotImages(self):
        '''Produces plots of the telluric and galaxy images. Axes should be 
        in celestial coordinates.'''
        
        tellwcs = wcs.WCS(self.tellimhdr)
        galwcs = wcs.WCS(self.galimhdr)

        fig = mpl.figure(figsize = (12,10))
        gs = gridspec.GridSpec(2,1)

        ax1 = mpl.subplot(gs[0,0], projection = tellwcs)
        ax2 = mpl.subplot(gs[1,0], projection = galwcs)
        axes = [ax1,ax2]
                
        norm = ImageNormalize(self.tellim, interval=ZScaleInterval())
        axes[0].imshow(self.tellim,interpolation = None, origin='lower',norm=norm, \
                      cmap='Greys')
        
        norm = ImageNormalize(self.galim, interval=ZScaleInterval())
        axes[1].imshow(self.galim,interpolation = None, origin='lower',norm=norm,\
                      cmap='Greys')
        
        if self.tlimits:
            rect = patches.Rectangle((self.tlimits[0], self.tlimits[1]), self.tlimits[2],\
                linewidth=2, edgecolor='r',facecolor='none')
            axes[0].add_patch(rect)
        if self.glimits:
            rect = patches.Rectangle((self.glimits[0], self.glimits[1]), self.glimits[2],\
                linewidth=2, edgecolor='r',facecolor='none')
            axes[1].add_patch(rect)
        if self.tcircle:
            if self.tcircle[3]:
                circ = patches.Circle([self.tcircle[1],self.tcircle[0]],\
                    radius=self.tcircle[2], linewidth=2, edgecolor='m',facecolor='none')
                axes[0].add_patch(circ)
                circ = patches.Circle([self.tcircle[1],self.tcircle[0]],\
                    radius=self.tcircle[3], linewidth=2, edgecolor='r',facecolor='none')
                axes[0].add_patch(circ)
            else:
                circ = patches.Circle([self.tcircle[1],self.tcircle[0]],\
                    radius=self.tcircle[2], linewidth=2, edgecolor='r',facecolor='none')
                axes[0].add_patch(circ)
        if self.gcircle:
            if self.gcircle[3]:
                circ = patches.Circle([self.gcircle[1],self.gcircle[0]],\
                    radius=self.gcircle[2], linewidth=2, edgecolor='m',facecolor='none')
                axes[1].add_patch(circ)
                circ = patches.Circle([self.gcircle[1],self.gcircle[0]],\
                    radius=self.gcircle[3], linewidth=2, edgecolor='r',facecolor='none')
                axes[1].add_patch(circ)
            else:
                circ = patches.Circle([self.gcircle[1],self.gcircle[0]],\
                    radius=self.gcircle[2], linewidth=2, edgecolor='r',facecolor='none')
                axes[1].add_patch(circ)
            
        axes[0].grid('both', color='g', alpha=0.5)
        axes[1].grid('both', color='g', alpha=0.5)
        
        lon, lat = axes[0].coords
        lon.set_ticks(spacing=5 * u.arcsec, size = 5)
        lon.set_ticklabel(size = 13)
        lon.set_ticks_position('lbtr')
        lon.set_ticklabel_position('lb')
        lat.set_ticks(spacing=10 * u.arcsec, size = 5)
        lat.set_ticklabel(size = 13)
        lat.set_ticks_position('lbtr')
        lat.set_ticklabel_position('lb')
        lat.display_minor_ticks(True)
        lon.display_minor_ticks(True)
        
        lon, lat = axes[1].coords
        lon.set_ticks(spacing=5 * u.arcsec, size = 5)
        lon.set_ticklabel(size = 13)
        lon.set_ticks_position('lbtr')
        lon.set_ticklabel_position('lb')
        lat.set_ticks(spacing=10 * u.arcsec, size = 5)
        lat.set_ticklabel(size = 13)
        lat.set_ticks_position('lbtr')
        lat.set_ticklabel_position('lb')
        lat.display_minor_ticks(True)
        lon.display_minor_ticks(True)
        
        mpl.show()        

        
        
#   def centroid_finder(self, objtype):

#       if objtype == 'target':
#           img = self.galim
#       else:
#           img = self.tellim
#       
#       imgsize = img.shape

#       #find bright pixels
#       imgmedian = np.median(img)
#       #print "MEDIAN: %f, MEAN: %f" % (imgmedian, np.mean(img))
#       imgstd = np.std(img[img < 2000])
#       nstd = 4.0
#       #print "IMG MEAN: %f\nIMGSTD: %f\nCUTOFF: %f" % (imgmedian, imgstd,imgmedian+nstd*imgstd)

#       brightpix = np.where(img >= imgmedian + nstd*imgstd)
#       new_img = np.zeros(imgsize)

#       if len(brightpix) == 0:
#           return False

#       for i in range(len(brightpix[0])):
#           new_img[brightpix[0][i],brightpix[1][i]] = 1.0

#       stars = []
#       for x in range(imgsize[0]):
#           for y in range(imgsize[1]):
#               if new_img[x,y] == 1:
#                   new_star, new_img = explore_region(x,y,new_img)
#                   if len(new_star[0]) >=3: #Check that the star is not just a hot pixel
#                       stars.append(new_star)
#       
#       centroidx, centroidy, Iarr  = [],[],[]
#       for star in stars:
#           xsum, ysum, Isum = 0.,0.,0.
#           sat = False
#           for i in range(len(star[0])):
#               x = star[0][i]
#               y = star[1][i]
#               I = img[x,y]
#               xsum += x*I
#               ysum += y*I
#               Isum += I
#           
#           centroidx.append(xsum/Isum)
#           centroidy.append(ysum/Isum)
#           Iarr.append(Isum)

#           gx0 = centroidx[-1] - 10
#           gx1 = centroidx[-1] + 10
#           gy0 = centroidy[-1] - 10
#           gy1 = centroidy[-1] + 10

#           if centroidx[-1] < 10:
#               gx0 = 0
#           if centroidx[-1] > imgsize[0]-11:
#               gx1 = imgsize[0]-1
#           
#           if centroidy[-1] < 10:
#               gy0 = 0
#           if centroidy[-1] > imgsize[1]-11:
#               gy1 = imgsize[1]-1
#           
#           gx = img[int(gx0):int(gx1),int(centroidy[-1])]
#           gy = img[int(centroidx[-1]), int(gy0):int(gy1)]
#           xs = range(len(gx))
#           ys = range(len(gy))
#       print("Calculated centroids for ", objtype)
#       print(centroidx)
#       print(centroidy)

#       return [centroidx,centroidy,Iarr]