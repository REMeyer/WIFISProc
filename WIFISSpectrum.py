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
from scipy.interpolate import interp1d, interp2d
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
    return spec/np.nanmedian(spec)

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

def ellipse_region(cube, center, a, ecc, theta, annulus = False):
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
            annulus: False if just simple ellipse, otherwise is the INNER
                     annular radius (spaxels)
        
    '''
    
    # Define angle, eccentricity term, and semi-minor axis
    an = Angle(theta, 'deg')
    e = np.sqrt(1 - (ecc**2))
    b = a * e
    
    # Create outer ellipse
    ell_region = Ellipse2D(amplitude=10, x_0 = center[0], y_0 = center[1],\
                a=a, b=b, theta=an.radian)
    x,y = np.mgrid[0:cube.shape[1], 0:cube.shape[2]]
    ell_good = ell_region(x,y)
    
    if annulus:
        # Define inner ellipse parameters and then create inner mask
        a2 = annulus
        b2 = a2 * e

        ell_region_inner = Ellipse2D(amplitude=10, x_0 = center[0],\
                y_0 = center[1], a=a2, b=b2, theta=an.radian)
        
        # Set region of outer mask and within inner ellipse to zero, leaving
        # an annular elliptical region
        ell_inner_good = ell_region_inner(x,y)
        ell_good[ell_inner_good > 0] = 0
        
    #fig, ax = mpl.subplots(figsize = (12,7))
    #mpl.imshow(np.nansum(cube, axis = 0), origin = 'lower', interpolation = None)
    #mpl.imshow(ell_good, alpha=0.2, origin = 'lower', interpolation = None)
    #mpl.show()
    
    # Return the mask
    return np.array(ell_good, dtype = bool)

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
        cubesplit = self.cubefile.split('/')
        self.cubename = ' '.join(cubesplit[-1][:-5].split('_'))
        self.cubebasepath = '/'.join(cubesplit[:-1]) + '/'

        # Extract cube data and header
        cubeff = fits.open(self.cubefile)
        self.cubedata = cubeff[0].data
        self.cubehead = cubeff[0].header
        
        # Create a flux image of the datacube
        self.cubeim = np.nansum(self.cubedata, axis = 0)
        
        # Generate wavelength array from header in Angstroms
        pixel = np.arange(self.cubedata.shape[0]) + 1.0
        self.cubewl = pixel*self.cubehead['CDELT3'] + self.cubehead['CRVAL3']
        self.cubewl *= 1e10
        
        self.z = z
        self.cubewlz = self.cubewl / (1. + self.z)
        
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
        
    def getUncertainties(self, mode = 'Shift', plot=False):
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
        
        #Checking to see if spectrum is extracted
        if not self.extracted:
            print("Need to extract spectrum to derive uncertainties...")
            return
        
        if mode == 'Direct':
            
            # Get cube integration time
            inttime = self.cubehead['INTTIME']
            gain = 1.33            
            #datasqrt = np.sqrt(self.cubedata * inttime * gain)
            datasqrt = self.cubedata * inttime * gain
            datasqrt[np.isnan(datasqrt)] = 0.0
            
            try:
                skyfls = glob(self.cubebasepath + '*_sky_cube.fits')
                skyff = fits.open(skyfls[0])
                skydata = skyff[0].data
                skyhead = skyff[0].header
                skyflat = skydata.reshape(skydata.shape[0],-1)
                #skyflat = np.nanmedian(skydata.reshape(skydata.shape[0],-1),\
                #                       axis = 1)
                skyflat[np.isnan(skyflat)] = 0
                
                pixel = np.arange(skydata.shape[0]) + 1.0
                skywl = pixel*skyhead['CDELT3'] + skyhead['CRVAL3']
                skywl *= 1e10
                
                xx = np.arange(skyflat.shape[1])
                skyinterp = interp2d(xx, skywl, skyflat, kind='cubic', fill_value=0.1)
                skytarget = skyinterp(xx, self.cubewl).reshape(self.cubedata.shape)
                
            except:
                print('### Problem finding and processing sky cubes,'+\
                      ' are they in the same directory as the merged cube?')
                skytarget = np.ones(self.cubedata.shape) * 8.0
            
            #skysqrt = np.sqrt(skytarget * inttime * gain)
            skysqrt = 2 * skytarget * inttime * gain
            skysqrt[np.isnan(skysqrt)] = 0.0
            
            #thermalsqrt = np.sqrt(0.22 * inttime * gain)
            thermalsqrt = 0
            
            #noise = np.sqrt(datasqrt**2.0 + skysqrt**2.0 + thermalsqrt**2.0)
            noise = np.sqrt(datasqrt + skysqrt + thermalsqrt)
            
            self.cubenoise = noise
            self.uncertainties = True
            
            errmean, errmedian, self.cubeerr = self.extractRegion(self.cubenoise, square_errors = True)
            
        elif mode == 'SEM':
            
            #Setting the class values
            self.cubenoise = self.calcErrors()
            self.uncertainties = True
        
        elif mode == 'Shift':
            
            #Setting the class values
            self.cubenoise = self.calcErrors(shift=True)
            self.uncertainties = True
        
        elif mode == 'ShiftMedian':

            #Setting the class values
            self.cubenoise = self.calcErrorsFinalCube()
            self.uncertainties = True
            
        if plot:
            fig, ax = mpl.subplots(figsize = (12,7))
            ax.plot(self.cubewl, self.cubenoise, label=mode)
            ax.set_title('Uncertanties '+ self.cubename, fontsize = 15)
            ax.set_xlabel('Error Value', fontsize = 15)
            ax.set_ylabel("Wavelength (A)", fontsize = 15)
            ax.tick_params(labelsize = 15)
            ax.legend()
            mpl.show()
            
    def extractSpectrum(self, kind = 'mean'):
        '''Function that extracts a telluric or science spectrum in the
        aperture provided.
        '''

        inttime = self.cubehead['INTTIME']
        gain = 1.33
        
        if (self.limits == False) and (self.circle == False) and \
          (self.ellipse == False):
            print("Extraction limits not set. Please set the "+\
                  "relevant ellipse, circle or limits class variables "+\
                  "to extract the spectra")
            return

        #Slicing telluric star, then taking the mean along the spatial axes.
        specmean, specmedian, specsum = self.extractRegion(self.cubedata)

        if kind == 'mean':
            self.spectrum = specmean
            self.spectrummedian = specmedian
            self.spectrumsum = specsum
            self.spectype = 'mean'
        elif kind == 'median':
            self.spectrum = specmedian
            self.spectrumsum = specsum
            self.spectrummean = specmean
            self.spectype = 'median'
        else:
            self.spectrum = specsum
            self.spectrummedian = specmedian
            self.spectrummean = specmean
            self.spectype = 'sum'
        
        self.extracted = True

    def plotSpectrum(self):

        if self.extracted:
            fig, axes = mpl.subplots(figsize = (13,7))
            axes.set_title("Extracted Spectrum "+self.cubename, fontsize=15)
            axes.set_ylabel('Wavelength (A)', fontsize = 15)
            axes.set_xlabel("Flux", fontsize = 15)
            axes.tick_params(labelsize = 15)
            axes.plot(self.cubewl, self.spectrum, label=self.spectype)
            axes.minorticks_on()
            mpl.legend()
            mpl.show()
        else:
            print("Spectrum not extracted yet")
    
    def plotImage(self, subimage = False, title=False, imagecoords=False):
        '''Produces a plot of the cube image with the defined regions overlaid. Axes should be 
        in celestial coordinates.
        
        Optional parameters:
        
        subimage:   A matplotlib gridspec.GridSpec image element. Use this to make the plot as
                    one axis of a multi-axis image.'''
        
        if not subimage:
            cubewcs = wcs.WCS(self.cubehead, naxis=2)

            fig = mpl.figure(figsize = (12,10))
            gs = gridspec.GridSpec(1,1)

            if not imagecoords:
                ax1 = mpl.subplot(gs[0,0], projection = cubewcs)
            else:
                ax1 = mpl.subplot(gs[0,0])

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

            if not imagecoords:
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
            else:
                ax1.tick_params(axis='both', labelsize = 13, top=True, right=True)

            if title:
                ax1.set_title(title)
            
            mpl.show() 
        
        else:
            cubewcs = wcs.WCS(self.cubehead, naxis=2)

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
                e = np.sqrt(1 - (self.ellipse[4]**2))
                an = Angle(self.ellipse[2], 'deg')

                if self.ellipse[5]:
                    b = self.ellipse[3] * e
                    
                    el1 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[3], 2*b, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='m', facecolor='none')
                    self.plotax.add_patch(el1)
                    
                    b2 = self.ellipse[5] * e
                    el2 = patches.Ellipse((self.ellipse[1],self.ellipse[0]),\
                        2*self.ellipse[5], 2*b2, angle = -an.degree + 90, \
                        linewidth=2, edgecolor='r',facecolor='none')
                    self.plotax.add_patch(el2)
                else:
                    an = Angle(self.ellipse[2], 'deg')
                    b = self.ellipse[3] * e
                    
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
            self.plotax.set_xlabel('Decl', fontsize = 15)
            self.plotax.set_ylabel('R.A.', fontsize = 15)
            
    
    def writeSpectrum(self, suffix=''):
        '''Function that writes the final reduced spectrum to file. Must have created a reduced spectrum first.
        
        If there is no reduced spectrum an extracted spectrum will be written instead.
        
        The first extension is the spectrum, second is the wavelength array, third (if calculated) is the 
        uncertainties'''
        
        try:
            if self.extracted:
                print(" writing extracted spectrum....")
                hdu = fits.PrimaryHDU(self.spectrum)
                hdu2 = fits.ImageHDU(self.cubewl, name = 'WL')
                if self.uncertainties:
                    hdu3 = fits.ImageHDU(self.cubeerr, name = 'ERR')
                    hdul = fits.HDUList([hdu,hdu2,hdu3])
                else:
                    print('NO UNCERTAINTIES CALCULATED...NOT INCLUDING IN FITS')
                    hdul = fits.HDUList([hdu,hdu2])
            else:
                print("Science spectrum not extracted...returning")
                return

            hdul.writeto(self.cubefile[:-5]+'_extracted_'+suffix+'.fits', overwrite=True)
            print("Wrote to "+self.cubefile[:-5]+'_extracted_'+suffix+'.fits')
        except:
            print("There was a problem saving the spectrum")
    
    def extractRegion(self, data, inttime = 1, gain = 1, square_errors = False):
        
        if self.limits:
            specslice = data[:,self.limits[0]:self.limits[1],self.limits[2]:self.limits[3]]
            specmean = np.nanmean(specslice, axis=(1,2)) #* inttime * gain
            specmedian = np.nanmedian(specslice, axis=(1,2)) #* inttime * gain
            specsum = np.nansum(specslice, axis=(1,2)) #* inttime * gain
            
            if square_errors:
                specsum = np.sqrt(np.sum(specslice**2.0, axis = (1,2)))
            
        elif self.circle:
            
            whgood = circle_spec(data, self.circle[0],self.circle[1],\
                                 self.circle[2], annulus = self.circle[3])
            flatdata = data.reshape(data.shape[0],-1)
            whgoodflat = whgood.flatten()
            specslice = flatdata[:,whgoodflat]
            
            #nans = np.isnan(specslice)
            #specslice[nans] = -1
            #gd = specslice > 0
            #specmean = np.mean(specslice[gd], axis = 1)
            #specmedian = np.median(specslice[gd], axis = 1)
            #specsum = np.sum(specslice[gd], axis = 1)
            
            specmean = np.nanmean(specslice, axis = 1)
            specmedian = np.nanmedian(specslice, axis = 1)
            specsum = np.nansum(specslice, axis = 1)

            if square_errors:
                #specsum = np.sqrt(np.sum(specslice[gd]**2.0, axis = 1))
                specsum = np.sqrt(np.sum(specslice**2.0, axis = 1))
                
        elif self.ellipse:
            whgood = ellipse_region(data, (self.ellipse[0], self.ellipse[1]),\
                        self.ellipse[3], self.ellipse[4], self.ellipse[2], \
                        annulus = self.ellipse[5])
            flatdata = data.reshape(data.shape[0],-1)
            whgoodflat = whgood.flatten()
            specslice = flatdata[:,whgoodflat]
            
            specmean = np.nanmean(specslice, axis = 1)
            specmedian = np.nanmedian(specslice, axis = 1)
            specsum = np.nansum(specslice, axis = 1)
            
            if square_errors:
                specsum = np.sqrt(np.nansum(specslice**2.0, axis = 1))
            
        else:
            specslice = data
                                  
            specmean = np.nanmean(specslice, axis=(1,2)) #* inttime * gain
            specmedian = np.nanmean(specslice, axis=(1,2)) #* inttime * gain
            specsum = np.nansum(specslice, axis = (1,2)) #* inttime * gain
                                  
            if square_errors:
                specsum = np.sqrt(np.sum(specslice**2.0, axis = (1,2)))

            
        return specmean, specmedian, specsum
    
    def skyFix(self):
        
        try:
            skyfls = glob(self.cubebasepath + '*_sky_cube.fits')
            skyff = fits.open(skyfls[0])
            self.skydata = skyff[0].data
            self.skyhead = skyff[0].header
            skyflat = self.skydata.reshape(self.skydata.shape[0],-1)
            skyflat[np.isnan(skyflat)] = 0

            pixel = np.arange(self.skydata.shape[0]) + 1.0
            self.skywl = pixel*self.skyhead['CDELT3'] + self.skyhead['CRVAL3']
            self.skywl *= 1e10

            xx = np.arange(skyflat.shape[1])
            skyinterp = interp2d(xx, self.skywl, skyflat, kind='cubic', fill_value=0.1)
            skytarget = skyinterp(xx, self.cubewl + 1).reshape(self.cubedata.shape)
        except Exception as e:
            print(e)
            print('### Problem finding and processing sky cubes,'+\
                  ' are they in the same directory as the merged cube?')
            return
        
        self.skymean, self.skymedian, self.skysum = self.extractRegion(skytarget)
        
        if self.extracted:
            reg = (12500 <= self.cubewl) & (13000 >= self.cubewl)
            fig,ax = mpl.subplots(figsize = (12,7))
            ax.plot(self.cubewl[reg], norm(self.spectrum[reg]),\
                    label='Spectrum')
            ax.plot(self.cubewl[reg], norm(self.skysum[reg]),\
                    label='Sky')
            mpl.legend()
            mpl.show()
            
            skycorr = 0.17

            reg = (12500 <= self.cubewl) & (13000 >= self.cubewl)
            fig,ax = mpl.subplots(figsize = (12,7))
            ax.plot(self.cubewl[reg], norm(self.spectrum[reg]),\
                    label='Spectrum')
            ax.plot(self.cubewl[reg], norm(self.spectrum[reg]*(norm(self.skysum[reg]**skycorr))),\
                    label='ratio')
            mpl.legend()
            mpl.show()
            
            
            #for i in np.arange(-5,5,0.2):
            #    skycorr = i

            #    fig,ax = mpl.subplots(figsize = (12,7))
                #ax.plot(self.cubewl[reg], norm(self.spectrum[reg]),\
                #        label='Spectrum')
                #ax.plot(self.cubewl[reg], \
                #        norm(self.spectrum[reg]*(norm(skysum[reg])*skycorr)),\
                #        label='SkyCorr')
            #    ax.plot(self.cubewl[reg], \
            #            norm(skysum[reg])*skycorr, label='SkyCorr')

            #    mpl.legend()
            #    mpl.show()
            
        else:
            print("Spectrum not extracted yet, can't compare")
            return
        
    def compareErrors(self, savefig=False):
        
        #fig,ax = mpl.subplots(figsize = (12,7))            
        #inttime = self.cubehead['INTTIME']
        #gain = 1.33            

        errold = self.calcErrors(savefig=savefig)
        errshift = self.calcErrors(shift=True,savefig=savefig)
        errshiftmedian = self.calcErrorsFinalCube(savefig=savefig)
        
        errold[errold == 0] = 0.001
        errshift[errshift == 0] = 0.001
        errshiftmedian[errshiftmedian == 0] = 0.001

        fig,ax = mpl.subplots(figsize = (12,7))
        mpl.plot(self.cubewl, errold, label = 'Old')
        mpl.plot(self.cubewl, errshift, label = 'Shifted Cubes')
        mpl.plot(self.cubewl, errshiftmedian, label = 'Shifted Median')
        ax.set_title("Errors Comparison")
        #mpl.plot(self.cubewl, self.spectrummean/(inttime*gain*err), label='Rolling')
        #mpl.plot(self.cubewl, self.spectrum/self.errsum, label='Direct')
        ax.set_ylim((0,0.4))
        mpl.legend(fontsize = 15)
        ax.tick_params(labelsize=15)
        mpl.savefig('/Users/relliotmeyer/Desktop/Errors_'+savefig+'.png', dpi=300)
        mpl.show()
        
        fig,ax = mpl.subplots(figsize = (12,7))
        mpl.plot(self.cubewl, self.spectrum/errold, label = 'Old')
        mpl.plot(self.cubewl, self.spectrum/errshift, label = 'Shifted Cubes')
        mpl.plot(self.cubewl, self.spectrum/errshiftmedian, label = 'Shifted Median')
        ax.set_title("SNR Comparison")
        #mpl.plot(self.cubewl, self.spectrummean/(inttime*gain*err), label='Rolling')
        #mpl.plot(self.cubewl, self.spectrum/self.errsum, label='Direct')
        ax.set_ylim((0,500))
        mpl.legend(fontsize = 15)
        ax.tick_params(labelsize=15)
        mpl.savefig('/Users/relliotmeyer/Desktop/SNR_'+savefig+'.png', dpi=300)
        mpl.show()

    def calcErrors(self, shift = False, savefig = False):
        
        if self.extracted:
            cubefls = glob(self.cubebasepath + '*obs_cube.fits')
            k = 0

            wls = []
            specs = []
            for cube in cubefls:
                cubeff = fits.open(cube)
                data = cubeff[0].data
                head = cubeff[0].header

                pixel = np.arange(data.shape[0]) + 1.0
                wl = pixel*head['CDELT3'] + head['CRVAL3']
                wl *= 1e10

                mean, median, dsum = self.extractRegion(data)

                wls.append(wl)
                specs.append(mean)
                #reg = (12500 <= wl) & (13000 >= wl)
                #ax.plot(wl[reg], dsum[reg]+k)
                #k+=100

            newspecs = []
            wlprime = self.cubewl

            if shift:
                k = len(specs) / 2
                for i, j in enumerate(np.arange(-2*k,2*k,2)):
                    s = specs[i]
                    
                    s[np.isnan(s)] = np.nanmedian(s)
                    f1 = interp1d(wls[i], s , kind='cubic', bounds_error = False, fill_value=1.0)
                    newspecs.append(np.roll(f1(wlprime), int(j)))

                newspecs = np.array(newspecs)
                if savefig:
                    fig,ax = mpl.subplots(figsize = (12,7))
                    ax.set_title('Shifted Spectra')
                    for s in newspecs:
                        ax.plot(self.cubewl, s)
                    #ax.set_ylim((0,np.percentile(speccopy/err, 98)))
                    ax.tick_params(labelsize=15)
                    mpl.savefig('/Users/relliotmeyer/Desktop/shifted_'+savefig+'.png', dpi=300)
                    mpl.show()

            else:
                for i, j in enumerate(specs):
                    s = specs[i]
                    s[np.isnan(s)] = np.nanmedian(s)

                    f1 = interp1d(wls[i], s , kind='cubic', bounds_error = False, fill_value=1.0)
                    newspecs.append(f1(wlprime))
                    
                if savefig:
                    fig,ax = mpl.subplots(figsize = (12,7))
                    ax.set_title('All Cubes')
                    for s in newspecs:
                        ax.plot(wlprime, s)
                    #ax.set_ylim((0,np.percentile(speccopy/err, 98)))
                    ax.tick_params(labelsize=15)
                    mpl.savefig('/Users/relliotmeyer/Desktop/notshifted_'+savefig+'.png', dpi=300)
                    mpl.show()                    
                    
            err = np.std(newspecs, axis = 0) / np.sqrt(len(newspecs))
        else:
            print("Spectrum must be extracted first.")

        return err
    
    def calcErrorsFinalCube(self, savefig = False):
        
        inttime = self.cubehead['INTTIME']
        gain = 1.33            
        
        speccopy = np.array(self.spectrum)# / (inttime * gain))
        speccopy[np.isnan(speccopy)] = -1

        newsums = []
        for j in np.arange(-6,6,1):
            f1 = interp1d(self.cubewl, speccopy , kind='cubic', bounds_error = False, fill_value=1.0)
            newsums.append(np.roll(speccopy, int(j)))

        newsums = np.array(newsums)
        err = np.std(newsums, axis = 0) / np.sqrt(20)
        
        if savefig:
            fig,ax = mpl.subplots(figsize = (12,7))
            ax.set_title('Shifted Median Spectra')
            for s in newsums:
                ax.plot(self.cubewl, s)
            #ax.set_ylim((0,np.percentile(speccopy/err, 98)))
            ax.tick_params(labelsize=15)
            mpl.savefig('/Users/relliotmeyer/Desktop/shiftedmedian_'+savefig+'.png', dpi=300)
            mpl.show()
        
        return err
