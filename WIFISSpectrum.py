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