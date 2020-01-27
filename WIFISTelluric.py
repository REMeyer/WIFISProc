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
import os

from astropy.visualization import (PercentileInterval, LinearStretch,
                                    ImageNormalize, ZScaleInterval)
from astropy import wcs
from astropy import units as u
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import Angle

import WIFISSpectrum as WS
import WIFISFitting as gf

mpl.rc('text', usetex=True)
mpl.rc('font', family='serif')

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
        self.wl = self.target.cubewl
        self.wlz = self.target.cubewlz
        
        if not self.target.extracted:
            print("Target spectrum not extracted, please extract spectrum before continuing.")
        if not self.telluric.extracted:
            print("Telluric spectrum not extracted, please extract spectrum before continuing.")

        self.reduced = False
        
        self.tellshift = 0
        self.tellscale = 0
        self.tarshift = 0 
        self.tarscale = 0
        
    def do_telluric_reduction(self, hlinemode = 'measure',\
                    interactivetelluric = False, plot = True, profile='lorentzian',\
                    telluricmask = [], hlineplot=False):
        '''Function that reduces the science spectrum with a telluric spectrum. 
        First removes h-lines from the telluric spectrum with either a vega spectrum from file,
        or directly fitting the lines. Has interactive modes to manually stretch and shift
        the fitting spectra to match the target spectra. Then corrects the galaxy spectrum by
        the telluric spectrum.'''

        ##Loading convolved Vega
        vega_con = pd.read_csv('/Users/relliotmeyer/WIFIS/WIFISProc/vega-con-new.txt',\
                               sep = ' ', header = None)
        vega_con.columns = ['a', 'b']   #Col 1 = wavelength, col 2 = FLux
        self.vegawl = vega_con['a']
        self.vegadata = vega_con['b']
        
        self.correctHLines(hlinemode, profile = profile, hlineplot = hlineplot)

        #self.write_reduced_spectrum(kind = 'Telluric')
        
        #Create interpolator of telluric spectrum (using non-NaN values)
        notnan = ~np.isnan(self.tellspecreduced)
        TelStar_Interp = interp1d(self.telluric.cubewl[notnan], self.tellspecreduced[notnan], \
                                  kind='cubic', bounds_error=False)
        notnanerr = ~np.isnan(self.tellerrreduced)
        TelErr_Interp = interp1d(self.telluric.cubewl[notnanerr], self.tellerrreduced[notnanerr], \
                                  kind='cubic', bounds_error=False)
                
        #If interactive then enter interactive fitting mode
        if interactivetelluric:
            #self.interactive_vega(TelStar_Interp, kind = 'Galaxy')
            self.shiftScale(TelStar_Interp, kind = 'Target')

        print("SHIFT AND SCALE FOR TELLURIC IS: ", self.tarshift, self.tarscale)

        #Create final telluric spectrum using interpolator, then adjust science spectrum
        self.tellinterp = TelStar_Interp(self.target.cubewl+self.tarshift) ** self.tarscale
        self.tellerrinterp = TelErr_Interp(self.target.cubewl+self.tarshift) #** self.tarscale
        self.tellerrinterp[self.tellerrinterp < 0] *= -1
        self.tellerrinterp = self.tellerrinterp ** self.tarscale

        
        normtell = WS.norm(self.tellinterp)
        normtellerr = self.tellerrinterp / np.nanmedian(self.tellinterp)
        
        # Removing specified regions from the telluric spectrum via a linear fit
        if len(telluricmask) > 0:
            for i in range(len(telluricmask)):
                whgd = np.where((self.target.cubewl >= telluricmask[i][0]) & \
                                (self.target.cubewl <= telluricmask[i][1]))[0]
                pf = np.polyfit([telluricmask[i][0],telluricmask[i][1]], \
                                [normtell[whgd][0],normtell[whgd][-1]], 1)
                contfit = np.poly1d(pf)
                cont = contfit(self.target.cubewl[whgd])
                normtell[whgd] = cont

        self.reducedspectrum = self.target.spectrum / normtell
        errterm = np.sqrt((self.target.cubenoise / self.target.spectrum)**2.0 + \
                         (normtellerr / normtell)**2.0)
        self.reducederr = self.reducedspectrum * errterm 
        self.reduced = True
            
        #Plot resulting spectrum
        if plot == True:
            wlval = self.target.cubewl
            wlvalz = self.target.cubewlz


            regions = [(9400,9700),(10300,10500),(11000,11500),(11500,11900),\
                       (12200,12500),(12600,13000)]
            for i, region in enumerate(regions):
                fig, axes = mpl.subplots(2,1,figsize = (15,10), sharex=True)
                whreg = (wlvalz >= region[0]) & (wlvalz <= region[1])

                axes[0].plot(wlvalz[whreg],WS.norm(normtell[whreg]), 'r')
                axes[0].plot(wlvalz[whreg],WS.norm(self.target.spectrum[whreg]),'b')

                axes[1].plot(wlvalz[whreg],WS.norm(self.reducedspectrum[whreg]), 'k')
                mpl.tight_layout()
                mpl.minorticks_on()
                mpl.grid(axis='x', which='both')

                mpl.subplots_adjust(wspace=0, hspace=0)
                mpl.show()

            fig, ax = mpl.subplots(figsize = (15,10))

            nonnan = ~np.isnan(self.reducedspectrum)
            ax.plot(wlvalz[nonnan][50:-20], WS.norm(self.reducedspectrum[nonnan][50:-20]), 'k')

            ax.set_title("Reduced and de-Redshifted Spectrum")
            ax.set_xlabel("Wavelength ($\AA$)", fontsize=13)
            ax.set_ylabel("Flux", fontsize = 13)

            ax.xaxis.set_minor_locator(AutoMinorLocator())

            mpl.show()
    
    def correctHLines(self, hlinemode, profile = 'voigt', hlineplot = False):
        
        #Correcting for the hlines (Paschen Series)
        if hlinemode == 'measure':
            # Measures the difference in the central wl of the hlines between the vega 
            # and telluric spectra. Then applies a polynomial fit to the difference
            # to the vega wavelength solution
            
            poly = self.measure_hlines(fittype = 'normal', plot = hlineplot, profile=profile)
            vegawlnew = self.vegawl + poly(self.vegawl)

            # Interpolate new vega spectrum using corrected wavelength solution
            vega_con_Interp = interp1d(vegawlnew, self.vegadata, kind='cubic',\
                                       bounds_error = False)

            # Create new vega correction spectrum using additional shift and scale factors
            vegainterp = vega_con_Interp(self.telluric.cubewl + self.tellshift) ** self.tellscale

            #Create final vega spectrum using interpolator, then adjust telluric spectrum
            self.tellspecreduced = self.telluric.spectrum / WS.norm(vegainterp)
            self.tellerrreduced = self.telluric.cubenoise / WS.norm(vegainterp)

            print("SHIFT AND SCALE FOR VEGA IS: ", self.tellshift, self.tellscale)
            if hlineplot:
                fig, axis = mpl.subplots(2,1,figsize=(15,10), sharex=True)
                axis[0].plot(self.telluric.cubewl, WS.norm(self.telluric.spectrum),'b', \
                             label='Standard Star Spectrum')
                axis[0].plot(self.telluric.cubewl, WS.norm(vegainterp),'r', label='Vega Spectrum')
                axis[1].plot(self.telluric.cubewl, WS.norm(self.tellspecreduced),'k', label='Telluric Spectrum')
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
                #mpl.savefig('/Users/relliotmeyer/Desktop/VegaCorrection.pdf', dpi=500)
                mpl.show()

        elif hlinemode == 'none':
            # Just shifts and scales the interpolated vega spectrum.

            vega_con_Interp = interp1d(self.vegawl, self.vegadata, kind='cubic',\
                                       bounds_error = False)

            vegainterp = vega_con_Interp(self.telluric.cubewl + self.tellshift) ** self.tellscale

            #Create final vega spectrum using interpolator, then adjust telluric spectrum
            self.tellspecreduced = self.telluric.spectrum / WS.norm(vegainterp) 
            self.tellerrreduced = self.telluric.cubenoise / WS.norm(vegainterp)

            print("SHIFT AND SCALE FOR VEGA IS: ", self.tellshift, self.tellscale)

            if plot:
                fig, axis = mpl.subplots(2,1,figsize=(15,10))
                axis[0].plot(self.telluric.cubewl, WS.norm(self.telluric.spectrum),'b')
                axis[0].plot(self.telluric.cubewl, norm(vegainterp),'r')
                axis[1].plot(self.telluric.cubewl, self.tellspecreduced)
                mpl.show()

        elif hlinemode == 'interactive':
            # Enables an interactive version of the shifting and scaling to fine tune 
            # shift and scale parameters. 
            
            vega_con_Interp = interp1d(self.vegawl, self.vegadata, kind='cubic', bounds_error = False)

            self.interactive_vega(vega_con_Interp)
            vegainterp = vega_con_Interp(self.telluric.cubewl + self.tellshift) ** self.tellscale
            #self.shiftScale(vega_con_Interp)

            #Create final vega spectrum using interpolator, then adjust telluric spectrum
            self.tellspecreduced = self.telluric.spectrum / WS.norm(vegainterp)
            self.tellerrreduced = self.telluric.cubenoise / WS.norm(vegainterp)

            print("SHIFT AND SCALE FOR VEGA IS: ", self.tellshift, self.tellscale)

        elif hlinemode == 'remove':
            # Removes the hlines from the telluric spectrum by performing a linear fit over the 
            # affected regions. 
            
            newtelluric = self.measure_hlines(fittype = 'normal', plot=False,\
                            remove=True, profile=profile)
            if plot:
                fig, axes = mpl.subplots(figsize = (15,10))
                axes.plot(self.telluric.cubewl, self.telluric.spectrum,'b')
                axes.plot(self.telluric.cubewl, newtelluric,'r')
                mpl.show()

            self.tellspecreduced = newtelluric
            self.tellerrreduced = self.telluric.cubenoise

        elif hlinemode == 'broaden':
            poly = self.measure_hlines(fittype = 'normal', plot = True, profile=profile)
            vegawlnew = self.vegawl + poly(self.vegawl)

            xs = np.arange(-50,51, self.vegawl[5]-self.vegawl[4])
            #vp = gf.voigtfullnorm(xs, 0.5115, -3.0)
            vp = gf.voigtfullnorm(xs, 15, 0.005)
            out = np.convolve(self.vegadata, vp, mode='same')
            #out = vegadata
            vega_con_Interp = interp1d(vegawlnew, out, kind='cubic', bounds_error = False)
            vegainterp = vega_con_Interp(self.telluric.cubewl+self.tellshift) ** self.tellscale
            self.tellspecreduced = self.telluric.spectrum / WS.norm(vegainterp)
            self.tellerrreduced = self.telluric.cubenoise / WS.norm(vegainterp)


            if plot:
                fig, axis = mpl.subplots(2,1,figsize=(15,10))
                axis[0].plot(self.telluric.cubewl, WS.norm(self.telluric.spectrum),'b')
                axis[0].plot(self.telluric.cubewl, WS.norm(vegainterp),'r')
                axis[1].plot(self.telluric.cubewl, self.tellspecreduced)
                axes.tick_params(which='minor')
            #axis.plot(vegawl, norm(vegadata), 'b')
            #axis.plot(vegawl, norm(out),'r')
                mpl.show()

    def measure_hlines(self, fittype='quadratic', plot=True, remove=False, profile='lorentzian'):
        '''TESTING FUNCTION: To determine the wl offset between the vega and telluric spectrum'''

        #hlinelow =  [8840,8960,9190,9520,10000,10880,12775]
        #hlinehigh = [8925,9050,9275,9600,10120,11020,12880]
        hlinelow =  [8840,8960,9190,9520,9900,10880,12720]
        hlinehigh = [8925,9050,9275,9600,10120,11020,12900]

        midguess = np.array([8865, 9017, 9232, 9550, 10052, 10941, 12822])
        midwl = [np.mean([hlinelow[i],hlinehigh[i]]) for i in range(len(hlinelow))]
        
        telluriccopy = np.array(self.telluric.spectrum) 

        diffs = []
        if plot and fittype == 'normal':
            fig, axes = mpl.subplots(2,4, figsize = (15,7))
            axes = axes.flatten()
        
        good = np.ones(len(midguess[2:]), dtype=bool)
        linefitwl = []
        for l in range(len(hlinelow)):
            
            wh = np.where((self.telluric.cubewl >= hlinelow[l]) &\
                          (self.telluric.cubewl <= hlinehigh[l]))[0]
            linewl = self.telluric.cubewl[wh]
            linedata = self.telluric.spectrum[wh]
            
            if remove:
                pf = np.polyfit([linewl[0],linewl[-1]], [linedata[0],linedata[-1]], 1)
                contfit = np.poly1d(pf)
                cont = contfit(linewl)

                telluriccopy[wh] = (linedata/fitg) * cont
                continue
                
            a = np.min(linedata)-np.max(linedata)
            o = np.median(linedata)

            try:
                ### Fitting Telluric line profile
                if profile == 'lorentzian':
                    #popt,pcov = gf.lorentzian_fit(linewl, linedata, [-150., 30., midguess[l], 7])
                    popt,pcov = gf.fit_func(linewl, linedata, \
                                            [a, 30., midguess[l], o], func=gf.lorentzian)
                    fitg = gf.lorentz(linewl, popt)
                    midline = popt[2]
                elif profile == 'voigt':
                    #popt,pcov = gf.voigt_fit(linewl, linedata, [midguess[l], -150., 30., 10., 7.])
                    popt,pcov = gf.fit_func(linewl, linedata, \
                                             [midguess[l], a, 35., 1., o], func=gf.voigtfull)
                    fitg = gf.voigt(linewl, popt)
                    midline = popt[0]

                if plot:
                    axes[l].plot(linewl, WS.norm(linedata),'b:')
                    axes[l].plot(linewl, WS.norm(fitg), 'r:')
                    axes[l].axvline(midline, color = 'g', linestyle = ':')
                    #axes[l].plot(linewl, linedata/fitg)

                ### Fitting Vega line profile
                vwh = np.where((self.vegawl >= hlinelow[l]) & \
                               (self.vegawl <= hlinehigh[l]))[0]
                vwl = self.vegawl[vwh]
                vdata = self.vegadata[vwh]

                try:
                    poptvega,pcov = gf.fit_func(vwl, vdata, [midguess[l], 15., 30., 10., 7.],\
                            func=gf.voigtfull)
                except:
                    poptvega,pcov = gf.fit_func(vwl, vdata, [midguess[l], 15., 30., 10., 7.],\
                            func=gf.voigtfull)
                fitvega = gf.voigt(vwl, poptvega)
                midline = poptvega[0]

                #Saving difference
                if l > 1:
                    if profile == 'lorentzian':
                        diffs.append(popt[2] - poptvega[2])
                        linefitwl.append(midline)
                    elif profile == 'voigt':
                        diffs.append(popt[0] - poptvega[0])
                        linefitwl.append(midline)

                #### Plot code 
                if plot:                    
                    axes[l].plot(vwl, 0.2+WS.norm(vdata),'b--')
                    axes[l].plot(vwl, 0.2+WS.norm(fitvega), 'r--')
                    axes[4].set_xlabel("Wavelength $(\AA)$")
                    axes[l].axvline(midline, color = 'g', linestyle = '--')
                    
            except Exception as e:
                print(e)
                print("Couldn't fit line #", l)
                if l > 1:
                    good[l - 2] = False
                
        pf = np.polyfit(linefitwl, diffs, 2)
        #pf = np.polyfit(midguess[2:][good], diffs[2:][good], 2)
        polyfit = np.poly1d(pf)
        cont = polyfit(midwl)

        if plot:
            mpl.show()
            fig, axes = mpl.subplots()
            axes.plot(linefitwl, diffs)
            axes.plot(midwl, cont)
            axes.set_xlabel("Wavelength $(\AA)$")
            axes.set_ylabel("Offset $(\AA)$")
            mpl.show()

        if remove:
            return telluriccopy
        else:
            return polyfit

                
    def plotSpectra(self, kind='both'):

        if kind == 'both':
            fig, axes = mpl.subplots(2,1, figsize = (15,10))
            axes[0].plot(self.telluric.cubewl, self.telluric.spectrum, label='Telluric')
            axes[1].plot(self.target.cubewl, self.target.spectrum, label='Target')

            axes[0].minorticks_on()
            axes[1].minorticks_on()
            mpl.show()
        elif kind == 'telluric':
            fig, axes = mpl.subplots(figsize = (15,10))
            axes.plot(self.telluric.cubewl, self.telluric.spectrum, label='Telluric')
            axes.minorticks_on()
            mpl.show()
        elif kind == 'target':
            fig, axes = mpl.subplots(figsize = (15,10))
            axes.plot(self.target.cubewl, self.target.spectrum, label='Target')
            axes.minorticks_on()
            mpl.show()

    def write_reduced_spectrum(self, suffix='', kind = 'target'):
        '''Function that writes the final reduced spectrum to file. Must have created a reduced spectrum first.
        
        If there is no reduced spectrum an extracted spectrum will be written instead.
        
        The first extension is the spectrum, second is the wavelength array, third (if calculated) is the 
        uncertainties'''
        
        if self.reduced:
            print("Writing reduced final spectrum....")
            hdu = fits.PrimaryHDU(self.reducedspectrum)
        else:
            print("Science spectrum not reduced...returning")
            return

        hdu2 = fits.ImageHDU(self.target.cubewl, name = 'WL')
        if self.target.uncertainties:
            hdu3 = fits.ImageHDU(self.reducederr, name = 'ERR')
            hdul = fits.HDUList([hdu,hdu2,hdu3])
        else:
            print('NO UNCERTAINTIES CALCULATED...NOT INCLUDING IN FITS')
            hdul = fits.HDUList([hdu,hdu2])

        hdul.writeto(self.target.cubefile[:-5]+'_telluricreduced_'\
                     +suffix+'.fits', overwrite=True)
        print("Wrote to "+self.target.cubefile[:-5]+'_telluricreduced_'+suffix+'.fits')

    def interactive_vega(self, vega_con_Interp, kind = 'Telluric'):
        '''Function that interactively allows for the fitting of either a telluric star and the vega spectrum
        or the telluric spectrum and the science target spectrum.'''

        mpl.ion()
        if kind == 'Telluric':
            target = 'Telluric'
            calib = 'Vega'
            wl = self.telluric.cubewl
            tstar = self.telluric.spectrum
            shift = self.tellshift
            scale = self.tellscale
        else:
            target = 'Galaxy'
            calib = 'Telluric'
            wl = self.target.cubewl
            tstar = self.target.spectrum
            shift = self.tarshift
            scale = self.tarscale

        vegainterp = vega_con_Interp(wl + shift) ** scale
        TellSpec = tstar / (vegainterp/np.nanmedian(vegainterp))

        #Creating the initial plot
        fig, axes = mpl.subplots(2, figsize = (14,8),sharex=True)
        axes[0].plot(wl, WS.norm(tstar), 'r', label='Target')
        axes[0].plot(wl, WS.norm(vegainterp), 'b', label='Calibration')
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
            TellSpec = tstar / WS.norm(vegainterp)

            axes[0].plot(wl, WS.norm(tstar), 'r', label=target)
            axes[0].plot(wl, WS.norm(vegainterp), 'b', label=calib)
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
            self.tellscale = scale
            self.tellshift = shift
        else:
            self.tarscale = scale
            self.tarshift = shift
            
    def shiftScale(self, vega_con_Interp, kind = 'Telluric'):
        '''Function that interactively allows for the fitting of either a telluric star and the vega spectrum
        or the telluric spectrum and the science target spectrum.'''

        if kind == 'Telluric':
            wl = self.telluric.cubewl
            tstar = self.telluric.spectrum
            shift = self.tellshift
            scale = self.tellscale
        else:
            wl = self.target.cubewl
            tstar = self.target.spectrum
            shift = self.tarshift
            scale = self.tarscale

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

    def remove_features(self, target, confirm=True, kind='target',\
                       inspect = True):

        #Creating the initial plot
        if kind == 'telluric':
            wl = np.array(self.telluric.cubewl)
            spec = np.array(self.tellspecreduced)
        elif kind == 'target_z':
            wl = np.array(self.target.cubewlz)
            spec = np.array(self.reducedspectrum)
        else:
            wl = np.array(self.target.cubewl)
            spec = np.array(self.reducedspectrum)
            
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
                        self.tellspecreduced = spec 
                    else:
                        print("Saving new Galaxy spectrum...")
                        self.reducedspectrum = spec
                else:
                    continue
            else:
                if kind == 'telluric':
                    print("Saving new Telluric spectrum...")
                    self.tellspecreduced = spec 
                else:
                    print("Saving new Galaxy spectrum...")
                    self.telluricspectrum = spec

    def telluricAdjust(self, region, scale, shift=0.0):
        
        start = region[0]
        end = region[1]

        wh = np.where((self.target.cubewl >= start) & (self.target.cubewl <= end))[0]

        wlslice = self.target.cubewl[wh]
        tellslice = self.TellInterp[wh]
        galslice = self.target.spectrum[wh]

        fig, axes = mpl.subplots(2,1, figsize = (15,10))
        axes[0].plot(wlslice, WS.norm(tellslice), 'r')
        axes[0].plot(wlslice, WS.norm(galslice), 'b')
        axes[1].plot(wlslice, galslice/WS.norm(tellslice**scale),'k')
        mpl.show()
    


    def plotImages(self):
        '''Produces plots of the telluric and galaxy images. Axes should be 
        in celestial coordinates.'''
        
        fig = mpl.figure(figsize = (12,10))
        gs = gridspec.GridSpec(2,1)
        
        self.telluric.plotImage(subimage=gs[0,0])
        self.target.plotImage(subimage=gs[1,0])
        
        self.telluric.plotax.set_title("Telluric", fontsize = 15)
        self.target.plotax.set_title("Target", fontsize = 15)

        mpl.show()        
