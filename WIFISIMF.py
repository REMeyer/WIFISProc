import pandas as pd
import matplotlib.pyplot as mpl
from scipy.interpolate import interp1d
from scipy import stats
import numpy as np
from astropy.io import fits
from sys import exit
import matplotlib.patches as patches
from glob import glob
import os
from astropy.visualization import (PercentileInterval, LinearStretch,
                                    ImageNormalize, ZScaleInterval)

import WIFISTelluric as WT
import WIFISSpectrum as WS
import WIFISFitting as gf
import scipy.ndimage

mbase = '/home/elliot/mcmcgemini/spec'

def convolvemodels(wlfull, datafull, veldisp):

    reg = (wlfull >= 9500) & (wlfull <= 13500)
    
    wl = wlfull[reg]
    dw = wl[1]-wl[0]
    data = datafull[reg]

    c = 299792.458

    #Sigma from description of models
    m_center = 11500
    m_sigma = np.abs((m_center / (1 + 100./c)) - m_center)
    f = m_center + m_sigma
    v = c * ((f/m_center) - 1)
    
    sigma_gal = np.abs((m_center / (veldisp/c + 1.)) - m_center)
    sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    #convolvex = np.arange(-5*sigma_conv,5*sigma_conv, 2.0)
    #gaussplot = gf.gauss_nat(convolvex, [sigma_conv,0.])

    #out = np.convolve(datafull, gaussplot, mode='same')
    out = scipy.ndimage.gaussian_filter(datafull, sigma_conv/dw)

    return out


####################################################################
class WIFISIMF(WT.WIFISTelluric):

    def __init__(self, target, telluric, veldisp, mode='lines'):
        super().__init__(target, telluric)
        
        if mode == 'lines':
            
            self.pa_raw = False
            self.veldisp = veldisp

        self.merged = True
        self.resampled = False

    def plotIMFLines(self, kind = 'Full', continuum = False, mask = [], save=False, \
                     oldline = False, gal='Galaxy', age=13.5):
        
        if not self.reduced:
            print("Spectrum not reduced. Please remove telluric lines first.")
            return
        
        if oldline:
            self.linelow = [9905,10337,11372,11680,11765,12505, 12670, 12810]
            self.linehigh = [9935,10360,11415,11705,11793,12545, 12690, 12840]
            self.bluelow = [9855,10300,11340,11667,11710,12460, 12648, 12780]  
            self.bluehigh = [9880,10320,11370,11680,11750,12495, 12660, 12800] 
            #self.bluelow = [9855,10300,11340,11667,11710,12460, 12600, 12780]  
            #self.bluehigh = [9880,10320,11370,11680,11750,12495, 12630, 12800] 
            #self.redlow = [9940,10365,11417,11710,11793,12555, 12855, 12700]  
            self.redlow = [9940,10365,11417,11710,11793,12555, 12700, 12860]   
            self.redhigh = [9970,10390,11447,11750,11810,12590, 12720, 12870] 
            self.line_name = ['FeH','CaI','NaI','KI a','KI b', 'KI 1.25', 'NaI 1.27',r'Pa$\beta$']
        else:
            self.linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12309, 12810]
            self.linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12333, 12840]
            self.bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12240, 12780]  
            self.bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12260, 12800] 
            self.redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12360, 12860]   
            self.redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12390, 12870]
            self.line_name = ['FeH','CaI','NaI','KI a','KI b', 'KI 1.25','NaI 1.23',r'Pa$\beta$']
        
        self.chem_names = ['WL', 'Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', \
            'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', \
            'Ti+', 'Ti-', 'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', \
            'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
            'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

        fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
        axes = axes.flatten()
        
        wl = self.wl / (1+self.z)

        fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
        chemage = np.array([1,3,5,7,9,11,13])
        agemin = np.argmin(np.abs(fullage - age))
        print(f"Using age: {fullage[agemin]}")
        if fullage[agemin] >= 10.0:
            mfl1 = mbase+'/vcj_ssp/VCJ_v8_mcut0.08_t'+str(fullage[agemin])+'_Zp0.0.ssp.imf_varydoublex.s100'
            mfl2 = mbase+'/atlas/atlas_ssp_t'+str(chemage[agemin])+'_Zp0.0.abund.krpa.s100'
        else:
            mfl1 = mbase+'/vcj_ssp/VCJ_v8_mcut0.08_t0'+str(fullage[agemin])+'_Zp0.0.ssp.imf_varydoublex.s100'
            mfl2 = mbase+'/atlas/atlas_ssp_t0'+str(chemage[agemin])+'_Zp0.0.abund.krpa.s100'
        #model2 = np.loadtxt('/Users/relliotmeyer/Thesis_Work/ssp_models/vcj_ssp/VCJ_v8_mcut0.08_t13.5_Zp0.0.ssp.imf_varydoublex.s100')

        model = pd.read_table(mfl1, delim_whitespace = True, header=None)
        model2 = pd.read_table(mfl2, skiprows=2, names = self.chem_names, \
                               delim_whitespace = True, header=None)
        na = np.array(model2['Na+0.9'])
        solar = np.array(model2['Solar'])

        ratio = na/solar

        mwl = np.array(model[0])

        mspec = np.array(model[73])

        #mspec2 = mspec * ratio
        #mspec2 = model[221]
        mspec2 = model[153]
        mspec3 = mspec + mspec*ratio

        #mspec = WT.convolvemodels(mwl, mspec, 140.)
        #mspec2 = WT.convolvemodels(mwl, mspec2, 140.)
        mspec = convolvemodels(mwl, mspec, self.veldisp)
        mspec2 = convolvemodels(mwl, mspec2, self.veldisp)
        mspec3 = convolvemodels(mwl, mspec3, self.veldisp)

        mspecs = [mspec,mspec2,mspec3]
        mlabels = ['Kroupa','BH','Kroupa, Na+0.9']
        mcolours = ['tab:green','tab:red','tab:blue']

        if kind == 'Full':
            data = self.reducedspectrum
        elif kind == 'Raw':
            data = self.target.spectrum
        
        for i in range(len(self.linelow)):
            wh = np.where((wl >= self.bluelow[i]) & (wl <= self.redhigh[i]))[0]
            wlslice = wl[wh]
            if (self.line_name[i] == 'Pa Beta') and self.pa_raw:
                print('Non-telluric PaB enabled')
                dataslice = self.target.spectrum[wh]
            else:
                dataslice = data[wh] #/ np.median(data[wh])
                
            #errslice = err[wh] / np.median(data[wh])
            polyfit, regions = self.removeLineSlope(wlslice, dataslice, i)
            dataslice /= polyfit(wlslice)
            errslice = self.reducederr[wh]/polyfit(wlslice)
            
            whm = np.where((mwl >= self.bluelow[i]) & (mwl <= self.redhigh[i]))[0]
            wlmslice = mwl[whm]
            
            
            axes[i].plot(wlslice, dataslice, linewidth = 2.5, color='k', label = gal)
            axes[i].fill_between(wlslice,dataslice + errslice, dataslice-errslice,\
                                 facecolor = 'gray', alpha = 0.5)
            axes[i].set_title(self.line_name[i], fontsize = 17)
            
            if gal == 'M87' and self.line_name[i] == 'KI 1.25':
                axes[i].axvspan(self.bluelow[i], self.bluehigh[i], facecolor='grey',\
                                alpha=0.2)
                axes[i].axvspan(self.redlow[i], self.redhigh[i],facecolor='grey',\
                                alpha=0.2)
                axes[i].axvspan(self.linelow[i], self.linehigh[i],facecolor='grey',\
                                alpha=0.2)
            else:
                axes[i].axvspan(self.bluelow[i], self.bluehigh[i], facecolor='b',\
                                alpha=0.2)
                axes[i].axvspan(self.redlow[i], self.redhigh[i],facecolor='b', alpha=0.2)
                axes[i].axvspan(self.linelow[i], self.linehigh[i],facecolor='g',\
                                alpha=0.2)

            for j, spec in enumerate(mspecs):
                mslice = spec[whm]
                mpolyfit, mregions = self.removeLineSlope(wlmslice, mslice, i)
                mslice /= mpolyfit(wlmslice)
                axes[i].plot(wlmslice, mslice, color=mcolours[j], linewidth = 4.5, \
                        linestyle='--', label = mlabels[j])

            axes[i].tick_params(axis='both', which='major', labelsize=15)
            axes[i].set_xlim((self.bluelow[i], self.redhigh[i]))

        handles, labels = axes[0].get_legend_handles_labels()
        #fig.legend(handles,labels, bbox_to_anchor=(1.09, 0.25), fontsize='large')
        fig.legend(handles,labels, fontsize='large')
        #mpl.tight_layout()
        #fig.legend(handles, labels, fontsize='large')
        if save:
            mpl.savefig(save, dpi = 400)
        else:
            mpl.show()        
        
    def removeLineSlope(self, wlc, mconv, i):
        
        #Define the bandpasses for each line 
        bluepass = np.where((wlc >= self.bluelow[i]) & (wlc <= self.bluehigh[i]))[0]
        redpass = np.where((wlc >= self.redlow[i]) & (wlc <= self.redhigh[i]))[0]
        mainpass = np.where((wlc >= self.linelow[i]) & (wlc <= self.linehigh[i]))[0]

        #Cacluating center value of the blue and red bandpasses
        blueavg = np.mean([self.bluelow[i], self.bluehigh[i]])
        redavg = np.mean([self.redlow[i], self.redhigh[i]])

        blueval = np.nanmean(mconv[bluepass])
        redval = np.nanmean(mconv[redpass])

        pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
        polyfit = np.poly1d(pf)

        return polyfit, [bluepass, redpass, mainpass]

    def investigateRegions(self):
        mpl.close('all')
        for i in range(len(self.linelow)):
            fig, axes = mpl.subplots(figsize = (7,5))
            
            twl = self.telluric.cubewl / (1. + self.z)
            
            fp = np.where((twl >= self.bluelow[i]) & (twl <= self.redhigh[i]))[0]
            axes.plot(twl[fp], WS.norm(self.tellspecreduced[fp]), label='Telluric')
            
            gwl = self.target.cubewlz
            
            fp = np.where((gwl >= self.bluelow[i]) & (gwl <= self.redhigh[i]))[0]
            axes.plot(gwl[fp], WS.norm(self.target.spectrum[fp]), label='Target')
            axes.plot(gwl[fp], WS.norm(self.reducedspectrum[fp]), label='Target Reduced')
            
            mpl.legend()
            mpl.show()
            
    def mergeSpectra(self, to_merge):

        if self.merged:
            self.revertMerge()

        basewlz = self.target.cubewl / (1 + self.z)

        stacked = []
        stackederr = []
        for imfobj in to_merge:
            stacked.append(np.interp(basewlz, imfobj.target.cubewl/(1+imfobj.z), \
                    imfobj.reducedspectrum))
            stackederr.append(np.interp(basewlz, imfobj.target.cubewl/(1+imfobj.z), \
                    imfobj.reducederr)**2.0)

        stacked = np.array(stacked)
        stackederr = np.array(stackederr)

        finalspectrum = np.mean(stacked, axis = 0)
        finalerr = np.sqrt(np.sum(stackederr, axis = 0)) / len(stackederr)

        self.original_reducedspectrum = np.array(self.reducedspectrum)
        self.reducedspectrum = np.array(finalspectrum)

        self.original_reducederr = np.array(self.reducederr)
        self.reducederr = np.array(finalerr)

        self.merged = True

    def revertMerge(self):

        self.reducedspectrum = np.array(self.original_reducedspectrum)
        self.reducederr = np.array(self.original_reducederr)

    def resampleSpectrum(self, n = 2):

        if self.resampled:
            self.revertResample()

        newdata = []
        newerr = []
        newwlarr = []

        i = 0
        while i < len(self.reducedspectrum):
            if i + n - 1 < len(self.reducedspectrum):
                newwlarr.append(np.mean(self.target.cubewl[i:i+n-1]))
                newdata.append(np.mean(self.reducedspectrum[i:i+n-1]))
                newerr.append(np.mean(self.reducederr[i:i+n-1]))
            else:
                newwlarr.append(self.target.cubewl[i])
                newdata.append(self.reducedspectrum[i])
                newerr.append(self.reducederr[i])
            i += n

        #print len(newwlarr), len(newdata) 
        self.original_wl = np.array(self.wl)
        self.wl = np.array(newwlarr)

        self.original_reducedspectrum = np.array(self.reducedspectrum)
        self.reducedspectrum = np.array(newdata)

        self.original_reducederr = np.array(self.reducederr)
        self.reducederr = np.array(newerr)

        self.resampled = True

    def revertResample(self):
        
        self.wl = np.array(self.original_wl)
        self.reducedspectrum = np.array(self.original_reducedspectrum)
        self.reducederr = np.array(self.original_reducederr)


    def maskRegion(self, wlstart, wlend, plot=False, confirm = True):
        
        if not self.reduced:
            print("Spectrum not reduced")
            return
        
        wh = np.where(np.logical_and(self.wl >= wlstart, self.wl <= wlend))[0]
        
        if plot:
            mpl.plot(self.wl[wh],self.reducedspectrum[wh])
            mpl.show()
        
        if not confirm:
            self.reducedspectrum[wh] = np.nan
        else:
            domask = input("Do mask?: ")
            if domask.lower() == 'y':
                self.reducedspectrum[wh] = np.nan
                
        