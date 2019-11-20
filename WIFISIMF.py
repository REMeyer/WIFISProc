import pandas as pd
import matplotlib.pyplot as mpl
from scipy.interpolate import interp1d
from scipy import stats
import numpy as np
from astropy.io import fits
from sys import exit
import matplotlib.patches as patches
from glob import glob
import gaussfit as gf
import os
from astropy.visualization import (PercentileInterval, LinearStretch,
                                    ImageNormalize, ZScaleInterval)

import WIFISTelluric as WT
import WIFISSpectrum as WS


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


####################################################################
class WIFISIMF(WT.WIFISTelluric):

    def __init__(self, target, telluric, veldisp, mode='lines'):
        super().__init__(target, telluric)
        
        if mode == 'lines':
            
            self.pa_raw = False
            self.veldisp = veldisp
            
    def plotIMFLines(self, kind = 'Full', continuum = False, mask = [], save=False, gal='M85'):
        
        if not self.reduced:
            print("Spectrum not reduced. Please remove telluric lines first.")
            return
        
        oldline = True
        if oldline:
            self.linelow = [9905,10337,11372,11680,11765,12505, 12670, 12810]
            self.linehigh = [9935,10360,11415,11705,11793,12545, 12690, 12840]
            self.bluelow = [9855,10300,11340,11667,11710,12460, 12648, 12780]  
            self.bluehigh = [9880,10320,11370,11680,11750,12495, 12660, 12800] 
            #self.bluelow = [9855,10300,11340,11667,11710,12460, 12600, 12780]  
            #self.bluehigh = [9880,10320,11370,11680,11750,12495, 12630, 12800] 
            #self.redlow = [9940,10365,11417,11710,11793,12555, 12855, 12700]  
            self.redlow = [9940,10365,11417,11710,11793,12555, 12700, 12860]   
            self.redhigh = [9970,10390,11447,11750,11810,12590, 12720, 12880] 
            self.line_name = ['FeH','CaI','NaI','KI a','KI b', 'KI 1.25', 'NaI 1.27', 'Pa Beta']
        else:
            self.linelow = [9905,10337,11372,11680,11765,12505,   12309,12810]
            self.linehigh = [9935,10360,11415,11705,11793,12545,  12333,12840]
            self.bluelow = [9855,10300,11340,11667,11710,12460,   12240,12780]  
            self.bluehigh = [9880,10320,11370,11680,11750,12495,  12260,12800] 
            self.redlow = [9940,10365,11417,11710,11793,12555,    12360,12860]   
            self.redhigh = [9970,10390,11447,11750,11810,12590,   12390,12870]
            self.line_name = ['FeH','CaI','NaI','KI a','KI b', 'KI 1.25','NaI 1.23','Pa Beta']
        
        self.chem_names = ['WL', 'Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-',\
            'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
            'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
            'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

        fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
        axes = axes.flatten()
        
        wl = self.target.cubewlz
        #wl = self.galwl / (1+0.002435)

        mfl1 = '/Users/relliotmeyer/Thesis_Work/ssp_models/vcj_ssp/'+\
                'VCJ_v8_mcut0.08_t13.5_Zp0.0.ssp.imf_varydoublex.s100'
        mfl2 = '/Users/relliotmeyer/Thesis_Work/ssp_models/atlas/'+\
                'atlas_ssp_t13_Zp0.0.abund.krpa.s100'
        #model2 = np.loadtxt('/Users/relliotmeyer/Thesis_Work/ssp_models/vcj_ssp/VCJ_v8_mcut0.08_t13.5_Zp0.0.ssp.imf_varydoublex.s100')

        model = pd.read_table(mfl1, delim_whitespace = True, header=None)
        model2 = pd.read_table(mfl2, skiprows=2, names = self.chem_names, \
                               delim_whitespace = True, header=None)
        na = np.array(model2['Na+0.9'])
        solar = np.array(model2['Solar'])

        ratio = na/solar

        mwl = np.array(model[0])

        mspec = np.array(model[74])

        #mspec2 = mspec * ratio
        mspec2 = model[221]

        #mspec = WT.convolvemodels(mwl, mspec, 140.)
        #mspec2 = WT.convolvemodels(mwl, mspec2, 140.)
        mspec = convolvemodels(mwl, mspec, self.veldisp)
        mspec2 = convolvemodels(mwl, mspec2, self.veldisp)

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
            errslice = self.target.cubeerr[wh]/polyfit(wlslice)


            whm = np.where((mwl >= self.bluelow[i]) & (mwl <= self.redhigh[i]))[0]
            wlmslice = mwl[whm]
            
            mslice = mspec[whm]
            mpolyfit, mregions = self.removeLineSlope(wlmslice, mslice, i)
            mslice /= mpolyfit(wlmslice)
            
            mslice2 = mspec2[whm]
            mpolyfit, mregions = self.removeLineSlope(wlmslice, mslice2, i)
            mslice2 /= mpolyfit(wlmslice)


            axes[i].plot(wlslice, dataslice,'b', linewidth = 2.5, color='k', label = gal)
            #axes[i].axhline(1.0, color = 'k')
            #axes[i].plot(wlmslice, bhslice,'r--', label='Bottom-Heavy')
            #axes[i].fill_between(wlslice,dataslice + errslice,dataslice-errslice, facecolor = 'gray', alpha=0.5)
            axes[i].set_title(self.line_name[i], fontsize = 15)
            
            if gal == 'M87' and self.line_name[i] == 'KI 1.25':
                axes[i].axvspan(self.bluelow[i], self.bluehigh[i], facecolor='grey', alpha=0.2)
                axes[i].axvspan(self.redlow[i], self.redhigh[i],facecolor='grey', alpha=0.2)
                axes[i].axvspan(self.linelow[i], self.linehigh[i],facecolor='grey', alpha=0.2)
            else:
                axes[i].axvspan(self.bluelow[i], self.bluehigh[i], facecolor='b', alpha=0.2)
                axes[i].axvspan(self.redlow[i], self.redhigh[i],facecolor='r', alpha=0.2)
                axes[i].axvspan(self.linelow[i], self.linehigh[i],facecolor='m', alpha=0.2)
            #axes[i].fill_between(wlslice[regions[2]], dataslice[regions[2]], y2 = 1.0)
            #axes[i].fill_between(wlslice, dataslice - errslice, dataslice+errslice, facecolor='grey')

            axes[i].plot(wlmslice, mslice,'g--',linewidth = 3.5, label = 'Kroupa')
            axes[i].plot(wlmslice, mslice2,'r--',linewidth = 3.5, label = 'BH')
            axes[i].tick_params(axis='both', which='major', labelsize=13)
            axes[i].set_xlim((self.bluelow[i], self.redhigh[i]))

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels, bbox_to_anchor=(0.98, 0.25), fontsize='large')
        mpl.tight_layout()
        #fig.legend(handles, labels, fontsize='large')
        if save:
            mpl.savefig('/Users/relliotmeyer/Desktop/'+save+'.pdf', dpi = 400)
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

        blueval = np.mean(mconv[bluepass])
        redval = np.mean(mconv[redpass])

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
            
