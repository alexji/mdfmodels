import numpy as np
from scipy import interpolate
import time

import os
datapath = os.path.join(os.path.dirname(__file__),"data")

def load_data(name, minlogpdf=-999):
    assert name in ["leaky_box","pre_enriched_box","extra_gas"]
    inputs = np.load(os.path.join(datapath,f"{name}_other.npy"),allow_pickle=True)
    start = time.time()
    lnpdfs = np.load(os.path.join(datapath,f"{name}_pdfs.npy"))
    print(f"{time.time()-start:.1f}s to load pdfs")
    start = time.time()
    lnpdfs = np.log(lnpdfs)
    lnpdfs[lnpdfs < minlogpdf] = minlogpdf
    print(f"{time.time()-start:.1f}s to take log")
    return lnpdfs, inputs

def fast_loglkhd_leaky_box(theta, fehdata, efehdata, griddata):
    N = len(fehdata)
    logp = np.log10(theta[0])
    xp = np.zeros((N, 3))
    xp[:,0] = logp
    xp[:,1] = efehdata
    xp[:,2] = fehdata
    lnpdfs, inputs = griddata
    lnp = interpolate.interpn(inputs, lnpdfs, xp, bounds_error=False, fill_value=-999)
    return np.sum(lnp)
def fast_pdf_leaky_box(p, sigma, griddata):
    logp = np.log10(p)
    lnpdfs, inputs = griddata
    feh = inputs[-1]
    lnpdf = interpolate.interpn(inputs[:-1], lnpdfs, [logp,sigma], bounds_error=False, fill_value=-999)
    return feh, np.exp(lnpdf[0])

def fast_loglkhd_pre_enriched_box(theta, fehdata, efehdata, griddata):
    N = len(fehdata)
    logp = np.log10(theta[0])
    feh0 = theta[1]
    xp = np.zeros((N, 4))
    xp[:,0] = logp
    xp[:,1] = feh0
    xp[:,2] = efehdata
    xp[:,3] = fehdata
    lnpdfs, inputs = griddata
    lnp = interpolate.interpn(inputs, lnpdfs, xp, bounds_error=False, fill_value=-999)
    return np.sum(lnp)
def fast_pdf_pre_enriched_box(p, feh0, sigma, griddata):
    logp = np.log10(p)
    lnpdfs, inputs = griddata
    feh = inputs[-1]
    lnpdf = interpolate.interpn(inputs[:-1], lnpdfs, [logp,feh0,sigma], bounds_error=False, fill_value=-999)
    return feh, np.exp(lnpdf[0])

def fast_loglkhd_extra_gas(theta, fehdata, efehdata, griddata):
    N = len(fehdata)
    logp = np.log10(theta[0])
    M = theta[1]
    xp = np.zeros((N, 4))
    xp[:,0] = logp
    xp[:,1] = M
    xp[:,2] = efehdata
    xp[:,3] = fehdata
    lnpdfs, inputs = griddata
    lnp = interpolate.interpn(inputs, lnpdfs, xp, bounds_error=False, fill_value=-999)
    return np.sum(lnp)
def fast_pdf_extra_gas(p, M, sigma, griddata):
    logp = np.log10(p)
    lnpdfs, inputs = griddata
    feh = inputs[-1]
    lnpdf = interpolate.interpn(inputs[:-1], lnpdfs, [logp,M,sigma], bounds_error=False, fill_value=-999)
    return feh, np.exp(lnpdf[0])
