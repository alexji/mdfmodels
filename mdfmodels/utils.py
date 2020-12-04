import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import functools

from . import mdfmodels

import dynesty.plotting as dyplot

def resample(dsampler):
    res = dsampler.results
    dyplot.resample

def _sampler(N, invcdf):
    """ Function template for making random samplers """
    return invcdf(np.random.uniform(size=N))
def make_cdf(feh, pdf):
    print("Assuming feh is linearly spaced")
    dfe = feh[1]-feh[0]
    cdf = dfe * pdf.cumsum()
    if np.any(cdf >= 1):
        ix = np.min(np.where(cdf >= 1)[0])
        cdf[ix:] = 1
        print(f"Setting cdf to 1 starting at {ix}")
    else:
        cdf[-1] = 1
    if np.any(cdf < 0):
        ix = np.max(np.where(cdf < 0)[0])
        cdf[:ix] = 0
        print(f"Setting cdf to 0 up to {ix}")
    elif np.all(cdf > 0):
        cdf[0] = 0
        print(f"Setting cdf first point to 0")        
    return cdf
def make_invcdf(feh, pdf):
    cdf = make_cdf(feh, pdf)
    invcdf = interpolate.interp1d(cdf, feh)
    return invcdf
def make_random_sampler(feh, pdf):
    invcdf = make_invcdf(feh, pdf)
    return functools.partial(_sampler, invcdf=invcdf)

def make_sampler_leaky_box(p, fehmin=-5, fehmax=1, fehprec=0.005):
    feh = np.arange(fehmin, fehmax+2*fehprec, fehprec)
    pdf = mdfmodels.leaky_box(feh, p)
    return make_random_sampler(feh, pdf)
def make_sampler_pre_enriched_box(p, feh0, fehmin=-5, fehmax=1, fehprec=0.005):
    feh = np.arange(fehmin, fehmax+2*fehprec, fehprec)
    pdf = mdfmodels.pre_enriched_box(feh, p, feh0)
    return make_random_sampler(feh, pdf)
def make_sampler_extra_gas(p, M, fehmin=-5, fehmax=1, fehprec=0.005):
    feh = np.arange(fehmin, fehmax+2*fehprec, fehprec)
    pdf = mdfmodels.extra_gas(feh, p, M)
    return make_random_sampler(feh, pdf)

def sample_leaky_box(N, p):
    sampler = make_sampler_leaky_box(p)
    return sampler(N)
def sample_pre_enriched_box(N, p, feh0):
    sampler = make_sampler_pre_enriched_box(p, feh0)
    return sampler(N)
def sample_extra_gas(N, p, M):
    sampler = make_sampler_pre_enriched_box(p, M)
    return sampler(N)

