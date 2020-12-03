import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import functools

from . import mdfmodels

import dynesty as dy
from dynesty import plotting as dyplot
from schwimmbad import MultiPool

"""
TODO: figure out how to deal with error bars.
Do I just have to hierarchical inference it?
"""

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
    return make_random_sampler(feharr, pdf)
def make_sampler_pre_enriched_box(p, feh0, fehmin=-5, fehmax=1, fehprec=0.005):
    feh = np.arange(fehmin, fehmax+2*fehprec, fehprec)
    pdf = mdfmodels.pre_enriched_box(feh, p, feh0)
    return make_random_sampler(feharr, pdf)
def make_sampler_extra_gas(p, M, fehmin=-5, fehmax=1, fehprec=0.005):
    feh = np.arange(fehmin, fehmax+2*fehprec, fehprec)
    pdf = mdfmodels.extra_gas(feh, p, M)
    return make_random_sampler(feharr, pdf)

def fit_leaky_box(fehdata, logpmin=-3, logpmax=-1,
                  pool=None, ptform=None, **run_nested_kwargs):
    def lnlkhd(theta):
        p = theta[0]
        lnp = mdfmodels.log_leaky_box(fehdata, p)
        return np.sum(lnp)
    if ptform is None:
        def ptform(u):
            return 10**((logpmax-logpmin)*u + logpmin)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=1, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler
def fit_pre_enriched_box(fehdata,
                         logpmin=-3, logpmax=-1,
                         feh0min=-5, feh0max=-2
                         pool=None, ptform=None, **run_nested_kwargs):
    def lnlkhd(theta):
        p, feh0 = theta[0]
        lnp = mdfmodels.log_pre_enriched_box(fehdata, p, feh0)
        return np.sum(lnp)
    if ptform is None:
        def ptform(u):
            u[0] = 10**((logpmax-logpmin)*u + logpmin)
            u[1] = (feh0max-feh0min)*u + feh0min
            return u
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=2, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler
def fit_extra_gas(fehdata,
                  logpmin=-3, logpmax=-1,
                  Mmin=1, Mmax=10,
                  pool=None, ptform=None, **run_nested_kwargs):
    def lnlkhd(theta):
        p, M = theta[0]
        lnp = mdfmodels.log_extra_gas(fehdata, p, M)
        return np.sum(lnp)
    if ptform is None:
        def ptform(u):
            u[0] = 10**((logpmax-logpmin)*u + logpmin)
            u[1] = (Mmax-Mmin)*u + Mmin
            return u
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=2, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler
