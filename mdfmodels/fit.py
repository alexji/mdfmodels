import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import functools

from . import mdfmodels, fast_mdfmodels

import dynesty as dy
from dynesty import plotting as dyplot

"""
TODO: figure out how to deal with error bars.
Do I just have to hierarchical inference it?
"""

def ptform_leaky_box(u, logpmin, logpmax):
    return 10**((logpmax-logpmin)*u + logpmin)
def lnlkhd_leaky_box(theta, fehdata):
    p = theta[0]
    lnp = mdfmodels.log_leaky_box(fehdata, p)
    return np.sum(lnp)
def fit_leaky_box(fehdata, logpmin=-3, logpmax=-1,
                  pool=None, ptform=None, **run_nested_kwargs):
    lnlkhd = functools.partial(lnlkhd_leaky_box, fehdata=fehdata)
    if ptform is None:
        ptform = functools.partial(ptform_leaky_box, logpmin=logpmin, logpmax=logpmax)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=1, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler

def lnlkhd_pre_enriched_box(theta, fehdata):
    p, feh0 = theta
    lnp = mdfmodels.log_pre_enriched_box(fehdata, p, feh0)
    return np.sum(lnp)
def ptform_pre_enriched_box(u, logpmin, logpmax, feh0min, feh0max):
    u[0] = 10**((logpmax-logpmin)*u[0] + logpmin)
    u[1] = (feh0max-feh0min)*u[1] + feh0min
    return u
def fit_pre_enriched_box(fehdata,
                         logpmin=-3, logpmax=-1,
                         feh0min=-5, feh0max=-2,
                         pool=None, ptform=None, **run_nested_kwargs):
    lnlkhd = functools.partial(lnlkhd_pre_enriched_box, fehdata=fehdata)
    if ptform is None:
        ptform = functools.partial(ptform_pre_enriched_box, logpmin=logpmin, logpmax=logpmax,
                                   feh0min=feh0min, feh0max=feh0max)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=2, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler

def lnlkhd_extra_gas(theta, fehdata):
    p, M = theta
    lnp = mdfmodels.log_extra_gas(fehdata, p, M)
    return np.sum(lnp)
def ptform_extra_gas(u, logpmin, logpmax, Mmin, Mmax):
    u[0] = 10**((logpmax-logpmin)*u[0] + logpmin)
    u[1] = (Mmax-Mmin)*u[1] + Mmin
    return u
def fit_extra_gas(fehdata,
                  logpmin=-3, logpmax=-1,
                  Mmin=1, Mmax=10,
                  pool=None, ptform=None, **run_nested_kwargs):
    lnlkhd = functools.partial(lnlkhd_extra_gas, fehdata=fehdata)
    if ptform is None:
        ptform = functools.partial(ptform_extra_gas, logpmin=logpmin, logpmax=logpmax,
                                   Mmin=Mmin, Mmax=Mmax)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=2, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler

def fit_leaky_box_errors(fehdata, efehdata,
                         logpmin=-3, logpmax=-1,
                         pool=None, ptform=None, **run_nested_kwargs):
    griddata = fast_mdfmodels.load_data("leaky_box")
    lnlkhd = functools.partial(fast_mdfmodels.fast_loglkhd_leaky_box,
                               fehdata=fehdata, efehdata=efehdata,
                               griddata=griddata)
    if ptform is None:
        ptform = functools.partial(ptform_leaky_box, logpmin=logpmin, logpmax=logpmax)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=1, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler
def fit_pre_enriched_box_errors(fehdata, efehdata,
                                logpmin=-3, logpmax=-1,
                                feh0min=-5, feh0max=-2,
                                pool=None, ptform=None, **run_nested_kwargs):
    griddata = fast_mdfmodels.load_data("pre_enriched_box")
    lnlkhd = functools.partial(fast_mdfmodels.fast_loglkhd_pre_enriched_box,
                               fehdata=fehdata, efehdata=efehdata,
                               griddata=griddata)
    if ptform is None:
        ptform = functools.partial(ptform_pre_enriched_box, logpmin=logpmin, logpmax=logpmax, feh0min=feh0min, feh0max=feh0max)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=2, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler
def fit_extra_gas_errors(fehdata, efehdata,
                         logpmin=-3, logpmax=-1,
                         Mmin=1, Mmax=10,
                         pool=None, ptform=None, **run_nested_kwargs):
    griddata = fast_mdfmodels.load_data("extra_gas")
    lnlkhd = functools.partial(fast_mdfmodels.fast_loglkhd_extra_gas,
                               fehdata=fehdata, efehdata=efehdata,
                               griddata=griddata)
    if ptform is None:
        ptform = functools.partial(ptform_extra_gas, logpmin=logpmin, logpmax=logpmax, Mmin=Mmin, Mmax=Mmax)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=2, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler

def ptform_gaussian(u, mumin, mumax, logsigmamin, logsigmamax):
    u[0] = (mumax-mumin)*u[0] + mumin
    u[1] = 10**((logsigmamax-logsigmamin)*u[1] + logsigmamin)
    return u
def fit_gaussian_errors(fehdata, efehdata,
                        mumin = -4, mumax = -1,
                        logsigmamin=-2, logsigmamax=1,
                        pool=None, ptform=None, **run_nested_kwargs):
    lnlkhd = functools.partial(fast_mdfmodels.fast_loglkhd_gaussian,
                               fehdata=fehdata, efehdata=efehdata)
    if ptform is None:
        ptform = functools.partial(ptform_gaussian, mumin=mumin, mumax=mumax,
                                   logsigmamin=logsigmamin, logsigmamax=logsigmamax)
    dsampler = dy.DynamicNestedSampler(lnlkhd, ptform, ndim=2, pool=pool)
    dsampler.run_nested(**run_nested_kwargs)
    return dsampler
