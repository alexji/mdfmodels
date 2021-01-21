import numpy as np
from scipy import special, interpolate, integrate, optimize, stats
from scipy.ndimage import gaussian_filter
import warnings

def convolve_pdf(feh, pdf, sigma):
    dfeh = feh[1]-feh[0]
    assert np.all(np.diff(feh) == dfeh)
    sigma_pix = sigma/dfeh
    return gaussian_filter(pdf, sigma_pix, mode="nearest")
    
def leaky_box(feh, p):
    """
    p = p_eff
    """
    z = 10**feh/p 
    return np.log(10) * z * np.exp(-z)

def log_leaky_box(feh, p):
    z = 10**feh/p 
    return np.log(np.log(10)) + np.log(z) - z

def pre_enriched_box(feh, p, feh0):
    """
    p = p_eff
    feh0 is the starting metallicity
    """
    z = (10**feh)/p
    z0 = (10**feh0)/p
    # I analytically integrated this to a special function and am very proud :)
    out = np.log(10) * (z-z0) * np.exp(-z) / (np.exp(-z0) - z0 * (-special.expi(-z0)))
    out[feh <= feh0] = 0.
    return out

def gaussian(feh, mu, sigma):
    return stats.norm.pdf(feh, loc=mu, scale=sigma)
def log_gaussian(feh, mu, sigma):
    return stats.norm.logpdf(feh, loc=mu, scale=sigma)

def log_pre_enriched_box(feh, p, feh0):
    z = 10**feh/p 
    z0 = (10**feh0)/p
    out = np.log(np.log(10)) + np.log(z-z0) - z - np.log(np.exp(-z0) - z0 * (-special.expi(-z0)))
    out[feh < feh0] = -np.inf
    return out

def _extra_gas_feh(s, p, M):
    sM = s/M
    return np.log10(p) + 2*np.log10(M) - 2*np.log10(1+s-sM) + np.log10(-np.log(1-sM) - sM * (1-1/M))
def _extra_gas_dndfeh(s, feh, p, M):
    """ Not normalized! """
    z = (10**feh)/p
    return z * (1 + s*(1-1/M)) / (1/(1-s/M) - 2*z*(1-1/M))
def _extra_gas_log_dndfeh(s, feh, p, M):
    """ Not normalized! """
    z = (10**feh)/p
    return np.log(z) + np.log((1 + s*(1-1/M))) - np.log(1/(1-s/M) - 2*z*(1-1/M))
def _extra_gas_solve_for_s(feh, p, M, get_all=False):
    _func = lambda s: feh - _extra_gas_feh(s, p, M)
    ## Bracketing here is the key to get the numerics to work out
    opt = optimize.root_scalar(_func, bracket=[0,M])
    if not opt.converged: warnings.warn("_extra_gas_solve_for_s: not converged!")
    if get_all: return opt
    return opt.root
def _extra_gas_compute_func_norm(p, M, fehmin=-5, fehmax=1, fehprec=0.005,
                                 get_all=False):
    _feharr = np.arange(fehmin, fehmax+2*fehprec, fehprec)
    _sarr = np.array([_extra_gas_solve_for_s(x, p, M) for x in _feharr])
    sfunc = interpolate.interp1d(_feharr, _sarr)
    _dndfeh = _extra_gas_dndfeh(_sarr, _feharr, p, M)
    norm = integrate.simps(_dndfeh, _feharr)
    if get_all: return sfunc, norm, _feharr, _sarr, _dndfeh
    return sfunc, norm
def extra_gas(feh, p, M, fehmin=-5, fehmax=1, fehprec=0.005,
              sfunc=None, norm=None, get_all=False):
    feh = np.ravel(feh)
    if np.any(feh) < fehmin: raise ValueError(f"Some feh is < than fehmin={fehmin}")
    if np.any(feh) >= fehmax+fehprec: raise ValueError(f"Some feh is >= than fehmax={fehmax}")
    
    if sfunc is None or norm is None:
        sfunc, norm = _extra_gas_compute_func_norm(p, M, fehmin=fehmin, fehmax=fehmax, fehprec=fehprec)
    
    s = sfunc(feh)
    dndfeh = _extra_gas_dndfeh(s, feh, p, M)/norm
    
    if get_all: return dndfeh, sfunc, norm
    return dndfeh

def log_extra_gas(feh, p, M, fehmin=-5, fehmax=1, fehprec=0.005,
              sfunc=None, norm=None, get_all=False):
    feh = np.ravel(feh)
    if np.any(feh) < fehmin: raise ValueError(f"Some feh is < than fehmin={fehmin}")
    if np.any(feh) >= fehmax+fehprec: raise ValueError(f"Some feh is >= than fehmax={fehmax}")
    
    if sfunc is None or norm is None:
        sfunc, norm = _extra_gas_compute_func_norm(p, M, fehmin=fehmin, fehmax=fehmax, fehprec=fehprec)
    
    s = sfunc(feh)
    logdndfeh = _extra_gas_log_dndfeh(s, feh, p, M) - np.log(norm)
    
    if get_all: return logdndfeh, sfunc, norm
    return logdndfeh

