import numpy as np
import time, sys

from mdfmodels import mdfmodels
from schwimmbad import MultiPool
import functools

grid_logp = np.arange(-4,0.01,0.025)
grid_feh0 = np.arange(-4,0.01,0.025)
grid_M    = np.arange(1,30.1,0.2)
grid_sigma = np.arange(0.00,0.51,0.025)
feh = np.arange(-5,1,0.005)

nproc = 8

def run_func(sigma, feh, pdf):
    return mdfmodels.convolve_pdf(feh, pdf, sigma)
if __name__=="__main__":
    print(f"logp: {grid_logp.size}")
    print(f"feh0: {grid_feh0.size}")
    print(f"M: {grid_M.size}")
    print(f"sigma: {grid_sigma.size}")
    
def run_leaky_box():
    print("Running leaky box")
    start = time.time()
    leaky_box_pdfs = np.zeros((grid_logp.size, grid_sigma.size, feh.size))
    with MultiPool(nproc) as pool:
        for i,logp in enumerate(grid_logp):
            pdf = mdfmodels.leaky_box(feh, 10**logp)
            #for k,sigma in enumerate(grid_sigma):
            #    leaky_box_pdfs[i,k] = mdfmodels.convolve_pdf(feh, pdf, sigma)
            func = functools.partial(run_func, feh=feh, pdf=pdf)
            leaky_box_pdfs[i] = pool.map(func, grid_sigma)
            if i % 10 == 0: sys.stdout.write(f"\rCumtime {i} {time.time()-start:.1f}s")
    print(f"\nLeaky box took {time.time()-start:.1f}s")
    np.save("leaky_box_pdfs.npy",leaky_box_pdfs)
    np.save("leaky_box_other.npy",[grid_logp, grid_sigma, feh])
    
def run_pre_enriched_box():
#if __name__=="__main__":
    print("Running pre-enriched box")
    start = time.time()
    pre_enriched_box_pdfs = np.zeros((grid_logp.size, grid_feh0.size, grid_sigma.size, feh.size))
    with MultiPool(nproc) as pool:
        for i,logp in enumerate(grid_logp):
            for j,feh0 in enumerate(grid_feh0):
                pdf = mdfmodels.pre_enriched_box(feh, 10**logp, feh0)
                #for k,sigma in enumerate(grid_sigma):
                #    pre_enriched_box_pdfs[i,j,k] = mdfmodels.convolve_pdf(feh, pdf, sigma)
                func = functools.partial(run_func, feh=feh, pdf=pdf)
                pre_enriched_box_pdfs[i,j] = pool.map(func, grid_sigma)
                if j % 10 == 0: sys.stdout.write(f"\rCumtime {i} {time.time()-start:.1f}s")
            if i % 10 == 0: sys.stdout.write(f"\rCumtime {i} {time.time()-start:.1f}s")
    np.save("pre_enriched_box_pdfs.npy",pre_enriched_box_pdfs)
    np.save("pre_enriched_box_other.npy",[grid_logp, grid_feh0, grid_sigma, feh])
    print(f"\nPre enriched box took {time.time()-start:.1f}s")
    
#def run_extra_gas():
if __name__=="__main__":
    print("Running extra gas")
    start = time.time()
    extra_gas_pdfs = np.zeros((grid_logp.size, grid_M.size, grid_sigma.size, feh.size))
    with MultiPool(nproc) as pool:
        for i,logp in enumerate(grid_logp):
            for j,M in enumerate(grid_M):
                pdf = mdfmodels.extra_gas(feh, 10**logp, M)
                #for k,sigma in enumerate(grid_sigma):
                #    extra_gas_pdfs[i,j,k] = mdfmodels.convolve_pdf(feh, pdf, sigma)
                func = functools.partial(run_func, feh=feh, pdf=pdf)
                extra_gas_pdfs[i,j] = pool.map(func, grid_sigma)
                if j % 10 == 0: sys.stdout.write(f"\rCumtime {i}-{j} {time.time()-start:.1f}s")
            if i % 10 == 0: sys.stdout.write(f"\rCumtime {i} {time.time()-start:.1f}s")
    np.save("extra_gas_pdfs.npy",extra_gas_pdfs)
    np.save("extra_gas_other.npy",[grid_logp, grid_M, grid_sigma, feh])
    print(f"\nExtra gas took {time.time()-start:.1f}s")
    
