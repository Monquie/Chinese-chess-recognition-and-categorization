import numpy as np
from scipy import stats as st

def gaussian_filter(sigma): 
    size = 2*np.ceil((2**(1/2))*sigma)+1 
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()

def gkern(kernlen, nsig):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

print(gaussian_filter(1.6))
print("shit happened")
print(gkern(7,1.6))
print("shit fucker")
print(gkern(7,1.6)-gaussian_filter(1.6))
