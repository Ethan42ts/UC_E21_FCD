import numpy as np
import scipy.signal.windows as sigWin

from getcarrier import Carrier, getcarrier
from findorthcarrierpks import findorthcarrierpks
from check_isclose import save_array
from utils.kvec import kvec
from fcd_dispfield import fcd_dispfield

zero_2x2 = np.zeros((2,2))
ones_2x2 = np.ones((2,2))
base_checkerboard = np.hstack((np.hstack((ones_2x2, zero_2x2)).T, np.flip(np.hstack((ones_2x2, zero_2x2))).T))
Iref = np.tile(base_checkerboard, (4,5))
Iref[0:4,0:4] = 1

Idef = np.roll(np.tile(base_checkerboard, (4,5)), shift=( -1, 1 ), axis=(0, 1))
Idef[0:4,0:4] = 1

rows, cols = Iref.shape

kr, ku = findorthcarrierpks(Iref, 4 * np.pi /np.min(Iref.shape), np.inf)
krad = np.sqrt(np.sum((kr-ku)**2))/2
fIref = np.fft.fft2(Iref)

cr = getcarrier(fIref, kr, krad)
cu = getcarrier(fIref, ku, krad)

kxvec = np.fft.fftshift(kvec(cols))
kyvec = np.fft.fftshift(kvec(rows))
wr = sigWin.hann(rows, sym = False)
wc = sigWin.hann(cols, sym = False)
win2d = np.outer(wr, wc)

fftIm = np.fft.fftshift(np.abs(np.fft.fft2((Iref-np.mean(Iref)) * win2d)))

# iFFT =
u, v = fcd_dispfield(np.fft.fft2(Idef), cr, cu, False)

nrmU = np.sqrt(u**2 + v**2)

# print(nrmU)