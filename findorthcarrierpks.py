import numpy as np
import scipy.signal.windows as sigWin

from utils.kvec import kvec 
from utils.findpeaks2 import findpeaks2

from check_isclose import save_array

def findorthcarrierpks(Iref, kmin, kmax, thresh = None, usehamming = False):
    """
    Args:
        Iref: 2D periodic reference pattern 
        kmin, kmax: Search range to kmin < |k| <kmax
        thresh: Threshold value for peak detection as a  fraction of the maximum in the search range

    Return:
        kr, ku: locations of the two peaks in k-space that maximses the normalised cross product kr x ku/|ku| among the four strongest peaks detected within the range

    """
    Iref = Iref.astype(np.float64)
    rows, cols = Iref.shape

    Iref = Iref - np.mean(Iref)

    if ((usehamming) or (thresh is None)) :
        wr = sigWin.hamming(rows, False)
        wc = sigWin.hamming(cols, False)

        w2d = np.outer(wr, wc)

        f_h = np.abs(np.fft.fftshift(np.fft.fft2(Iref * w2d)))
    else:
        f_h = abs(np.fft.fftshift(np.fft.fft2(Iref)))

    kx = np.fft.fftshift(kvec(cols))
    ky = np.fft.fftshift(kvec(rows))
    kxgrid, kygrid = np.meshgrid(kx, ky)

    k2 = kxgrid **2 + kygrid **2

    f_h[(k2>kmax**2) | (k2<=kmin**2)] = 0

    if thresh is None:
        thresh = 0.5 * np.max(f_h.ravel())
    else:
        thresh = thresh * np.max(f_h.ravel())
    
    peakrows, peakcols, _ = findpeaks2(f_h, thresh, 4, True)

    if ((peakrows.size < 4) or (np.any(np.isnan(peakrows))) or (np.any(np.isnan(peakrows)))):
        raise ReferenceError("Could not detect carrier signal")
    
    f1_peakrows = np.floor(peakrows)
    f1_peakcols = np.floor(peakcols)

    alphar = peakrows - f1_peakrows
    alphac = peakcols - f1_peakcols
    kyPeaks = (1 - alphar) * ky[f1_peakrows.astype(np.int64)] + alphar * ky[f1_peakrows.astype(np.int64) + 1]
    kxPeaks = (1 - alphac) * kx[f1_peakcols.astype(np.int64)] + alphac * kx[f1_peakcols.astype(np.int64) + 1]

    peakAngles = np.arctan2(kyPeaks, kxPeaks)

    krInd = np.argmin(np.abs(peakAngles))
    kr = np.array([kxPeaks[krInd], kyPeaks[krInd]])

    kuInd = np.argmax((kr[0] * kyPeaks - kr[1] * kxPeaks / np.sqrt(kxPeaks **2 + kyPeaks **2)))
    ku = [kxPeaks[kuInd], kyPeaks[kuInd]]

    return kr, ku