import numpy as np
import sys
sys.path.insert(1, 'utils' )

from utils.kvec import kvec

from utils.fftinvgrad import fftinvgrad

from utils.designgrad import designgrad # this function is not working properly 

def unwrap2(ph_w, roi=None):
    """
    Unwraps a 2D phase map

    Args:
    ph_w : 2D wrapped phase map (phase map with 2*pi discontinuities)
    roi : (optional) binary region of interest mask with the same shape as ph_w. Where True = valid regions, false = masked region

    Returns:
    ph : Unwrapped phase map

    """

    if roi is not None:
        hasroi = True
        if roi.shape != ph_w.shape:
            raise ValueError("ROI must have the same shape as the phase map.")

    else:
        hasroi = False

    Nx = ph_w.shape[1]
    Ny = ph_w.shape[0]
    
    if hasroi:
        ph = ph_w.copy()  
        ph_w = ph_w[roi]

    # auxiliary complex function without phase jumps
    Z = np.exp(1j * ph_w)

    # Calculate the derivatives:
    if hasroi:
        # uses central difference method
        [dx, dy] = designgrad(roi)
        phx_w = dx @ ph_w
        phy_w = dy @ ph_w
        phx = dx @ Z
        phy = dy @ Z
    else:
        # Performs differentation in the fourier domain
        [KX, KY] = np.meshgrid(kvec(Nx), kvec(Ny))
        fph_w = np.fft.fft2(ph_w)
        phx_w = np.fft.ifft2(1j * KX * fph_w).real
        phy_w = np.fft.ifft2(1j * KY * fph_w).real
        fZ = np.fft.fft2(Z)
        phx = np.fft.ifft2(1j * KX * fZ)
        phy = np.fft.ifft2(1j * KY * fZ)

    phx = (phx / (1j * Z)).real
    phy = (phy / (1j * Z)).real
    
    jx = phx - phx_w
    jy = phy - phy_w

    if hasroi:
        rhs = np.concatenate((jx.ravel(), jy.ravel()))
        lhs = np.vstack((dx[:, 1:], dy[:, 1:]))
        j = np.concatenate(([0], np.linalg.lstsq(lhs, rhs, rcond=None)[0]))
    else:
        j = fftinvgrad(jx, jy, gradType='spectral', bcFix='none')
        j_rows, j_cols = j.shape
        j = j - j[(j_rows - 1)//2, (j_cols - 1)//2]
    

    # Ensures correctiion is an interger number of 2pi with most values being at 0
    j = np.round(j/(2 * np.pi))

    # start counting from 1
    min_j = j.min()
    j = j - min_j + 1

    # count the number of occurences of each jump value
    # Flatten j and count occurrences
    values, counts = np.unique(j, return_counts=True)

    # Find the most frequent value
    counts = np.bincount(j.astype(int).ravel())
    mxcntj = np.argmax(counts)   # index, like MATLAB
    j = (j - mxcntj) * 2 * np.pi

    # unwrap phase
    if hasroi:
        ph[roi] = ph_w + j
    else:
        ph = ph_w + j

    return ph
