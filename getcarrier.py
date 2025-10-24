import numpy as np
from utils.kvec import kvec
class Carrier:
    def __init__(self, k, krad, mask, ccsgn):
          self.k = k # k vector
          self.krad = krad # filtering radius
          self.mask = mask # mask for filtering in fourier domain
          self.ccsgn = ccsgn # complex conjugate of signal


def getcarrier(fftIref, kc, krad):
    if not isinstance(fftIref, np.ndarray):
        raise TypeError("Should be a numpy array")

    rows, cols = fftIref.shape
    kx, ky = np.meshgrid(kvec(cols), kvec(rows))
    mask = 1 * ((kx - kc[0])**2 + (ky-kc[1])**2 < krad**2)
    ccsgn = np.conj(np.fft.ifft2(fftIref* mask))
    print(ccsgn)

    return Carrier(kc, krad, mask, ccsgn)