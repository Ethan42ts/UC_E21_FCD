import numpy as np
from getcarrier import Carrier 

from check_isclose import save_array

def fcd_phasefield(fftIdef, car):
    """
    Extract the phase modulation signal for a given carrier peak
    Args:
        fftIdef: fft2 of Idef. Idef is Idef is a distorted reference pattern
        car: Carrier signal extracted from an undistorted reference image

    Return:
        phi: extracted phase signal
    """
    if (isinstance(car, Carrier) == False):
        raise("Car should be of instance Carrier")

    # Apply filter in k-space, selecting the signal around the carrier peak
    fftIdef_filtered = fftIdef * car.mask

    # Transform back to spatial domain
    f_flt = np.fft.ifft2(fftIdef_filtered)

    # Extract modulation of carrier signal
    phi = -np.angle(f_flt * car.ccsgn)

    return phi