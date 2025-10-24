from unwrap import unwrap2
from getcarrier import Carrier
from check_isclose import save_array

from fcd_phasefield import fcd_phasefield
from phase_to_displacement import phase_to_displacement

def fcd_dispfield(fft_Idef, cr, cu, try_unwrap = False):
    """
    Finds the displacement field that wraps a periodic reference image (Iref) into the deformed image (Idef)
    
    Parameters:
    fft_Idef : fft2(Idef), where Idef is a distorted reference pattern of the approximate form I(r) = c0 +cos(cr.k * r) +cos(cu.k * r)
    cr, cu : orthogonal carrier signals extracted from an undistorted reference image
    try_unwrap : boolean, if true, calls unwrap2 on the extracted phase fields 
    
    Returns:
    u, v : local displacement fields in x and y directions respectively. This allows Idef = Iref(x - u, y - v) 

    """
    if (isinstance(cr, Carrier) == False) and (isinstance(cu, Carrier) == False):
        raise("fcd_dispfield cr and cu should be of instance Carrier")
    
    # phase signals in x and y directions
    phi_r = fcd_phasefield(fft_Idef, cr)
    phi_u = fcd_phasefield(fft_Idef, cu)

    if try_unwrap:
        # Wrapping around 2pi
        phi_r = unwrap2(phi_r)
        phi_u = unwrap2(phi_u)

    # Displacements in x and y directions
    (u, v) = phase_to_displacement(phi_r, phi_u, cr, cu)

    return (u, v)
