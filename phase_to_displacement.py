from getcarrier import Carrier

def phase_to_displacement(phi_r, phi_u, cr, cu):
    if (isinstance(cr, Carrier) == False) and (isinstance(cu, Carrier) == False):
        raise("phase_to_displacement() cr and cu should be of instance Carrier")
    
    detA = cr.k[0] * cu.k[1] - cr.k[1] * cu.k[0]
    u = (cu.k[1] * phi_r - cr.k[1] * phi_u) / detA
    v = (cr.k[0] * phi_u - cu.k[0] * phi_r) / detA

    return u, v