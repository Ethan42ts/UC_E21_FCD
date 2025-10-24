import numpy as np
from kvec import kvec

def fftinvgrad(fx, fy, bcFix = 'impulse', gradType ='spectral'):
    """
    Args:
        fx: Gradient across columns (left to right)
        fy: Gradient across rows (top to bottom)
        bcFix:  Adds a special boundray correction function on the boundraies of the domain to correct for 
                    periodicity assumed in fft
                Can be either {'impulse', 'mirror', or 'none'}. 
                Default is 'impulse'
        gradType:   Specifies how the gradient is formed. 
                        Spectral means fx and fy is assumed to come from an analytical method
                        Difference assumes  fx and fy comes from a central difference gradient
                    Can be either {'spectral', 'difference'}
                    Default is 'spectral'
    Returns:
        f: Integradted gradient field such that [fx, fy] = grad(f) in a least squares sense

    """

    impbc = (bcFix.lower() == 'impulse') # impulse boundry correction
    mirbc = (bcFix.lower() == 'mirror') # mirror boundry correction

    finitediff = (gradType.lower() == 'difference') # Finite difference is using central difference method
    if mirbc:
        fx = np.block([[fx,             -np.fliplr(fx)], 
                       [np.flipud(fx),  -np.rot90(fx, 2)]])
        fy = np.block([[fy,             np.fliplr(fy)],
                       [-np.flipud(fy), -np.rot90(fy, 2)]])


    Ny, Nx = fx.shape

    if impbc:
        # breakpoint()
        impIx = -0.5 * np.sum(fx[:,1:-1], 1)
        # impIx = impIx.reshape((np.size(impIx), 1))
        impIy = -0.5 * np.sum(fy[1:-1,:], 0)
        fx_edge = fx[:,[0,-1]].copy()
        fy_edge = fy[[0, -1], :].copy()
        fx[:, [0,-1]] = np.block([[impIx], [impIx]]).T 
        fy[[0,-1],:] = np.block([[impIy], [impIy]])

    mx = np.mean(fx)
    my = np.mean(fy)

    if finitediff:
        kx, ky = np.meshgrid(np.sin(kvec(Nx)), np.sin(kvec(Ny)))
    else:
        kx, ky = np.meshgrid(kvec(Nx), kvec(Ny))
    
    k2 = kx**2 + ky**2

    if Nx%2 == 0:
        kx[:, Nx//2] = 0
    
    if Ny%2 == 0:
        ky[Ny//2, :] = 0

    fx_hat = np.fft.fft2(fx)
    fy_hat = np.fft.fft2(fy)

    k2[k2< np.finfo(float).eps] = 1 # Avoids division by zero
    f_hat = (-1j * kx * fx_hat -1j * ky * fy_hat) / k2
    f = np.fft.ifft2(f_hat).real

    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))

    f = f + mx * x + my * y

    if impbc:
        f[:, 0] = (4 * f[:,1] - f[:,2] - 2*fx_edge[:,0]) / 3
        f[:, -1] = (4 * f[:,-2] - f[:, -3] + 2 * fx_edge[:,1]) /3
        f[0,:] = (4 * f[1,:] - f[2,:] - 2*fy_edge[0,:]) / 3
        f[-1,:] = (4 * f[-2,:] - f[-3,:] + 2 * fy_edge[1, :]) / 3
        # breakpoint()
    
    elif mirbc:
        f = f[1:Ny/2, 1:Nx/2]
    return f

