import numpy as np
def kvec(N):
    if N%2 == 0:
        k = np.concatenate((np.arange(0,N/2), np.arange(N/2-N, 0)))

    else:
        k = np.concatenate((np.arange(0, (N-1)/2 + 1), np.arange((N+1)/2 - N, 0)))
    
    k = k * 2 * np.pi /N
    return k
