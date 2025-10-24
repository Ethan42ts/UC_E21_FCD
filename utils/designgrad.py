import numpy as np
np.set_printoptions(threshold=np.inf)
 
from scipy import sparse
from scipy.ndimage import binary_dilation

def designgrad(roi):
    """
    Generate sparse matrices Dx, Dy for central differences in x and y.
    Generated with the help of GPT V5
    
    Parameters
    ----------
    roi : array_like
        Either:
          - a tuple/list (Ny, Nx) giving the image shape (full ROI),
          - or a boolean mask of shape (Ny, Nx) indicating the region of interest.
    
    Returns
    -------
    Dx, Dy : scipy.sparse.csr_matrix
        Sparse derivative operators acting on the unrolled ROI pixels.
        For a flattened array I[roi], Dx @ I[roi] and Dy @ I[roi]
        give the gradients in x and y respectively.
    """
    # Handle input
    roi = np.array(roi, dtype=bool)
    if roi.ndim == 1 and roi.size == 2:
        roi = np.ones(roi, dtype=bool)
    if roi.ndim != 2:
        raise ValueError("roi must be a 2D mask or image shape")

    # Pad mask with False to simplify boundary handling
    roi = np.pad(roi, 1, constant_values=False)
    IRows, ICols = roi.shape

    N = roi.sum()  # number of valid pixels
    # Linear index mapping from padded ROI -> [1..N]
    linInd = np.zeros_like(roi, dtype=int)
    linInd[roi] = np.arange(1, N+1).reshape(roi.T.sum(), order="F")
    
    # print(linInd)

    # Allocate triplets for sparse matrix
    # rowDM = np.zeros((N, 3), dtype = 'double')
    rowDM = []
    colDM = []
    valDM = []
    # ---------- Horizontal derivative (Dx) ----------
    roiD = np.diff(roi.astype(int), axis=1)

    # Left edges (forward difference)
    pxind = np.argwhere(np.hstack([np.zeros((IRows,1)), roiD]) == 1)
    for r, c in pxind:
        i = linInd[r, c]
        j1 = linInd[r, c]
        j2 = linInd[r, c+1]
        j3 = linInd[r, c+2]
        rowDM.extend([i, i, i])
        colDM.extend([j1, j2, j3])
        valDM.extend([-3/2, 2.0, -0.5])
    # print(pxind)

    # Right edges (backward difference)
    pxind = np.argwhere(np.hstack([roiD, np.zeros((IRows,1))]) == -1)
    for r, c in pxind:
        i = linInd[r, c]
        j1 = linInd[r, c-2]
        j2 = linInd[r, c-1]
        j3 = linInd[r, c]
        rowDM.extend([i, i, i])
        colDM.extend([j1, j2, j3])
        valDM.extend([0.5, -2.0, 1.5])

    # Interior (central difference)
    interior = ~binary_dilation(~roi, structure=np.array([[1,1,1]]))
    pxind = np.argwhere(interior)
    for r, c in pxind:
        i = linInd[r, c]
        j1 = linInd[r, c-1]
        j2 = linInd[r, c]
        j3 = linInd[r, c+1]
        rowDM.extend([i, i, i])
        colDM.extend([j1, j2, j3])
        valDM.extend([-0.5, 0.0, 0.5])

    # print(np.shape(valDM))
    Dx = sparse.csr_matrix((valDM, (rowDM, colDM)), shape=(N+1, N+1))

    # ---------- Vertical derivative (Dy) ----------
    rowDM.clear(); colDM.clear(); valDM.clear()

    roiD = np.diff(roi.astype(int), axis=0)

    # Top edges
    pxind = np.argwhere(np.vstack([np.zeros((1,ICols)), roiD]) == 1)
    for r, c in pxind:
        i = linInd[r, c]
        j1 = linInd[r, c]
        j2 = linInd[r+1, c]
        j3 = linInd[r+2, c]
        rowDM.extend([i, i, i])
        colDM.extend([j1, j2, j3])
        valDM.extend([-3/2, 2.0, -0.5])

    # Bottom edges
    pxind = np.argwhere(np.vstack([roiD, np.zeros((1,ICols))]) == -1)
    for r, c in pxind:
        i = linInd[r, c]
        j1 = linInd[r-2, c]
        j2 = linInd[r-1, c]
        j3 = linInd[r, c]
        rowDM.extend([i, i, i])
        colDM.extend([j1, j2, j3])
        valDM.extend([0.5, -2.0, 1.5])

    # Interior
    interior = ~binary_dilation(~roi, structure=np.array([[1],[1],[1]]))
    pxind = np.argwhere(interior)
    for r, c in pxind:
        i = linInd[r, c]
        j1 = linInd[r-1, c]
        j2 = linInd[r, c]
        j3 = linInd[r+1, c]
        rowDM.extend([i, i, i])
        colDM.extend([j1, j2, j3])
        valDM.extend([-0.5, 0.0, 0.5])

    Dy = sparse.csr_matrix((valDM, (rowDM, colDM)), shape=(N+1, N+1))

    return Dx, Dy
