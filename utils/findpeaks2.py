import numpy as np
import cv2

def findpeaks2(A: np.array, thresh, n, subpixel = True):
    rows, cols = A.shape

    A_masked = (A > thresh).astype(np.uint8)
    numObjects, labels, stats, centroids = cv2.connectedComponentsWithStats(A_masked, 8, cv2.CV_32S)

    vals = np.zeros((1, numObjects-1))
    inds = np.zeros((1, numObjects-1), dtype=np.int64)

    for i in range(1, numObjects):  # skip background (label 0)
        ids = np.flatnonzero(labels == i)  
        regionalMaxId = np.argmax(A.flat[ids])
        vals[0, i-1] = A.flat[ids[regionalMaxId]]
        inds[0, i-1] = ids[regionalMaxId]

    # convert linear indices back to subscripts (r, c)
    r, c = np.unravel_index(inds, (rows, cols))

    # Remove border peaks
    borderpks = (r == 0) | (r == rows-1) | (c == 0) | (c == cols-1)
    r = r[~borderpks]
    c = c[~borderpks]
    vals = vals[~borderpks]
    inds = inds[~borderpks]

    # Sort peaks by height descending
    order = np.argsort(-vals, kind='stable')
    r = r[order]
    c = c[order]
    vals = vals[order]
    inds = inds[order]

    # Keep top n peaks
    if n < len(inds):
        r = r[:n]
        c = c[:n]
        vals = vals[:n]
        inds = inds[:n]

    # Subpixel Gaussian peak fitting
    if subpixel:
        
        lv = A.flat[inds - rows]   # left neighbor
        rv = A.flat[inds + rows]   # right neighbor
        tv = A.flat[inds - 1]      # top neighbor
        bv = A.flat[inds + 1]      # bottom neighbor

        def subpxpeak(ii, iiv, lv, rv):
            return ii - 0.5 * ((np.log(rv) - np.log(lv)) / (np.log(lv) + np.log(rv) - 2 * np.log(iiv)))

        c = subpxpeak(c, vals, lv, rv)
        r = subpxpeak(r, vals, tv, bv)

    return r, c, vals