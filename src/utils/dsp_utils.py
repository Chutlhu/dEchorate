import numpy as np
import scipy as sp
import scipy.signal as sg

def normalize(x):
    return x/np.max(np.abs(x))

def center(x):
    return x - np.mean(x)

def envelope(x):
    return np.abs(sg.hilbert(x))


def make_toepliz_as_in_mulan(v, L):
    D = v.shape[0]
    T = np.zeros([D-L+1, L], dtype=np.complex64)
    R, _ = T.shape
    for r in range(R):
        T[r, :] = v[r:r+L][::-1]
    return T


def make_toepliz_as_in_mulan2(v, L):
    D = len(v)
    r1 = v[:L][::-1]
    c1 = v[L-1:]
    return sp.linalg.toeplitz(c1, r1)

def reconstruct_toeplitz(Ta):
    # to reconstruct the toeplitz take the last column (-1 last element)
    # and the last row in reverse
    return np.concatenate([Ta[:-1, -1], Ta[-1, :][::-1]])

def reshape_toeplitz(Ta, L):
    a = reconstruct_toeplitz(Ta)
    return make_toepliz_as_in_mulan(a, L)


def build_frobenius_weights(A):
    N, L = A.shape
    D = N + L - 1

    # matrix of weights for the weighted Frobenius norm
    r = np.arange(1, L+1)[::-1]
    c = np.concatenate([np.arange(1,L+1), L*np.ones(N-L)])[::-1]
    W = sp.linalg.toeplitz(c, r)
    return W

def enforce_toeplitz(A):
    N, P = A.shape
    z = np.zeros(N + P - 1, dtype=np.complex64)
    for i in range(z.shape[0]):
        z[i] = np.mean(np.diag(A, P - i - 1))

    return make_toepliz_as_in_mulan(z, P)

def peakdetect(y_axis, x_axis= None, lookahead = 300, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html
    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively
    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function
    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []  # Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)

    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                       y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]

    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]
