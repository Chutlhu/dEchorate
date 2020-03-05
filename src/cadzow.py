import numpy as np

from src.utils.dsp_utils import make_toepliz_as_in_mulan, reshape_toeplitz, enforce_toeplitz, build_frobenius_weights

def cadzow_denoise(A, n_spikes, thr_Cadzow=2e-5):
    '''
    Cadzow denoising method
    from Condat implementation
    '''
    N, P = A.shape
    K = n_spikes
    # run Cadzow denoising
    for _ in range(100):
        # low-rank projection
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        A = np.dot(u[:, :K] * s[:K], vh[:K, :])
        print(s[:K], s[K])

        # enforce Toeplitz structure
        A = enforce_toeplitz(A)

        if s[K] < thr_Cadzow:
            break

    A = reshape_toeplitz(A, K+1)
    assert A.shape[1] == K+1
    return A


def condat_denoise(A, n_spikes, thr_Cadzow=2e-5):
    '''
    Method from Condat
    the matrices have size D-L-1 x L. K <= L <= M required.
    '''
    N, L = A.shape # matrix have size D-L+1 x L
    D = N + L - 1
    K = n_spikes

    # parameters
    niter = 20  # number of iterations.
    μ = 0.1      # parameter. Must be in ]0,2[
    γ = 0.51*μ   # parameter. Must be in ]μ/2,1[

    # initialization of the weighted matrix, w
    W = build_frobenius_weights(A)

    Tnoisy = A.copy()
    Tdensd = A.copy() # the noisy matrix is the initialization
    Tauxil = A.copy() # auxtiliary matrix

    for _ in range(niter):
        U, s, Vh = np.linalg.svd(
            Tauxil + γ*(Tdensd-Tauxil) + μ*(Tnoisy-Tdensd)/W,
            full_matrices=False)
        # SVD truncation -> Tdenoised has rank K
        Tdensd = np.dot(U[:, :K] * s[:K], Vh[:K, :])
        print(s[:K], s[K])
        Tauxil = Tauxil-Tdensd+enforce_toeplitz(2*Tdensd-Tauxil)

    # at this point, Tdensd has rank K but is not exactly Toeplitz
    Tdensd = enforce_toeplitz(Tdensd)
    # we reshape the Toeplitz matrix Tdensd into a Toeplitz matrix with K+1 columns
    Tdensd = reshape_toeplitz(Tdensd, K+1)
    assert Tdensd.shape[1] == K+1
    return Tdensd
