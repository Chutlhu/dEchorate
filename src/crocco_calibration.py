import numpy as np

def crocco_calibration(D, ):

# function[R, S, D] = unfold_crocco(sqT, c)
# %
# % [R, S, D] = unfold_crocco(sqT, c)
# %
# % Solves the multidimensional unfolding(MDU) problem using the method in
# % Marco Crocco, Alessio Del Bue, and Vittorio Murino: A Bilinear Approach to
# % the Position Self-Calibration of Multiple Sensors
# %
# %
# % INPUT:  T ... M by K matrix, where M is the number of microphones, and K
# %               the number of sources(acoustic events)
# T(i, j) is the
# %               propagation time between the i-th microphone and the j-th
# %               source
# %         c ... speed of sound
# %
# % OUTPUT: R ... (dim by m) Estimated microphone locations
# %         S ... (dim by k) Estimated source locations
# %         D ... ((m+k) by(m+k)) Resulting EDM
# %
# %
# % Author: Ivan Dokmanic, 2014

# [M, K] = size(sqT)

# T = (sqT * c). ^ 2
# % convert to squared "distances"
# T   = bsxfun(@minus, T, T(: , 1))
# % (*)
# T   = bsxfun(@minus, T, T(1, : ))
# T = T(2: end, 2: end)

# D = (sqT * c). ^ 2

# [U, Sigma, V] = svd(T)

# Sigma = Sigma(1: 3, 1: 3)
# U     = U(:, 1: 3)
# V     = V(:, 1: 3)

# % Assume we know the distance between the first sensor and the first
# % microphone. This is realistic.
# a1 = sqT(1, 1) * c

# opt = optimset('MaxFunEvals', 1e8, 'MaxIter', 1e6)

# MAX_ITER = 0
# [Cbest, costbest] = fminsearch(@(C) costC2(C, U, Sigma, V, D, a1), randn(3), opt)
# for i = 1:
#     MAX_ITER
#     i
#     [C, costval] = fminsearch(@(C) costC2(C, U, Sigma, V, D, a1), randn(3), opt)
#     if costval < costbest
#     costbest = costval
#     Cbest = C
#     end
# end
# C = Cbest

# tilde_R = (U*C)'
# tilde_S = -1/2 * C\(Sigma*V')

# R = [[0 0 0]' tilde_R]

# % This doesn't work for some reason(S)!!!
# tilde_S(1, :) = tilde_S(1, : ) + a1
# S = [[a1 0 0]' tilde_S]

# D = edm([R S], [R S])


# function C = costC2(C, U, Sigma, V, D, a1)

# X_tilde = (U*C)'
# Y_tilde = -1/2*inv(C)*Sigma*V'

# X = [[0 0 0]' X_tilde]

# Y = [[0 0 0]' Y_tilde]
# Y(1, :) = Y(1, : ) + a1

# C = norm(edm(X, Y) - D, 'fro') ^ 2
