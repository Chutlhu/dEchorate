import numpy as np
import scipy.optimize
import functools


def plane_from_points(points):
    D, N = points.shape
    assert D == 3
    assert N > 3

    def plane(x, y, params):
        a = params[0]
        b = params[1]
        c = params[2]
        z = a*x + b*y + c
        return z

    def error(params, points):
        result = 0
        for n in range(N):
            [x, y, z] = points[:, n]
            plane_z = plane(x, y, params)
            diff = abs(plane_z - z)
            result += diff**2
        return result


    fun = functools.partial(error, points=points)
    params0 = [0, 0, 0]
    res = scipy.optimize.minimize(fun, params0)

    a = res.x[0]
    b = res.x[1]
    c = res.x[2]

    return np.array([a, b, c])

def mesh_from_plane(point, normal):

    def cross(a, b):
        return [a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0]]

    a, b, c = normal

    normal = np.array(cross([1, 0, a], [0, 1, b]))
    d = -point.dot(normal)
    xx, yy = np.meshgrid([-5, 5], [-5, 5])
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    return xx, yy, z
