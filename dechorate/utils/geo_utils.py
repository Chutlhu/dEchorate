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


def dist_point_plane(point, plane_point, plane_normal):
    x, y, z = plane_point
    a, b, c = plane_normal
    d = - (a*x + b*y + c*z)
    q, r, s = point
    dist = np.abs(a*q + b*r + c*s + d) / np.sqrt(a**2 + b**2 + c**2)
    return dist


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


def square_within_plane(center, normal, size=(5,5)):
    a, b = size[0]/2, size[1]/2
    B = np.array([[-a, -b, 0],
                  [ a, -b, 0],
                  [ a,  b, 0],
                  [-a,  b, 0]]).T # 3xN convention
    assert B.shape == (3, 4)
    # find the rotation matrix that bring [0,0,1] to the input normal
    a = np.array([0, 0, 1])
    b = normal

    R = rotation_matrix(a, b)
    # apply rotation
    B = R @ B
    assert np.allclose(np.mean(B, -1).sum(), 0)
    # translate
    B = B + center[:, None]
    return B


def rotation_matrix(a, b):
    # https://math.stackexchange.com/questions/180418

    assert a.shape == b.shape
    assert len(a) == len(b) == 3

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    c = np.dot(a, b)

    if np.allclose(c, 1) or np.allclose(c, -1):
        R = np.eye(3)

    else:
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        assert np.allclose(1 - c**2,  s**2)

        sk_v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        skw_v2 = np.dot(sk_v, sk_v)
        R = np.eye(3) + sk_v + skw_v2 / (1 - c)

    return R

if __name__ == "__main__":
    a = np.array([0, 1, 0])
    n = np.array([0, 1, 0])
    R = square_within_plane(a, n, size=(5, 5))
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig)
    verts = [list(zip(R[0, :], R[1, :], R[2, :]))]
    ax.add_collection3d(Poly3DCollection(verts))
    plt.show()
    pass

