import numpy as _np

def project_point_on_line(point, line_point_and_vector):
    r0 = _np.array(point)
    r1 = _np.array(line_point_and_vector[0])
    s1 = _np.array(line_point_and_vector[1])
    
    return r1 - ((r1 - r0).dot(s1.T))/s1.dot(s1.T) * s1


def get_axis_rotation_matrix(vector, angle):
    if _np.linalg.norm(vector) != 1:
        vector = vector / _np.linalg.norm(vector)
    
    cf = _np.cos(angle)
    sf = _np.sin(angle)
    vf = 1 - cf
    x, y, z = vector
    
    return _np.array([[x*x*vf + cf, x*y*vf - z*sf, x*z*vf + y*sf],
                        [x*y*vf + z*sf, y*y*vf + cf, y*z*vf - x*sf],
                        [x*z*vf - y*sf, y*z*vf + x*sf, z*z*vf + cf]])


def get_plane_by_3_points(p0, p1, p2, verbose=False):
    cols12 = _np.vstack((p1 - p0, p2 - p0)).T
    if verbose:
        print('cols12:', cols12)
        print(cols12[[1,2]])
        print(cols12[[0,2]])
        print(cols12[[0,1]])
        
    A = _np.linalg.det(cols12[[1,2]])
    B = _np.linalg.det(cols12[[0,2]])
    C = _np.linalg.det(cols12[[0,1]])
    
    koeffs = _np.array([A,-B,C])
    koeffs = _np.append(koeffs, -koeffs.dot(p0))
    
    return koeffs


def get_plane_by_vector_and_point(vec, p):
    plane = _np.array(vec)
    plane = _np.append(plane, -vec.dot(p))
    return plane


def project_point_on_plane(p, plane):
    a,b,c = plane[:3]
    x0, y0, z0 = p
    A = _np.array([[b,-a,0],
                   [c,0,-a],
                   [a,b,c]])
    b = _np.array([b*x0 - a*y0,
                   c*x0 - a*z0,
                   -plane[-1]])
    return _np.linalg.solve(A, b)

