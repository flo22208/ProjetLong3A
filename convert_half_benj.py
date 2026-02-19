import numpy

def complete_basis(a, b):
    u1 = a / numpy.linalg.norm(a)
    proj_b_on_u1 = numpy.dot(b, u1) * u1
    b_ortho = b - proj_b_on_u1
    u2 = b_ortho / numpy.linalg.norm(b_ortho)
    u3 = numpy.cross(u1, u2)
    M = numpy.array([u1, u2, u3])
    return M

def spherical_to_cartesian(theta, phi):
    x = numpy.sin(theta) * numpy.cos(phi)
    y = numpy.sin(theta) * numpy.sin(phi)
    z = numpy.cos(theta)
    return numpy.array([x, y, z])

def cartesian_to_spherical(vec):
    x, y, z = vec
    r = numpy.linalg.norm(vec)
    theta = numpy.arccos(z / r)
    phi = numpy.arctan2(y, x)
    return theta, phi

def spherical_tangent_base(theta, phi):
    M = numpy.array([
        [numpy.cos(theta) * numpy.cos(phi), numpy.cos(theta) * numpy.sin(phi), -numpy.sin(theta)],
        [-numpy.sin(phi), numpy.cos(phi),0],
        [numpy.sin(theta) * numpy.cos(phi), numpy.sin(theta) * numpy.sin(phi), numpy.cos(theta)],
        ])
    return M

def natural_to_halfangle(theta_i, phi_i, theta_o, phi_o):
    wi_tangent = spherical_to_cartesian(theta_i, phi_i)
    wo_tangent = spherical_to_cartesian(theta_o, phi_o)
    unnormed_half = 0.5 * (wi_tangent + wo_tangent)
    half = unnormed_half / numpy.linalg.norm(unnormed_half)
    theta_h, phi_h = cartesian_to_spherical(half)
    M_half = spherical_tangent_base(theta_h, phi_h)
    wi_local = M_half @ wi_tangent
    theta_d, phi_d = cartesian_to_spherical(wi_local)
    return theta_h, phi_h, theta_d, phi_d

def halfangle_to_natural(theta_h, phi_h, theta_d, phi_d):
    wi_local = spherical_to_cartesian(theta_d, phi_d)
    wo_local = spherical_to_cartesian(theta_d, phi_d + numpy.pi)
    M_half = spherical_tangent_base(theta_h, phi_h)
    wi_tangent, wo_tangent = M_half.T @ wi_local, M_half.T @ wo_local
    theta_i, phi_i = cartesian_to_spherical(wi_tangent)
    theta_o, phi_o = cartesian_to_spherical(wo_tangent)
    return theta_i, phi_i, theta_o, phi_o

def get_angles(wi, wo, n, t):
    M_tangent = complete_basis(n, t)
    wi_tangent, wo_tangent = M_tangent @ wi, M_tangent @ wo
    theta_i, phi_i = cartesian_to_spherical(wi_tangent)
    theta_o, phi_o = cartesian_to_spherical(wo_tangent)
    theta_h, phi_h, theta_d, phi_d = natural_to_halfangle(theta_i, phi_i, theta_o, phi_o)
    return theta_i, phi_i, theta_o, phi_o, theta_h, phi_h, theta_d, phi_d
