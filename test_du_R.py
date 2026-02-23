import numpy as np
from utilsBRDFDisney import rusinkiewicz_to_LV
from convert_half_benj import get_angles

def angle_diff_phi(a, b):
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return abs(d)

def angle_diff_theta(a, b):
    if (a < 0 and b > 0) or (a > 0 and b < 0):
        print("a:", a, "b:", b)
        return abs(a + b)
    else:
        return abs(a - b)

def test_rusinkiewicz_function():
    Ntests = 1000

    max_err_theta_h = 0
    max_err_phi_h   = 0
    max_err_theta_d = 0
    max_err_phi_d   = 0

    for _ in range(Ntests):
        theta_h = np.random.uniform(0, np.pi/2)
        phi_h   = 0.0
        theta_d = np.random.uniform(0, np.pi/2)
        phi_d   = np.random.uniform(0, np.pi)

        L, V, n, _, t = rusinkiewicz_to_LV(theta_h, phi_h, theta_d, phi_d)

        _,_,_,_,theta_h2, phi_h2, theta_d2, phi_d2 = get_angles(L, V, n, t)

        max_err_theta_h = max(max_err_theta_h, angle_diff_theta(theta_h, theta_h2))
        max_err_phi_h   = max(max_err_phi_h,   angle_diff_phi(phi_h,  phi_h2))
        max_err_theta_d = max(max_err_theta_d, angle_diff_theta(theta_d, theta_d2))
        max_err_phi_d   = max(max_err_phi_d,   angle_diff_phi(phi_d,  phi_d2))

    print("=== Résultats du test de cohérence ===")
    print("Erreur max theta_h :", max_err_theta_h)
    print("Erreur max phi_h   :", max_err_phi_h)
    print("Erreur max theta_d :", max_err_theta_d)
    print("Erreur max phi_d   :", max_err_phi_d)

if __name__ == "__main__":
    test_rusinkiewicz_function()