import numpy as np
from utilsBRDFDisney import rusinkiewicz_to_LV
from convert_half_benj import get_angles

def angle_diff(a, b):
    d = a - b
    d = (d + np.pi) % (2*np.pi) - np.pi
    return abs(d)

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
        phi_d   = np.random.uniform(-np.pi, np.pi)

        L, V, n, t, _ = rusinkiewicz_to_LV(theta_h, theta_d, phi_d)

        _,_,_,_,theta_h2, phi_h2, theta_d2, phi_d2 = get_angles(L, V, n, t)

        max_err_theta_h = max(max_err_theta_h, abs(theta_h - theta_h2))
        max_err_phi_h   = max(max_err_phi_h,   angle_diff(phi_h,  phi_h2))
        max_err_theta_d = max(max_err_theta_d, abs(theta_d - theta_d2))
        max_err_phi_d   = max(max_err_phi_d,   angle_diff(phi_d,  phi_d2))

    print("=== Résultats du test de cohérence ===")
    print("Erreur max theta_h :", max_err_theta_h)
    print("Erreur max phi_h   :", max_err_phi_h)
    print("Erreur max theta_d :", max_err_theta_d)
    print("Erreur max phi_d   :", max_err_phi_d)

if __name__ == "__main__":
    test_rusinkiewicz_function()