import torch
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



EPS = 1e-12


# ============================================================
# UTILITAIRES (fidèles ALTA)
# ============================================================

def _normalize(v):
    n = torch.norm(v, dim=0, keepdim=True)
    fallback = torch.tensor([[0.0], [0.0], [1.0]], dtype=v.dtype, device=v.device)
    return torch.where(n > EPS, v / (n + EPS), fallback)


def _rotate_z(v, phi):
    """
    v : shape (3, N)
    phi : shape (N,)
    """
    c = torch.cos(phi)
    s = torch.sin(phi)

    x =  c * v[0] + s * v[1]
    y = -s * v[0] + c * v[1]
    z =  v[2]

    return torch.vstack([x, y, z])


def _rotate_y(v, theta):
    """
    v : shape (3, N)
    theta : shape (N,)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    x =  c * v[0] - s * v[2]
    y =  v[1]
    z =  s * v[0] + c * v[2]

    return torch.vstack([x, y, z])


# ============================================================
# NATURAL → RUSINKIEWICZ (ALTA exact)
# ============================================================

def natural_to_rusinkiewicz(theta_i, phi_i, theta_o, phi_o, anisotropic=True):
    """
    Reproduction fidèle de ALTA from_cartesian() pour Rusinkiewicz.
    """

    # --- wi, wo en cartésien ---
    wi = torch.vstack([
        torch.sin(theta_i) * torch.cos(phi_i),
        torch.sin(theta_i) * torch.sin(phi_i),
        torch.cos(theta_i)
    ])

    wo = torch.vstack([
        torch.sin(theta_o) * torch.cos(phi_o),
        torch.sin(theta_o) * torch.sin(phi_o),
        torch.cos(theta_o)
    ])

    # --- Half vector ---
    h = wi + wo
    h = _normalize(h)

    # θh
    theta_h = torch.acos(torch.clamp(h[2], -1.0, 1.0))

    # φh
    phi_h = torch.atan2(h[1], h[0])

    # --- Rotation ALTA ---
    # diff = Ry(-θh) · Rz(-φh) · wi
    diff = _rotate_z(wi, -phi_h)
    diff = _rotate_y(diff, -theta_h)

    theta_d = torch.acos(torch.clamp(diff[2], -1.0, 1.0))
    phi_d   = torch.atan2(diff[1], diff[0])

    if anisotropic:
        return theta_h, phi_h, theta_d, phi_d
    else:
        return theta_h, theta_d, phi_d


# ============================================================
# RUSINKIEWICZ → NATURAL (ALTA exact)
# ============================================================

def rusinkiewicz_to_natural(theta_h, phi_h, theta_d, phi_d):
    """
    Reproduction fidèle de ALTA to_cartesian().
    """

    # --- Half vector global ---
    h = torch.vstack([
        torch.sin(theta_h) * torch.cos(phi_h),
        torch.sin(theta_h) * torch.sin(phi_h),
        torch.cos(theta_h)
    ])

    # --- Diff vector dans repère local ---
    diff_local = torch.vstack([
        torch.sin(theta_d) * torch.cos(phi_d),
        torch.sin(theta_d) * torch.sin(phi_d),
        torch.cos(theta_d)
    ])

    # Inverse rotation : Rz(φh) · Ry(θh)
    tmp = _rotate_y(diff_local, theta_h)
    wi  = _rotate_z(tmp, phi_h)

    wi = _normalize(wi)

    # --- Reflection ALTA ---
    dot = torch.sum(wi * h, dim=0)
    wo = 2.0 * dot * h - wi
    wo = _normalize(wo)

    # --- Angles naturels ---
    theta_i = torch.acos(torch.clamp(wi[2], -1.0, 1.0))
    phi_i   = torch.atan2(wi[1], wi[0])

    theta_o = torch.acos(torch.clamp(wo[2], -1.0, 1.0))
    phi_o   = torch.atan2(wo[1], wo[0])

    return theta_i, phi_i, theta_o, phi_o

def matrice_sens_direct():
    """
    Test de conversion entre coordonnées naturelles et Rusinkiewicz (half-diff)
    pour la création de la matrice de conversion BRDF isotropique
    """
    import numpy as np
    
    print("=" * 80)
    print("TEST DE CONVERSION : NATUREL ↔ RUSINKIEWICZ (HALF-DIFF)")
    print("=" * 80)
    print("\nCe test simule la conversion utilisée pour la matrice BRDF isotropique\n")
    
    # Générer 5 configurations aléatoires d'angles naturels
    ntheta_i = 15
    ntheta_o = 15
    nphi_rel = 30
    theta_i_grid = torch.linspace(0, torch.pi/2, ntheta_i, dtype=torch.float32)
    theta_o_grid = torch.linspace(0, torch.pi/2, ntheta_o, dtype=torch.float32)
    phi_rel_grid = torch.linspace(0, torch.pi, nphi_rel, dtype=torch.float32)

    mat_res = torch.zeros((ntheta_i, ntheta_o, nphi_rel, 3), dtype=torch.int32)

    ntheta_h = 90
    ntheta_d = 90
    nphi_d = 180
    theta_h_grid = torch.linspace(0, torch.pi/2, ntheta_h, dtype=torch.float32)
    theta_d_grid = torch.linspace(0, torch.pi, ntheta_d, dtype=torch.float32)
    phi_d_grid = torch.linspace(0, torch.pi, nphi_d, dtype=torch.float32) 

    for i in range(ntheta_i):
        print(f"Test {i+1}/{ntheta_i} : θi = {theta_i_grid[i].item():.3f} rad")
        for j in range(ntheta_o):
            for k in range(nphi_rel):
                theta_i = theta_i_grid[i]
                theta_o = theta_o_grid[j]
                phi_rel = phi_rel_grid[k]
                phi_i = phi_rel/2
                phi_o = -phi_rel/2
                
                # Conversion vers Rusinkiewicz (half-diff)
                theta_h, theta_d, phi_d, phi_h = natural_to_rusinkiewicz(theta_i, phi_i, theta_o, phi_o)

                # conversion en indices
                theta_h_idx = int(torch.clamp((theta_h / (torch.pi/2)) * (ntheta_h - 1), 0, ntheta_h - 1).item())
                theta_d_idx = int(torch.clamp((theta_d / (torch.pi)) * (ntheta_d - 1), 0, ntheta_d - 1).item())
                phi_d_idx = int(torch.clamp((phi_d / torch.pi) * (nphi_d - 1), 0, nphi_d - 1).item())
                phi_h_idx = 0
                
                # Stocker la valeur dans la matrice de résultats
                mat_res[i, j, k] = torch.tensor([theta_h_idx, theta_d_idx, phi_d_idx])

    print("nombre d'éléments uniques dans la matrice de résultats : ", torch.unique(mat_res[:, :, :, 0]).shape[0], " / ", ntheta_i * ntheta_o * nphi_rel)

def matrice_sens_indirect():
    """
    Test de conversion entre coordonnées naturelles et Rusinkiewicz (half-diff)
    pour la création de la matrice de conversion BRDF isotropique
    """
    import numpy as np
    
    print("=" * 80)
    print("TEST DE CONVERSION : NATUREL ↔ RUSINKIEWICZ (HALF-DIFF)")
    print("=" * 80)
    print("\nCe test simule la conversion utilisée pour la matrice BRDF isotropique\n")
    
    # Générer 5 configurations aléatoires d'angles naturels
    ntheta_i = 10
    ntheta_o = 10
    nphi_rel = 20
    theta_i_grid = torch.linspace(0, torch.pi/2, ntheta_i, dtype=torch.float32)
    theta_o_grid = torch.linspace(0, torch.pi/2, ntheta_o, dtype=torch.float32)
    phi_rel_grid = torch.linspace(0, torch.pi, nphi_rel, dtype=torch.float32)

    ntheta_h = 10
    ntheta_d = 10
    nphi_d = 20
    theta_h_grid = torch.linspace(0, torch.pi/2, ntheta_h, dtype=torch.float32)
    theta_d_grid = torch.linspace(0, torch.pi/2, ntheta_d, dtype=torch.float32)
    phi_d_grid = torch.linspace(-torch.pi/2, torch.pi/2, nphi_d, dtype=torch.float32)   

    mat_res = torch.zeros((ntheta_i, ntheta_o, nphi_rel, 4))

    non_istrop_count = 0
    for i in range(ntheta_h):
        print(f"Test {i+1}/{ntheta_h} : θh = {theta_h_grid[i].item():.3f} rad")
        for j in range(ntheta_d):
            for k in range(nphi_d):
                theta_h = theta_h_grid[i]
                theta_d = theta_d_grid[j]
                phi_d = phi_d_grid[k]
                phi_h = torch.tensor(0.0, dtype=torch.float32)
                
                # Conversion vers Rusinkiewicz (half-diff)
                theta_i, phi_i, theta_o, phi_o = rusinkiewicz_to_natural(theta_h, theta_d, phi_d, phi_h)
                
                # Calculer les indices avec clamping pour éviter les dépassements
                theta_i_idx = int(torch.clamp((theta_i / (torch.pi/2)) * (ntheta_i - 1), 0, ntheta_i - 1).item())
                theta_o_idx = int(torch.clamp((theta_o / (torch.pi/2)) * (ntheta_o - 1), 0, ntheta_o - 1).item())
                phi_rel = torch.abs(phi_i)
                phi_rel_idx = int(torch.clamp((phi_rel / torch.pi) * (nphi_rel - 1), 0, nphi_rel - 1).item())

                mat_res[theta_i_idx, theta_o_idx, phi_rel_idx] = torch.stack([theta_h, theta_d, phi_d, phi_h])
    

    print("nombre d'éléments remplis dans la matrice de résultats : ", torch.count_nonzero(mat_res[:, :, :, 0])," / ", ntheta_i * ntheta_o * nphi_rel)
if __name__ == "__main__":
    matrice_sens_direct()