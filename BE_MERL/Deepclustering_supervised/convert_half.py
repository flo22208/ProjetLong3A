import torch
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch

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




def plot_vectors_3d(wi, wo, h, theta_i, phi_i, theta_o, phi_o, theta_h, phi_h, theta_d, phi_d, iteration_info):
    """
    Affiche les vecteurs wi, wo et h en 3D
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Origine
    origin = [0, 0, 0]
    
    # Convertir les tensors en numpy
    wi_np = wi.cpu().numpy().flatten()
    wo_np = wo.cpu().numpy().flatten()
    h_np = h.cpu().numpy().flatten()
    
    # Tracer les vecteurs
    ax.quiver(origin[0], origin[1], origin[2], wi_np[0], wi_np[1], wi_np[2], 
              color='blue', arrow_length_ratio=0.15, linewidth=2.5, label='wi (incident)')
    ax.quiver(origin[0], origin[1], origin[2], wo_np[0], wo_np[1], wo_np[2], 
              color='red', arrow_length_ratio=0.15, linewidth=2.5, label='wo (sortant)')
    ax.quiver(origin[0], origin[1], origin[2], h_np[0], h_np[1], h_np[2], 
              color='green', arrow_length_ratio=0.15, linewidth=2.5, label='h (half)')
    
    # Tracer la normale (axe Z)
    ax.quiver(origin[0], origin[1], origin[2], 0, 0, 1, 
              color='black', arrow_length_ratio=0.15, linewidth=1.5, linestyle='--', label='normale')
    
    # Configuration des axes
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.2])
    
    # Titre avec informations
    title = f"{iteration_info}\n"
    title += f"Angles naturels: θi={theta_i:.3f}, φi={phi_i:.3f}, θo={theta_o:.3f}, φo={phi_o:.3f}\n"
    title += f"Rusinkiewicz: θh={theta_h:.3f}, φh={phi_h:.3f}, θd={theta_d:.3f}, φd={phi_d:.3f}"
    ax.set_title(title, fontsize=9)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test():
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
    ntheta_i = 30
    ntheta_o = 30
    nphi_rel = 60
    theta_i_grid = torch.linspace(0, torch.pi/2, ntheta_i, dtype=torch.float32)
    theta_o_grid = torch.linspace(0, torch.pi/2, ntheta_o, dtype=torch.float32)
    phi_rel_grid = torch.linspace(torch.pi,torch.pi, nphi_rel, dtype=torch.float32)

    ## test de phi_h = 0
    ereur = 0.0
    ereur_phi_h = 0.0
    eps = 0.1
    dist_angulaire = 0.0
    dist_angulaire_phi_h = 0.0
    nombre_tests = ntheta_i * ntheta_o * nphi_rel

    rosin_angle = []
    
    # Paramètre pour contrôler la fréquence d'affichage 3D
    # Afficher seulement toutes les N itérations pour éviter trop de graphiques
    display_every = 15050  # Afficher environ 6 exemples sur les 54000 itérations
    iteration_count = 0

    
    for i in range(ntheta_i):
        print(f"Test {i+1}/{ntheta_i} : θi = {theta_i_grid[i].item():.3f} rad")
        for j in range(ntheta_o):
            for k in range(nphi_rel):
                theta_i = theta_i_grid[i]
                theta_o = theta_o_grid[j]
                phi_rel = phi_rel_grid[k]
                phi_i = phi_rel/2
                phi_o = -phi_rel/2
                
                # Calculer les vecteurs wi, wo et h
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
                
                h = wi + wo
                h = _normalize(h)
                
                # Conversion vers Rusinkiewicz (half-diff)
                theta_h, phi_h, theta_d, phi_d = natural_to_rusinkiewicz(theta_i, phi_i, theta_o, phi_o)

                
                
                # Afficher la visualisation 3D tous les N itérations
                if iteration_count % display_every == 0:
                    iteration_info = f"Itération {iteration_count}/{nombre_tests} (i={i}, j={j}, k={k})"
                    plot_vectors_3d(wi, wo, h, 
                                   theta_i.item(), phi_i.item(), theta_o.item(), phi_o.item(),
                                   theta_h.item(), phi_h.item(), theta_d.item(), phi_d.item(),
                                   iteration_info)
                
                iteration_count += 1
                
                if phi_h > eps:
                    ereur_phi_h += 1
            
                    dist_angulaire_phi_h += torch.abs(phi_h).item() 

                rosin_angle.append([theta_h.item(), phi_h.item(), theta_d.item(), phi_d.item()])
                # Conversion inverse vers naturel
                theta_i_rec, phi_i_rec, theta_o_rec, phi_o_rec = rusinkiewicz_to_natural(theta_h, phi_h, theta_d, phi_d)
                
                # Vérification de la précision de la conversion
                if (torch.abs(theta_i - theta_i_rec) > eps or torch.abs(phi_i - phi_i_rec) > eps or
                    torch.abs(theta_o - theta_o_rec) > eps or torch.abs(phi_o - phi_o_rec) > eps):

                    dist_angulaire += torch.sqrt((theta_i - theta_i_rec)**2 + (phi_i - phi_i_rec)**2 + (theta_o - theta_o_rec)**2 + (phi_o - phi_o_rec)**2).item()
                    ereur += 1
                
                
            

    print(f"Nombre de tests : {nombre_tests}, Erreurs : {ereur}")
    print("taux d'erreur : {:.2f}%".format(100 * ereur / (nombre_tests)))
    print("taux d'erreur φh : {:.2f}%".format(100 * ereur_phi_h / (nombre_tests)))
    print("Distance angulaire moyenne : {:.4f}".format(dist_angulaire / (ereur if ereur > 0 else 1)))
    print("Distance angulaire moyenne φh : {:.4f}".format(dist_angulaire_phi_h / (nombre_tests)))


    print("\n\nDistribution des angles Rusinkiewicz (θh, φh) :")
    rosin_angle = np.array(rosin_angle)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1)
    plt.hist(rosin_angle[:, 0], bins=20, color='blue', alpha=0.7)
    plt.title("Distribution de θh")
    plt.xlabel("θh (rad)")
    plt.ylabel("Fréquence")
    plt.subplot(1, 4, 2)
    plt.hist(rosin_angle[:, 1], bins=20, color='orange', alpha=0.7)
    plt.title("Distribution de φh")
    plt.xlabel("φh (rad)")
    plt.ylabel("Fréquence")
    plt.subplot(1, 4, 3)
    plt.hist(rosin_angle[:, 2], bins=20, color='green', alpha=0.7)
    plt.title("Distribution de θd")
    plt.xlabel("θd (rad)")
    plt.ylabel("Fréquence")
    plt.subplot(1, 4, 4)
    plt.hist(rosin_angle[:, 3], bins=20, color='red', alpha=0.7)
    plt.title("Distribution de φd")
    plt.xlabel("φd (rad)")
    plt.ylabel("Fréquence")
   
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()