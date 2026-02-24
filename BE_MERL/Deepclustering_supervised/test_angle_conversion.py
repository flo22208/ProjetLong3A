"""
Script de test pour vérifier les conversions d'angles halfangle_to_natural
et les calculs d'indices résultants
"""

import numpy as np
from convert_half import halfangle_to_natural
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from convert_half import halfangle_to_natural
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(theta, phi):
    """Convertit des coordonnées sphériques en cartésiennes
    theta: angle polaire (0 à π/2 pour la demi-sphère supérieure)
    phi: angle azimutal (0 à 2π)
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])

def visualize_angle_conversion_3d(test_cases_indices):
    """Visualise en 3D les conversions d'angles pour quelques cas de test
    
    test_cases_indices: liste de tuples (theta_h_idx, theta_d_idx, phi_d_idx)
    """
    ntheta_h = 90
    ntheta_d = 90
    nphi_d = 180
    
    theta_i_data = np.linspace(0, np.pi/2, ntheta_d, dtype=np.float32)
    phi_d_data = np.linspace(0, np.pi, nphi_d, dtype=np.float32)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Créer une demi-sphère pour référence
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi / 2, 15)
    x_sphere = np.outer(np.sin(v), np.cos(u))
    y_sphere = np.outer(np.sin(v), np.sin(u))
    z_sphere = np.outer(np.cos(v), np.ones(np.size(u)))
    
    for idx, (theta_h_idx, theta_d_idx, phi_d_idx) in enumerate(test_cases_indices):
        # Subplot pour la paramétrisation Rusinkiewicz (halfangle)
        ax1 = fig.add_subplot(2, len(test_cases_indices), idx + 1, projection='3d')
        
        # Calculer theta_h à partir de l'indice
        th = np.arccos(np.sqrt(theta_h_idx / (ntheta_h - 1)))
        phi_h = 0  # toujours 0 dans le code
        theta_d = theta_i_data[theta_d_idx]
        phi_d = phi_d_data[phi_d_idx]
        
        # Conversion halfangle -> natural
        theta_i, phi_i, theta_o, phi_o = halfangle_to_natural(th, phi_h, theta_d, phi_d)
        
        # Normalisation
        phi_i_norm = phi_i % (2 * np.pi)
        phi_o_norm = phi_o % (2 * np.pi)
        theta_i_norm = np.clip(theta_i, 0, np.pi/2)
        theta_o_norm = np.clip(theta_o, 0, np.pi/2)
        
        # === VISUALISATION RUSINKIEWICZ (halfangle) ===
        # Afficher la demi-sphère de référence
        ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        # Calculer le vecteur half (mi-vecteur)
        h = spherical_to_cartesian(th, phi_h)
        
        # Afficher le vecteur half
        ax1.quiver(0, 0, 0, h[0], h[1], h[2], 
                   color='red', arrow_length_ratio=0.2, linewidth=2.5, label='half-vector (h)')
        
        # Afficher la normale
        ax1.quiver(0, 0, 0, 0, 0, 1, 
                   color='black', arrow_length_ratio=0.15, linewidth=2, label='normale (n)', alpha=0.5)
        
        # Configuration du subplot
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([0, 1])
        ax1.set_xlabel('X', fontsize=8)
        ax1.set_ylabel('Y', fontsize=8)
        ax1.set_zlabel('Z', fontsize=8)
        ax1.set_title(f'Rusinkiewicz [{theta_h_idx},{theta_d_idx},{phi_d_idx}]\n'
                      f'θh={np.degrees(th):.1f}°, φh={np.degrees(phi_h):.1f}°\n'
                      f'θd={np.degrees(theta_d):.1f}°, φd={np.degrees(phi_d):.1f}°', 
                      fontsize=9)
        ax1.legend(fontsize=7, loc='upper right')
        ax1.view_init(elev=20, azim=45)
        
        # === VISUALISATION NATURAL ===
        ax2 = fig.add_subplot(2, len(test_cases_indices), len(test_cases_indices) + idx + 1, projection='3d')
        
        # Afficher la demi-sphère de référence
        ax2.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        # Vecteurs incident et sortant en coordonnées natural
        wi = spherical_to_cartesian(theta_i_norm, phi_i_norm)
        wo = spherical_to_cartesian(theta_o_norm, phi_o_norm)
        
        # Afficher les vecteurs
        ax2.quiver(0, 0, 0, wi[0], wi[1], wi[2], 
                   color='blue', arrow_length_ratio=0.2, linewidth=2.5, label='incident (ωi)')
        ax2.quiver(0, 0, 0, wo[0], wo[1], wo[2], 
                   color='green', arrow_length_ratio=0.2, linewidth=2.5, label='sortant (ωo)')
        ax2.quiver(0, 0, 0, 0, 0, 1, 
                   color='black', arrow_length_ratio=0.15, linewidth=2, label='normale (n)', alpha=0.5)
        
        # Calculer et afficher le half-vector (moyenne normalisée)
        h_natural = (wi + wo) / 2
        if np.linalg.norm(h_natural) > 1e-6:
            h_natural = h_natural / np.linalg.norm(h_natural)
            ax2.quiver(0, 0, 0, h_natural[0], h_natural[1], h_natural[2], 
                       color='red', arrow_length_ratio=0.2, linewidth=1.5, linestyle='--', 
                       label='half (calculé)', alpha=0.7)
        
        # Configuration du subplot
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([0, 1])
        ax2.set_xlabel('X', fontsize=8)
        ax2.set_ylabel('Y', fontsize=8)
        ax2.set_zlabel('Z', fontsize=8)
        ax2.set_title(f'Natural (converti)\n'
                      f'θi={np.degrees(theta_i_norm):.1f}°, φi={np.degrees(phi_i_norm):.1f}°\n'
                      f'θo={np.degrees(theta_o_norm):.1f}°, φo={np.degrees(phi_o_norm):.1f}°', 
                      fontsize=9)
        ax2.legend(fontsize=7, loc='upper right')
        ax2.view_init(elev=20, azim=45)
    
    plt.suptitle('Comparaison des paramétrisations : Rusinkiewicz (haut) vs Natural (bas)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('angle_conversion_comparison_3d.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualisation 3D sauvegardée dans 'angle_conversion_comparison_3d.png'")
    plt.show()

def test_angle_conversions():
    """Test la conversion halfangle -> natural avec des cas connus"""
    
    print("=" * 80)
    print("TEST DE CONVERSION HALFANGLE -> NATURAL")
    print("=" * 80)
    
    # Paramètres de sampling similaires à ceux du code
    ntheta_h = 90
    ntheta_d = 90
    nphi_d = 180
    ntheta_i = ntheta_d  # 90
    nphi_i = 2
    ntheta_o = 90
    nphi_o = 90
    
    theta_i_data = np.linspace(0, np.pi/2, ntheta_i, dtype=np.float32)
    phi_i_data = np.linspace(0, np.pi, nphi_i, dtype=np.float32)
    phi_d_data = np.linspace(0, np.pi, nphi_d, dtype=np.float32)
    
    print(f"\nParamètres de sampling:")
    print(f"  ntheta_h = {ntheta_h}, ntheta_d = {ntheta_d}, nphi_d = {nphi_d}")
    print(f"  ntheta_i = {ntheta_i}, nphi_i = {nphi_i}")
    print(f"  ntheta_o = {ntheta_o}, nphi_o = {nphi_o}")
    print(f"  theta_i range: [0, {np.pi/2:.4f}]")
    print(f"  phi_i range: [0, {np.pi:.4f}]")
    print(f"  phi_d range: [0, {np.pi:.4f}]")
    
    # Test avec des cas spécifiques
    test_cases = [
        # (theta_h_idx, theta_d_idx, phi_d_idx, description)
        (0, 0, 0, "Début de la grille (0,0,0)"),
        (45, 45, 90, "Milieu de la grille (45,45,90)"),
        (89, 89, 179, "Fin de la grille (89,89,179)"),
        (0, 45, 0, "theta_h=0, theta_d=milieu"),
        (45, 0, 90, "theta_h=milieu, theta_d=0"),
        (10, 20, 30, "Cas arbitraire (10,20,30)"),
    ]
    
    print(f"\n{'='*80}")
    print("TESTS DES CAS SPÉCIFIQUES")
    print(f"{'='*80}\n")
    
    for idx, (theta_h_idx, theta_d_idx, phi_d_idx, description) in enumerate(test_cases, 1):
        print(f"--- Test {idx}: {description} ---")
        
        # Calculer theta_h à partir de l'indice (comme dans le code)
        th = np.arccos(np.sqrt(theta_h_idx / (ntheta_h - 1)))
        phi_h = 0  # toujours 0 dans le code
        
        # Récupérer theta_d et phi_d
        theta_d = theta_i_data[theta_d_idx]
        phi_d = phi_d_data[phi_d_idx]
        
        print(f"  Indices BRDF source: [theta_h={theta_h_idx}, theta_d={theta_d_idx}, phi_d={phi_d_idx}]")
        print(f"  Angles halfangle: theta_h={th:.4f} rad ({np.degrees(th):.2f}°), "
              f"phi_h={phi_h:.4f} rad ({np.degrees(phi_h):.2f}°)")
        print(f"                    theta_d={theta_d:.4f} rad ({np.degrees(theta_d):.2f}°), "
              f"phi_d={phi_d:.4f} rad ({np.degrees(phi_d):.2f}°)")
        
        # Conversion halfangle -> natural
        theta_i, phi_i, theta_o, phi_o = halfangle_to_natural(th, phi_h, theta_d, phi_d)
        
        print(f"  Angles natural (bruts): theta_i={theta_i:.4f} rad ({np.degrees(theta_i):.2f}°), "
              f"phi_i={phi_i:.4f} rad ({np.degrees(phi_i):.2f}°)")
        print(f"                          theta_o={theta_o:.4f} rad ({np.degrees(theta_o):.2f}°), "
              f"phi_o={phi_o:.4f} rad ({np.degrees(phi_o):.2f}°)")
        
        # Normalisation comme dans le code de production
        phi_i_norm = phi_i % (2 * np.pi)
        phi_o_norm = phi_o % (2 * np.pi)
        theta_i_norm = np.clip(theta_i, 0, np.pi/2)
        theta_o_norm = np.clip(theta_o, 0, np.pi/2)
        
        print(f"  Angles normalisés: theta_i={theta_i_norm:.4f} rad ({np.degrees(theta_i_norm):.2f}°), "
              f"phi_i={phi_i_norm:.4f} rad ({np.degrees(phi_i_norm):.2f}°)")
        print(f"                     theta_o={theta_o_norm:.4f} rad ({np.degrees(theta_o_norm):.2f}°), "
              f"phi_o={phi_o_norm:.4f} rad ({np.degrees(phi_o_norm):.2f}°)")
        
        # Conversion angles -> indices (comme dans le code)
        tii = int(np.clip(theta_i_norm / (np.pi/2) * ntheta_i, 0, ntheta_i - 1))
        pii = int(np.clip(phi_i_norm / (2*np.pi) * nphi_i, 0, nphi_i - 1))
        tio = int(np.clip(theta_o_norm / (np.pi/2) * ntheta_o, 0, ntheta_o - 1))
        pio = int(np.clip(phi_o_norm / (2*np.pi) * nphi_o, 0, nphi_o - 1))
        
        print(f"  Indices BSDF destination: tii={tii}, pii={pii}, tio={tio}, pio={pio}")
        
        # Vérifier si les valeurs normalisées sont dans les limites
        valid = True
        if theta_i_norm < 0 or theta_i_norm > np.pi/2:
            print(f"  ⚠️  WARNING: theta_i_norm={theta_i_norm:.4f} hors limites [0, π/2]")
            valid = False
        if phi_i_norm < 0 or phi_i_norm > 2*np.pi:
            print(f"  ⚠️  WARNING: phi_i_norm={phi_i_norm:.4f} hors limites [0, 2π]")
            valid = False
        if theta_o_norm < 0 or theta_o_norm > np.pi/2:
            print(f"  ⚠️  WARNING: theta_o_norm={theta_o_norm:.4f} hors limites [0, π/2]")
            valid = False
        if phi_o_norm < 0 or phi_o_norm > 2*np.pi:
            print(f"  ⚠️  WARNING: phi_o_norm={phi_o_norm:.4f} hors limites [0, 2π]")
            valid = False
            
        # Afficher les warnings pour les angles bruts (pour info)
        if theta_i < 0 or theta_i > np.pi/2:
            print(f"  ℹ️  INFO: theta_i brut={theta_i:.4f} ({np.degrees(theta_i):.2f}°) hors limites [0, π/2] → clippé")
        if phi_i < 0 or phi_i > 2*np.pi:
            print(f"  ℹ️  INFO: phi_i brut={phi_i:.4f} ({np.degrees(phi_i):.2f}°) hors limites [0, 2π] → normalisé")
        if theta_o < 0 or theta_o > np.pi/2:
            print(f"  ℹ️  INFO: theta_o brut={theta_o:.4f} ({np.degrees(theta_o):.2f}°) hors limites [0, π/2] → clippé")
        if phi_o < 0 or phi_o > 2*np.pi:
            print(f"  ℹ️  INFO: phi_o brut={phi_o:.4f} ({np.degrees(phi_o):.2f}°) hors limites [0, 2π] → normalisé")
            
        if valid:
            print(f"  ✓ Tous les angles normalisés sont dans les limites valides")
        
        print()
    
    # Statistiques globales
    print(f"{'='*80}")
    print("STATISTIQUES SUR UN ÉCHANTILLON ALÉATOIRE")
    print(f"{'='*80}\n")
    
    sample_size = 1000
    theta_i_values = []
    phi_i_values = []
    theta_o_values = []
    phi_o_values = []
    theta_i_norm_values = []
    phi_i_norm_values = []
    theta_o_norm_values = []
    phi_o_norm_values = []
    
    np.random.seed(42)
    for _ in range(sample_size):
        theta_h_idx = np.random.randint(0, ntheta_h)
        theta_d_idx = np.random.randint(0, ntheta_d)
        phi_d_idx = np.random.randint(0, nphi_d)
        
        th = np.arccos(np.sqrt(theta_h_idx / (ntheta_h - 1)))
        theta_d = theta_i_data[theta_d_idx]
        phi_d = phi_d_data[phi_d_idx]
        
        theta_i, phi_i, theta_o, phi_o = halfangle_to_natural(th, 0, theta_d, phi_d)
        
        # Normalisation
        phi_i_norm = phi_i % (2 * np.pi)
        phi_o_norm = phi_o % (2 * np.pi)
        theta_i_norm = np.clip(theta_i, 0, np.pi/2)
        theta_o_norm = np.clip(theta_o, 0, np.pi/2)
        
        theta_i_values.append(theta_i)
        phi_i_values.append(phi_i)
        theta_o_values.append(theta_o)
        phi_o_values.append(phi_o)
        theta_i_norm_values.append(theta_i_norm)
        phi_i_norm_values.append(phi_i_norm)
        theta_o_norm_values.append(theta_o_norm)
        phi_o_norm_values.append(phi_o_norm)
    
    theta_i_values = np.array(theta_i_values)
    phi_i_values = np.array(phi_i_values)
    theta_o_values = np.array(theta_o_values)
    phi_o_values = np.array(phi_o_values)
    theta_i_norm_values = np.array(theta_i_norm_values)
    phi_i_norm_values = np.array(phi_i_norm_values)
    theta_o_norm_values = np.array(theta_o_norm_values)
    phi_o_norm_values = np.array(phi_o_norm_values)
    
    print(f"Statistiques sur {sample_size} échantillons aléatoires:")
    print(f"\n  === ANGLES BRUTS (avant normalisation) ===")
    print(f"\n  theta_i:")
    print(f"    min={np.min(theta_i_values):.4f} rad ({np.degrees(np.min(theta_i_values)):.2f}°)")
    print(f"    max={np.max(theta_i_values):.4f} rad ({np.degrees(np.max(theta_i_values)):.2f}°)")
    print(f"    mean={np.mean(theta_i_values):.4f} rad ({np.degrees(np.mean(theta_i_values)):.2f}°)")
    print(f"    valeurs hors limites [0, π/2]: {np.sum((theta_i_values < 0) | (theta_i_values > np.pi/2))}")
    
    print(f"\n  phi_i:")
    print(f"    min={np.min(phi_i_values):.4f} rad ({np.degrees(np.min(phi_i_values)):.2f}°)")
    print(f"    max={np.max(phi_i_values):.4f} rad ({np.degrees(np.max(phi_i_values)):.2f}°)")
    print(f"    mean={np.mean(phi_i_values):.4f} rad ({np.degrees(np.mean(phi_i_values)):.2f}°)")
    print(f"    valeurs hors limites [0, 2π]: {np.sum((phi_i_values < 0) | (phi_i_values > 2*np.pi))}")
    
    print(f"\n  theta_o:")
    print(f"    min={np.min(theta_o_values):.4f} rad ({np.degrees(np.min(theta_o_values)):.2f}°)")
    print(f"    max={np.max(theta_o_values):.4f} rad ({np.degrees(np.max(theta_o_values)):.2f}°)")
    print(f"    mean={np.mean(theta_o_values):.4f} rad ({np.degrees(np.mean(theta_o_values)):.2f}°)")
    print(f"    valeurs hors limites [0, π/2]: {np.sum((theta_o_values < 0) | (theta_o_values > np.pi/2))}")
    
    print(f"\n  phi_o:")
    print(f"    min={np.min(phi_o_values):.4f} rad ({np.degrees(np.min(phi_o_values)):.2f}°)")
    print(f"    max={np.max(phi_o_values):.4f} rad ({np.degrees(np.max(phi_o_values)):.2f}°)")
    print(f"    mean={np.mean(phi_o_values):.4f} rad ({np.degrees(np.mean(phi_o_values)):.2f}°)")
    print(f"    valeurs hors limites [0, 2π]: {np.sum((phi_o_values < 0) | (phi_o_values > 2*np.pi))}")
    
    print(f"\n  === ANGLES NORMALISÉS (après normalisation) ===")
    print(f"\n  theta_i_norm:")
    print(f"    min={np.min(theta_i_norm_values):.4f} rad ({np.degrees(np.min(theta_i_norm_values)):.2f}°)")
    print(f"    max={np.max(theta_i_norm_values):.4f} rad ({np.degrees(np.max(theta_i_norm_values)):.2f}°)")
    print(f"    mean={np.mean(theta_i_norm_values):.4f} rad ({np.degrees(np.mean(theta_i_norm_values)):.2f}°)")
    print(f"    valeurs hors limites [0, π/2]: {np.sum((theta_i_norm_values < 0) | (theta_i_norm_values > np.pi/2))}")
    
    print(f"\n  phi_i_norm:")
    print(f"    min={np.min(phi_i_norm_values):.4f} rad ({np.degrees(np.min(phi_i_norm_values)):.2f}°)")
    print(f"    max={np.max(phi_i_norm_values):.4f} rad ({np.degrees(np.max(phi_i_norm_values)):.2f}°)")
    print(f"    mean={np.mean(phi_i_norm_values):.4f} rad ({np.degrees(np.mean(phi_i_norm_values)):.2f}°)")
    print(f"    valeurs hors limites [0, 2π]: {np.sum((phi_i_norm_values < 0) | (phi_i_norm_values > 2*np.pi))}")
    
    print(f"\n  theta_o_norm:")
    print(f"    min={np.min(theta_o_norm_values):.4f} rad ({np.degrees(np.min(theta_o_norm_values)):.2f}°)")
    print(f"    max={np.max(theta_o_norm_values):.4f} rad ({np.degrees(np.max(theta_o_norm_values)):.2f}°)")
    print(f"    mean={np.mean(theta_o_norm_values):.4f} rad ({np.degrees(np.mean(theta_o_norm_values)):.2f}°)")
    print(f"    valeurs hors limites [0, π/2]: {np.sum((theta_o_norm_values < 0) | (theta_o_norm_values > np.pi/2))}")
    
    print(f"\n  phi_o_norm:")
    print(f"    min={np.min(phi_o_norm_values):.4f} rad ({np.degrees(np.min(phi_o_norm_values)):.2f}°)")
    print(f"    max={np.max(phi_o_norm_values):.4f} rad ({np.degrees(np.max(phi_o_norm_values)):.2f}°)")
    print(f"    mean={np.mean(phi_o_norm_values):.4f} rad ({np.degrees(np.mean(phi_o_norm_values)):.2f}°)")
    print(f"    valeurs hors limites [0, 2π]: {np.sum((phi_o_norm_values < 0) | (phi_o_norm_values > 2*np.pi))}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    test_angle_conversions()
    
    # Visualisation 3D pour quelques cas de test
    print("\n" + "="*80)
    print("GÉNÉRATION DE LA VISUALISATION 3D")
    print("="*80)
    
    # Sélectionner quelques cas intéressants à visualiser
    test_cases_for_viz = [
        (0, 0, 0),      # Début de la grille
        (45, 45, 90),   # Milieu de la grille
        (10, 20, 30),   # Cas arbitraire
        (89, 89, 179),  # Fin de la grille
    ]
    
    visualize_angle_conversion_3d(test_cases_for_viz)
