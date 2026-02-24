"""
Script de diagnostic pour analyser les problèmes de conversion d'angles
"""

import numpy as np
from convert_half import natural_to_halfangle
import matplotlib.pyplot as plt

def analyze_angle_conversion_issues():
    """Analyse les problèmes potentiels dans la conversion natural -> halfangle"""
    
    print("="*80)
    print("DIAGNOSTIC DES PROBLEMES DE CONVERSION")
    print("="*80)
    
    # Paramètres similaires au code de production
    nphi_i = 2
    ntheta_i = 90
    nphi_o = 90
    ntheta_o = 90
    ntheta_h = 90
    ntheta_d = 90
    nphi_d = 180
    
    # Grilles d'angles
    phi_o_grid = np.linspace(0, 2*np.pi, nphi_o, dtype=np.float32)
    theta_o_grid = np.linspace(0, np.pi/2, ntheta_o, dtype=np.float32)
    phi_i_grid = np.linspace(0, 2*np.pi, nphi_i, dtype=np.float32)
    theta_i_grid = np.linspace(0, np.pi/2, ntheta_i, dtype=np.float32)
    
    print(f"\nGrilles d'angles:")
    print(f"  phi_i: {nphi_i} points de {np.degrees(phi_i_grid[0]):.1f}° à {np.degrees(phi_i_grid[-1]):.1f}°")
    print(f"         Valeurs: {[f'{np.degrees(p):.1f}°' for p in phi_i_grid]}")
    print(f"  theta_i: {ntheta_i} points de {np.degrees(theta_i_grid[0]):.1f}° à {np.degrees(theta_i_grid[-1]):.1f}°")
    print(f"  phi_o: {nphi_o} points de {np.degrees(phi_o_grid[0]):.1f}° à {np.degrees(phi_o_grid[-1]):.1f}°")
    print(f"  theta_o: {ntheta_o} points de {np.degrees(theta_o_grid[0]):.1f}° à {np.degrees(theta_o_grid[-1]):.1f}°")
    
    # Problème 1: nphi_i = 2 est trop petit !
    print(f"\n⚠️  PROBLÈME 1: nphi_i = {nphi_i}")
    print(f"   Avec linspace(0, 2π, 2), on a seulement 2 directions azimutales:")
    print(f"   - phi_i[0] = {np.degrees(phi_i_grid[0]):.1f}° = 0°")
    print(f"   - phi_i[1] = {np.degrees(phi_i_grid[1]):.1f}° = 360° = 0° (équivalent)")
    print(f"   → Cela ne couvre PAS tout l'espace azimutal incident!")
    
    # Analyser quelques conversions
    print(f"\n" + "="*80)
    print("ANALYSE DE CONVERSIONS SPÉCIFIQUES")
    print("="*80)
    
    test_cases = [
        (45, 0, 45, 0, "Centre: θi=θo=45°, φi=φo=0°"),
        (45, 0, 45, 180, "Diagonal: θi=θo=45°, φi=0°, φo=180°"),
        (0, 0, 90, 0, "Grazing incident: θi=0°, θo=90°"),
        (90, 0, 0, 0, "Grazing outgoing: θi=90°, θo=0°"),
        (45, 90, 45, 270, "Perpendiculaire: θi=θo=45°, φi=90°, φo=270°"),
    ]
    
    theta_h_values = []
    theta_d_values = []
    phi_d_values = []
    out_of_bounds_count = 0
    
    for theta_i_deg, phi_i_deg, theta_o_deg, phi_o_deg, desc in test_cases:
        print(f"\n--- {desc} ---")
        
        theta_i = np.radians(theta_i_deg)
        phi_i = np.radians(phi_i_deg)
        theta_o = np.radians(theta_o_deg)
        phi_o = np.radians(phi_o_deg)
        
        print(f"  Input (natural): θi={theta_i_deg}°, φi={phi_i_deg}°, θo={theta_o_deg}°, φo={phi_o_deg}°")
        
        theta_h, phi_h, theta_d, phi_d = natural_to_halfangle(theta_i, phi_i, theta_o, phi_o)
        
        print(f"  Output (halfangle): θh={np.degrees(theta_h):.2f}°, φh={np.degrees(phi_h):.2f}°")
        print(f"                      θd={np.degrees(theta_d):.2f}°, φd={np.degrees(phi_d):.2f}°")
        
        # Vérifier les limites
        issues = []
        if theta_h < 0 or theta_h > np.pi/2:
            issues.append(f"θh={np.degrees(theta_h):.2f}° hors [0, 90°]")
        if theta_d < 0 or theta_d > np.pi/2:
            issues.append(f"θd={np.degrees(theta_d):.2f}° hors [0, 90°]")
            out_of_bounds_count += 1
        if phi_d < 0 or phi_d > np.pi:
            issues.append(f"φd={np.degrees(phi_d):.2f}° hors [0, 180°]")
            out_of_bounds_count += 1
        
        if issues:
            print(f"  ⚠️  HORS LIMITES: {', '.join(issues)}")
        else:
            print(f"  ✓ Dans les limites")
        
        # Calculer les indices BRDF
        theta_h_idx = int(np.clip((np.cos(theta_h)**2) * (ntheta_h - 1), 0, ntheta_h - 1))
        theta_d_idx = int(np.clip(theta_d / (np.pi/2) * (ntheta_d - 1), 0, ntheta_d - 1))
        phi_d_idx = int(np.clip(phi_d / np.pi * (nphi_d - 1), 0, nphi_d - 1))
        
        print(f"  BRDF indices: [θh={theta_h_idx}, θd={theta_d_idx}, φd={phi_d_idx}]")
        
        theta_h_values.append(np.degrees(theta_h))
        theta_d_values.append(np.degrees(theta_d))
        phi_d_values.append(np.degrees(phi_d))
    
    # Test sur un échantillon aléatoire
    print(f"\n" + "="*80)
    print("STATISTIQUES SUR ÉCHANTILLON ALÉATOIRE")
    print("="*80)
    
    n_samples = 1000
    theta_h_all = []
    theta_d_all = []
    phi_d_all = []
    theta_d_negative = 0
    phi_d_negative = 0
    theta_d_too_large = 0
    phi_d_too_large = 0
    
    np.random.seed(42)
    for _ in range(n_samples):
        theta_i = np.random.uniform(0, np.pi/2)
        phi_i = np.random.uniform(0, 2*np.pi)
        theta_o = np.random.uniform(0, np.pi/2)
        phi_o = np.random.uniform(0, 2*np.pi)
        
        theta_h, phi_h, theta_d, phi_d = natural_to_halfangle(theta_i, phi_i, theta_o, phi_o)
        
        theta_h_all.append(np.degrees(theta_h))
        theta_d_all.append(np.degrees(theta_d))
        phi_d_all.append(np.degrees(phi_d))
        
        if theta_d < 0:
            theta_d_negative += 1
        if theta_d > np.pi/2:
            theta_d_too_large += 1
        if phi_d < 0:
            phi_d_negative += 1
        if phi_d > np.pi:
            phi_d_too_large += 1
    
    print(f"\nSur {n_samples} échantillons aléatoires:")
    print(f"\n  θh (half theta):")
    print(f"    min={np.min(theta_h_all):.2f}°, max={np.max(theta_h_all):.2f}°")
    print(f"\n  θd (diff theta):")
    print(f"    min={np.min(theta_d_all):.2f}°, max={np.max(theta_d_all):.2f}°")
    print(f"    Valeurs négatives: {theta_d_negative} ({theta_d_negative/n_samples*100:.1f}%)")
    print(f"    Valeurs > 90°: {theta_d_too_large} ({theta_d_too_large/n_samples*100:.1f}%)")
    print(f"\n  φd (diff phi):")
    print(f"    min={np.min(phi_d_all):.2f}°, max={np.max(phi_d_all):.2f}°")
    print(f"    Valeurs négatives: {phi_d_negative} ({phi_d_negative/n_samples*100:.1f}%)")
    print(f"    Valeurs > 180°: {phi_d_too_large} ({phi_d_too_large/n_samples*100:.1f}%)")
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(theta_h_all, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('θh (degrés)')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution de θh (half theta)')
    axes[0].axvline(0, color='r', linestyle='--', label='Limite inf')
    axes[0].axvline(90, color='r', linestyle='--', label='Limite sup')
    axes[0].legend()
    
    axes[1].hist(theta_d_all, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('θd (degrés)')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution de θd (diff theta)')
    axes[1].axvline(0, color='r', linestyle='--', label='Limite inf')
    axes[1].axvline(90, color='r', linestyle='--', label='Limite sup')
    axes[1].legend()
    
    axes[2].hist(phi_d_all, bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('φd (degrés)')
    axes[2].set_ylabel('Fréquence')
    axes[2].set_title('Distribution de φd (diff phi)')
    axes[2].axvline(0, color='r', linestyle='--', label='Limite inf')
    axes[2].axvline(180, color='r', linestyle='--', label='Limite sup')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('angle_conversion_diagnostics.png', dpi=150)
    print(f"\n✓ Graphiques sauvegardés dans 'angle_conversion_diagnostics.png'")
    plt.show()
    
    # Recommandations
    print(f"\n" + "="*80)
    print("RECOMMANDATIONS")
    print("="*80)
    print(f"\n1. ⚠️  CRITIQUE: nphi_i = 2 est insuffisant!")
    print(f"   → Augmenter à au moins nphi_i = 32 ou 64 pour une meilleure couverture")
    print(f"\n2. Gestion des angles hors limites:")
    print(f"   → {theta_d_negative + theta_d_too_large + phi_d_negative + phi_d_too_large} valeurs hors limites sur {n_samples} ({(theta_d_negative + theta_d_too_large + phi_d_negative + phi_d_too_large)/n_samples*100:.1f}%)")
    print(f"   → Le clipping crée des artefacts (valeurs blanches aux contours)")
    print(f"\n3. Vérifier la normalisation des valeurs BRDF:")
    print(f"   → Le remapping log-space pourrait saturer à 1.0 (blanc)")

if __name__ == "__main__":
    analyze_angle_conversion_issues()
