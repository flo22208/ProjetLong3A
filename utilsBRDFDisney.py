import numpy as np
import time
import merlDB.database as db
# ==========================
# Parameters (defaults from shader)
# ==========================

PI = np.pi
folder_brdfs = "brdfs_disney/"


# ==========================
# Helper functions
# ==========================
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def SchlickFresnel(u):
    m = clamp(1 - u)
    return m**5


def GTR1(NdotH, a):
    if a >= 1:
        return 1 / PI
    a2 = a**2
    t = 1 + (a2 - 1) * NdotH**2
    return (a2 - 1) / (PI * np.log(a2) * t)


def GTR2(NdotH, a):
    a2 = a**2
    t = 1 + (a2 - 1) * NdotH**2
    return a2 / (PI * t**2)


def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
    return 1 / (PI * ax * ay * ((HdotX / ax) ** 2 + (HdotY / ay) ** 2 + NdotH**2) ** 2)


def smithG_GGX(NdotV, alphaG):
    a2 = alphaG**2
    b2 = NdotV**2
    return 1 / (NdotV + np.sqrt(a2 + b2 - a2 * b2))


def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
    return 1 / (NdotV + np.sqrt((VdotX * ax) ** 2 + (VdotY * ay) ** 2 + NdotV**2))


def mon2lin(x):
    return x**2.2


def mix(a, b, t):
    return a * (1 - t) + b * t

from convert_half_benj import (
    halfangle_to_natural,
    spherical_to_cartesian,
    complete_basis,
)

def rusinkiewicz_to_LV(theta_h, theta_d, phi_d):
    # 1. On fixe phi_h = 0 (convention Rusinkiewicz utilisée chez toi)
    phi_h = 0.0

    # 2. On passe des coordonnées half-angle (Rusinkiewicz)
    #    aux coordonnées "naturelles" (theta_i, phi_i, theta_o, phi_o)
    theta_i, phi_i, theta_o, phi_o = halfangle_to_natural(
        theta_h, phi_h, theta_d, phi_d
    )

    # 3. On construit wi, wo dans le repère tangent
    wi_tangent = spherical_to_cartesian(theta_i, phi_i)
    wo_tangent = spherical_to_cartesian(theta_o, phi_o)

    # 4. On utilise EXACTEMENT le même repère que get_angles :
    #    n = (0,0,1), t = (1,0,0), et la même complete_basis
    N = np.array([0.0, 0.0, 1.0])
    t = np.array([1.0, 0.0, 0.0])
    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 1.0, 0.0])
    M_tangent = complete_basis(N, t)

    # 5. On repasse dans le repère monde
    L = M_tangent.T @ wi_tangent
    V = M_tangent.T @ wo_tangent
    # 1. Empêcher L ou V d’être sous la surface
    if np.dot(N, L) <= 1e-6:
        L = L - 2 * np.dot(N, L) * N
        L /= np.linalg.norm(L)

    # if np.dot(N, V) <= 1e-6:
    #     V = V - 2 * np.dot(N, V) * N
    #     V /= np.linalg.norm(V)

    # 2. Empêcher le cas D = 0 (L = V)
    # if np.linalg.norm(L - V) < 1e-6:
    #     # On pousse légèrement V
    #     V = (V + 1e-3 * N)
    #     V /= np.linalg.norm(V)

    return L, V, N, X, Y


# ==========================
# Main BRDF function
# ==========================
def BRDF(
    L,
    V,
    N,
    X,
    Y,
    baseColor=np.array([0.82, 0.67, 0.16]),
    metallic=0.0,
    subsurface=0.0,
    specular=0.5,
    roughness=0.5,
    specularTint=0.0,
    anisotropic=0.0,
    sheen=0.0,
    sheenTint=0.5,
    clearcoat=0.0,
    clearcoatGloss=1.0,
):
    NdotL = np.dot(N, L)
    NdotV = np.dot(N, V)
    if NdotL < 0 or NdotV < 0:
        return np.zeros(3)

    H = L + V
    H = H / np.linalg.norm(H)
    NdotH = np.dot(N, H)
    LdotH = np.dot(L, H)

    Cdlin = mon2lin(baseColor)
    Cdlum = 0.3 * Cdlin[0] + 0.6 * Cdlin[1] + 0.1 * Cdlin[2]

    Ctint = Cdlin / Cdlum if Cdlum > 0 else np.ones(3)
    Cspec0 = mix(specular * 0.08 * np.ones(3), specular * 0.08 * Ctint, specularTint)
    Cspec0 = mix(Cspec0, Cdlin, metallic)
    Csheen = mix(np.ones(3), Ctint, sheenTint)

    FL = SchlickFresnel(NdotL)
    FV = SchlickFresnel(NdotV)
    Fd90 = 0.5 + 2 * LdotH**2 * roughness
    Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV)

    Fss90 = LdotH**2 * roughness
    Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV)
    if abs(NdotL + NdotV - 0.5) < 1e-6: print("caca")
    ss = 1.25 * (Fss * (1 / (NdotL + NdotV) - 0.5) + 0.5)

    aspect = np.sqrt(1 - anisotropic * 0.9)
    ax = max(0.001, roughness**2 / aspect)
    ay = max(0.001, roughness**2 * aspect)

    Ds = GTR2_aniso(NdotH, np.dot(H, X), np.dot(H, Y), ax, ay)
    FH = SchlickFresnel(LdotH)
    Fs = mix(Cspec0, np.ones(3), FH)
    Gs = smithG_GGX_aniso(NdotL, np.dot(L, X), np.dot(L, Y), ax, ay) * smithG_GGX_aniso(
        NdotV, np.dot(V, X), np.dot(V, Y), ax, ay
    )

    Fsheen = FH * sheen * Csheen

    Dr = GTR1(NdotH, mix(0.1, 0.001, clearcoatGloss))
    Fr = mix(0.04, 1.0, FH)
    Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25)

    diffuse = (1 / PI) * mix(Fd, ss, subsurface) * Cdlin * (1 - metallic) + Fsheen
    spec = Gs * Fs * Ds
    clear = 0.25 * clearcoat * Gr * Fr * Dr

    # if (diffuse[0] + clear + spec[0] > 1) or (diffuse[1] + clear + spec[1] > 1) or (
    #     diffuse[2] + clear + spec[2] > 1
    # ):
    #     return 1,1,1

    return diffuse + spec + clear


brdf = None
params = None
theta_hs = None
theta_ds = None
phi_ds = None


def load_brdf(index=2):
    global brdf, params, theta_hs, theta_ds, phi_ds

    # Load selected BRDF
    sample = np.load(f"{folder_brdfs}/brdf_{index}.npz", allow_pickle=True)
    brdf = sample["brdf"]
    brdf = brdf / (1.0 + brdf)
    params = sample["params"]

    # Load angles (only once)
    angles_file = np.load(f"{folder_brdfs}/angles.npz")
    theta_hs = angles_file["theta_h"]
    theta_ds = angles_file["theta_d"]
    phi_ds = angles_file["phi_d"]

    print(f"Loaded BRDF {index}")
    print(params)


def brdf_for_rendering(angles):
    """angles : size (N,3)"""

    res = np.zeros(angles.shape)

    for i, triplet in enumerate(angles):
        phi_d, theta_d, theta_h = triplet

        # Find closest triplet in theta_hs, theta_ds and phi_ds
        phi_d_idx = np.argmin(np.abs(phi_ds - phi_d))
        theta_d_idx = np.argmin(np.abs(theta_ds - theta_d))
        theta_h_idx = np.argmin(np.abs(theta_hs - theta_h))

        # Get the corresponding RGB value
        rgb = np.array(brdf[theta_h_idx, theta_d_idx, phi_d_idx])
        res[i] = rgb

    return res


if __name__ == "__main__":
    res = rusinkiewicz_to_LV(np.pi / 4, np.pi / 4, np.pi / 4)
    normal = np.array([res[2]])
    light = np.array([res[0]])
    view = np.array([res[1]])
    print(light, view, normal)
    print(db.rusinkiewicz_angles(normal, light, view))
    # load_brdf(index=2)
    # brdf_for_rendering(np.array([[0.5, 0.1, 0.321],[2.8, 1.8, 0.321]]))

    L = np.array([0.3,  0.0, 0.8660254])
    V = np.array([-0.3, 0.0, 0.8660254])
    N = np.array([0, 0, 1])
    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])
    print(np.array([N]).shape)
    phi_d, theta_d, theta_h = db.rusinkiewicz_angles(
        np.array([N]), np.array([L]), np.array([V])
    )
    print(rusinkiewicz_to_LV(theta_h[0], theta_d[0], phi_d[0]))

    res = BRDF(L, V, N, X, Y)
