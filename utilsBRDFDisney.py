import numpy as np
import time
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


def rusinkiewicz_to_LV(theta_h, theta_d, phi_d):
    """
    Convert Rusinkiewicz coordinates (with phi_h = 0)
    to light and view direction vectors.

    All angles in radians.
    """

    # Surface frame
    N = np.array([0.0, 0.0, 1.0])
    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 1.0, 0.0])

    # ----------------------
    # 1. Construct half vector H
    # φ_h = 0
    # ----------------------
    H = np.array([np.sin(theta_h), 0.0, np.cos(theta_h)])

    # ----------------------
    # 2. Build orthonormal frame around H
    # ----------------------
    tangent = np.cross(N, H)
    if np.linalg.norm(tangent) != 0:
        tangent /= np.linalg.norm(tangent)

    bitangent = np.cross(H, tangent)

    # ----------------------
    # 3. Construct difference vector D
    # ----------------------
    D = (
        np.sin(theta_d) * np.cos(phi_d) * tangent
        + np.sin(theta_d) * np.sin(phi_d) * bitangent
        + np.cos(theta_d) * H
    )

    # ----------------------
    # 4. Compute L and V
    # ----------------------
    L = D
    V = 2 * np.dot(D, H) * H - D  # reflect D about H

    # Normalize
    L /= np.linalg.norm(L)
    V /= np.linalg.norm(V)

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
    
    if (diffuse[0] + clear + spec[0] > 1) or (diffuse[1] + clear + spec[1] > 1) or (
        diffuse[2] + clear + spec[2] > 1
    ):
        return 1,1,1
        
    return diffuse + spec + clear

# read brdf file number 0
sample = np.load(f"{folder_brdfs}/brdf_1.npz", allow_pickle=True)
brdf = sample["brdf"]  # shape (RES_THETA_H, RES_THETA_D, RES_PHI_D, 3)
params = sample["params"]

# read angles file
angles_file = np.load(f"{folder_brdfs}/angles.npz")
theta_hs = angles_file["theta_h"]
theta_ds = angles_file["theta_d"]
phi_ds = angles_file["phi_d"]

def brdf_for_rendering(angles):
    """ angles : size (N,3) """

    res = np.zeros(angles.shape)    

    for (i, triplet) in enumerate(angles):
        phi_d, theta_d, theta_h = triplet

        # Find closest triplet in theta_hs, theta_ds and phi_ds
        phi_d_idx = np.argmin(np.abs(phi_ds - phi_d))
        theta_d_idx = np.argmin(np.abs(theta_ds - theta_d))
        theta_h_idx = np.argmin(np.abs(theta_hs - theta_h))

        # Get the corresponding RGB value
        rgb = np.array(brdf[theta_h_idx, theta_d_idx, phi_d_idx])
        res[i] = rgb
    
    return res

if __name__ == "__main__":
    brdf_for_rendering(np.array([[0.5, 0.1, 0.321],[2.8, 1.8, 0.321]]))

    L = np.array([0, 0, 1])
    V = np.array([0, 0, 1])
    N = np.array([0, 0, 1])
    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])
    res = BRDF(L, V, N, X, Y)

