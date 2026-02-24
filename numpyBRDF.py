import numpy as np

# ==========================
# Parameters
# ==========================

PI = np.pi

# ==========================
# Helper functions for Numpy Disney BRDF
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
    # if abs(NdotL + NdotV) < 1e-6: print("caca")
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

# ==========================
# Numpy rusinkiewicz to LV
# ==========================

def rotate_normal(vec, theta):
    """
    Rotate vector around z-axis by theta radians.
    Equivalent to Eigen::AngleAxis(theta, (0,0,1))
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vec2 = np.copy(vec)
    vec2[0] = cos_t * vec[0] - sin_t * vec[1]
    vec2[1] = sin_t * vec[0] + cos_t * vec[1]
    return vec2


def rotate_binormal(vec, theta):
    """
    Rotate vector around y-axis by theta radians.
    Equivalent to Eigen::AngleAxis(theta, (0,1,0))
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vec2 = np.copy(vec)
    vec2[0] = cos_t * vec[0] + sin_t * vec[2]
    vec2[2] = -sin_t * vec[0] + cos_t * vec[2]
    return vec2


def rusinkiewicz_to_LV(theta_h, phi_h, theta_d, phi_d):
    """
    Convert half/difference parameterization to Cartesian wi and wo.

    Returns:
        np.array shape (6,)
        [wi_x, wi_y, wi_z, wo_x, wo_y, wo_z]
    """

    # --- Half vector (spherical → Cartesian)
    half = np.array([
        np.sin(theta_h) * np.cos(phi_h),
        np.sin(theta_h) * np.sin(phi_h),
        np.cos(theta_h)
    ])

    # --- Difference vector (local frame)
    wi = np.array([
        np.sin(theta_d) * np.cos(phi_d),
        np.sin(theta_d) * np.sin(phi_d),
        np.cos(theta_d)
    ])

    # --- Rotate into world frame
    wi = rotate_binormal(wi, theta_h)
    wi = rotate_normal(wi, phi_h)

    # --- Reflect wi about half to get wo
    dot = np.dot(wi, half)
    wo = -wi + 2.0 * dot * half

    N = np.array([0.0, 0.0, 1.0])
    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 1.0, 0.0])

    return wi, wo, N, X, Y


if __name__ == "__main__":

    L = np.array([0.85355339, 0.5       , 0.14644661])
    V = np.array([0.14644661,-0.5       , 0.85355339])
    N = np.array([0, 0, 1])
    X = np.array([1, 0, 0])
    Y = np.array([0, 1, 0])

    res = BRDF(L, V, N, X, Y)

    print(res)
