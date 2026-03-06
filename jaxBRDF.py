import jax.numpy as jnp

def normalize(v):
    return v / (jnp.linalg.norm(v, axis=0, keepdims=True)+1e-4)

def rotate_normal_jax(vec, theta):
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    
    x = vec[0]
    y = vec[1]
    z = vec[2]
    
    new_x = cos_t * x - sin_t * y
    new_y = sin_t * x + cos_t * y
    
    out = jnp.stack([new_x, new_y, z], axis=0)
    return out

def rotate_binormal_jax(vec, theta):
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    
    x = vec[0]
    y = vec[1]
    z = vec[2]
    
    new_x =  cos_t * x + sin_t * z
    new_z = -sin_t * x + cos_t * z
    
    return jnp.stack([new_x, y, new_z], axis=0)

def rusinkiewicz_to_LV_jax(theta_h, phi_h, theta_d, phi_d):
    """
    Convert half/difference parameterization to Cartesian wi and wo.

    Returns:
        jnp.array shape (6,)
        [wi_x, wi_y, wi_z, wo_x, wo_y, wo_z]
    """

    # --- Half vector (spherical → Cartesian)
    half = jnp.array([
        jnp.sin(theta_h) * jnp.cos(phi_h),
        jnp.sin(theta_h) * jnp.sin(phi_h),
        jnp.cos(theta_h)
    ])
    # --- Difference vector (local frame)
    wi = jnp.array([
        jnp.sin(theta_d) * jnp.cos(phi_d),
        jnp.sin(theta_d) * jnp.sin(phi_d),
        jnp.cos(theta_d)
    ])
    # --- Rotate into world frame
    wi = rotate_binormal_jax(wi, theta_h)
    wi = rotate_normal_jax(wi, phi_h)

    # --- Reflect wi about half to get wo
    dot = jnp.sum(wi*half, axis=0)
    wo = -wi + 2.0 * dot * half

    N = jnp.array([0.0, 0.0, 1.0]).reshape(3, 1, 1, 1)
    N = jnp.broadcast_to(N,half.shape)
    X = jnp.array([1,0,0]).reshape(3, 1, 1, 1)
    X = jnp.broadcast_to(X,half.shape)
    Y = jnp.array([0,1,0]).reshape(3, 1, 1, 1)
    Y = jnp.broadcast_to(Y,half.shape)

    return wi, wo, N, X, Y

def clamp_jax(x, a=0.0, b=1.0):
    return jnp.clip(x,a,b)

def SchlickFresnel_jax(u):
    m = clamp_jax(1 - u)
    return m**5

def GTR1_jax(NdotH, a):
    a2 = a**2
    t = 1 + (a2 - 1) * NdotH**2
    return (a2 - 1) / (jnp.pi * jnp.log(a2) * t)

def GTR2_jax(NdotH, a):
    a2 = a**2
    t = 1 + (a2 - 1) * NdotH**2
    return a2 / (jnp.pi * t**2)

def GTR2_aniso_jax(NdotH, HdotX, HdotY, ax, ay):
    return 1 / (jnp.pi * ax * ay * ((HdotX / ax) ** 2 + (HdotY / ay) ** 2 + NdotH**2) ** 2)

def smithG_GGX_jax(NdotV, alphaG):
    a2 = alphaG**2
    b2 = NdotV**2
    return 1 / (NdotV + jnp.sqrt(a2 + b2 - a2 * b2))

def smithG_GGX_aniso_jax(NdotV, VdotX, VdotY, ax, ay):
    return 1 / (NdotV + jnp.sqrt((VdotX * ax) ** 2 + (VdotY * ay) ** 2 + NdotV**2))

def mon2lin_jax(x):
    return x**2.2

def mix_jax(a, b, t):
    return a * (1 - t) + b * t

def BRDF_jax(
    L,
    V,
    N,
    X,
    Y,
    baseColor=jnp.array([0.82, 0.67, 0.16]),
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
    NdotL = jnp.sum(N * L, axis=0)
    NdotV = jnp.sum(N * V, axis=0)

    # équivalent du masque en numpy
    valid = (NdotL > 0.0) & (NdotV > 0.0)

    H = L + V
    H = normalize(H)
    NdotH = jnp.sum(N * H, axis=0)
    LdotH = jnp.sum(L * H, axis=0)
    # Eviter les valeurs nulles (problème avec le gradient) 
    NdotL = jnp.clip(NdotL, 1e-4, 1.0)
    NdotV = jnp.clip(NdotV, 1e-4, 1.0)
    NdotH = jnp.clip(NdotH, 1e-4, 1.0)
    LdotH = jnp.clip(LdotH, 1e-4, 1.0)
    
    Cdlin = mon2lin_jax(baseColor)
    Cdlum = 0.3 * Cdlin[0] + 0.6 * Cdlin[1] + 0.1 * Cdlin[2]
    taille_vec = (3,1,1,1)
    Cdlum = jnp.clip(Cdlum,1e-4)
    Ctint = jnp.where(Cdlum > 0.0, Cdlin / Cdlum, jnp.ones_like(Cdlin))
    Ctint = jnp.reshape(Ctint, taille_vec)
    Cspec0 = mix_jax(specular * 0.08 * jnp.ones(taille_vec), specular * 0.08 * Ctint, specularTint)
    Cdlin = jnp.reshape(Cdlin,taille_vec)
    Cspec0 = mix_jax(Cspec0, Cdlin, metallic)
    Csheen = mix_jax(jnp.ones(taille_vec), Ctint, sheenTint)
    
    FL = SchlickFresnel_jax(NdotL)
    FV = SchlickFresnel_jax(NdotV)
    Fd90 = 0.5 + 2.0 * (LdotH**2) * roughness
    Fd = mix_jax(1.0, Fd90, FL) * mix_jax(1.0, Fd90, FV)

    Fss90 = (LdotH**2) * roughness
    Fss = mix_jax(1.0, Fss90, FL) * mix_jax(1.0, Fss90, FV)
    ss = 1.25 * (Fss * (1.0 / (NdotL + NdotV + 1e-4) - 0.5) + 0.5)

    aspect = jnp.sqrt(1.0 - anisotropic * 0.9)
    ax = jnp.maximum(0.001, roughness**2 / aspect)
    ay = jnp.maximum(0.001, roughness**2 * aspect)

    Hx = jnp.sum(H * X, axis=0)
    Hy = jnp.sum(H * Y, axis=0)
    Lx = jnp.sum(L * X, axis=0)
    Ly = jnp.sum(L * Y, axis=0)
    Vx = jnp.sum(V * X, axis=0)
    Vy = jnp.sum(V * Y, axis=0)

    Ds = GTR2_aniso_jax(NdotH, Hx, Hy, ax, ay)
    FH = SchlickFresnel_jax(LdotH)
    Fs = mix_jax(Cspec0, jnp.ones(taille_vec), FH)
    Gs = (
        smithG_GGX_aniso_jax(NdotL, Lx, Ly, ax, ay)
        * smithG_GGX_aniso_jax(NdotV, Vx, Vy, ax, ay)
    )
    Fsheen = FH * sheen * Csheen

    Dr = GTR1_jax(NdotH, mix_jax(0.1, 0.001, clearcoatGloss))
    Fr = mix_jax(0.04, 1.0, FH)
    Gr = smithG_GGX_jax(NdotL, 0.25) * smithG_GGX_jax(NdotV, 0.25)

    diffuse = (1.0 / jnp.pi) * mix_jax(Fd, ss, subsurface) * Cdlin * (1.0 - metallic) + Fsheen
    spec = Gs[None, ...] * Fs * Ds[None, ...]
    clear = 0.25 * clearcoat * Gr * Fr * Dr
    
    out = diffuse + spec + clear[None, ...]
    out = jnp.where(valid[None, ...], out, 0)
    return out