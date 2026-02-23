import torch
import torch.nn.functional as F
import math

PI = math.pi

# ------------------------------------------------------------
# Helper functions (vectorized)
# ------------------------------------------------------------

def schlick_fresnel(u):
    return (1 - u).clamp(0, 1) ** 5


def GTR1(NdotH, a):
    a2 = a * a
    t = 1 + (a2 - 1) * NdotH**2
    return torch.where(
        a >= 1,
        torch.full_like(NdotH, 1 / PI),
        (a2 - 1) / (PI * torch.log(a2) * t)
    )


def GTR2(NdotH, a):
    a2 = a * a
    t = 1 + (a2 - 1) * NdotH**2
    return a2 / (PI * t**2)

def GTR2_aniso(NdotH, HdotX, HdotY, ax, ay):
    return 1.0 / (
        PI * ax * ay *
        ((HdotX / ax) ** 2 + (HdotY / ay) ** 2 + NdotH**2) ** 2
    )

def smithG_GGX(NdotV, alpha):
    a2 = alpha * alpha
    b2 = NdotV * NdotV
    return 1 / (NdotV + torch.sqrt(a2 + b2 - a2 * b2 + 1e-8))

def smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay):
    return 1 / (
        NdotV + torch.sqrt(
            (VdotX * ax) ** 2 +
            (VdotY * ay) ** 2 +
            NdotV**2 + 1e-8
        )
    )

def mix(a, b, t):
    return a * (1 - t) + b * t


def mon2lin(x):
    return x ** 2.2

def create_rusinkiewicz_grid(device):
    D, H, W = 90, 90, 180

    theta_h = torch.linspace(0, PI/2, D, device=device)
    theta_d = torch.linspace(0, PI/2, H, device=device)
    phi_d   = torch.linspace(0, PI, W, device=device)

    theta_h, theta_d, phi_d = torch.meshgrid(
        theta_h, theta_d, phi_d, indexing='ij'
    )

    # Half vector (phi_h = 0)
    half = torch.stack([
        torch.sin(theta_h),
        torch.zeros_like(theta_h),
        torch.cos(theta_h)
    ], dim=-1)  # (D,H,W,3)

    # Difference vector in local frame
    wi = torch.stack([
        torch.sin(theta_d) * torch.cos(phi_d),
        torch.sin(theta_d) * torch.sin(phi_d),
        torch.cos(theta_d)
    ], dim=-1)

    # Rotate wi by theta_h around Y
    cos_th = torch.cos(theta_h)
    sin_th = torch.sin(theta_h)

    wi_rot = torch.zeros_like(wi)
    wi_rot[..., 0] = cos_th * wi[..., 0] + sin_th * wi[..., 2]
    wi_rot[..., 1] = wi[..., 1]
    wi_rot[..., 2] = -sin_th * wi[..., 0] + cos_th * wi[..., 2]

    wi = wi_rot

    # Reflect wi about half to get wo
    dot = (wi * half).sum(-1, keepdim=True)
    wo = -wi + 2 * dot * half

    N = torch.tensor([0.,0.,1.], device=device)
    N = N.view(1,1,1,3)

    return wi, wo, N

def disney_brdf_torch(params, wi, wo, N):
    """
    params: (B,12) in [0,1]
    wi, wo: (D,H,W,3)
    N: (1,1,1,3)

    returns: (B,D,H,W,3)
    """

    device = params.device
    B = params.shape[0]
    D, H, W, _ = wi.shape

    wi = wi.unsqueeze(0).expand(B,-1,-1,-1,-1)
    wo = wo.unsqueeze(0).expand(B,-1,-1,-1,-1)
    N  = N.expand(B,D,H,W,3)

    # Tangent frame (fixed)
    X = torch.tensor([1.,0.,0.], device=device).view(1,1,1,1,3)
    Y = torch.tensor([0.,1.,0.], device=device).view(1,1,1,1,3)

    # unpack parameters
    baseColor = params[:,0:3].view(B,1,1,1,3)
    metallic = params[:,3:4].view(B,1,1,1,1)
    subsurface = params[:,4:5].view(B,1,1,1,1)
    specular = params[:,5:6].view(B,1,1,1,1)
    roughness = params[:,6:7].view(B,1,1,1,1)
    specularTint = params[:,7:8].view(B,1,1,1,1)
    anisotropic = torch.zeros_like(roughness)  # fixed to 0
    sheen = params[:,8:9].view(B,1,1,1,1)
    sheenTint = params[:,9:10].view(B,1,1,1,1)
    clearcoat = params[:,10:11].view(B,1,1,1,1)
    clearcoatGloss = params[:,11:12].view(B,1,1,1,1)

    # dot products
    NdotL = (N * wi).sum(-1, keepdim=True).clamp(min=0)
    NdotV = (N * wo).sum(-1, keepdim=True).clamp(min=0)

    Hvec = F.normalize(wi + wo, dim=-1)
    NdotH = (N * Hvec).sum(-1, keepdim=True).clamp(min=0)
    LdotH = (wi * Hvec).sum(-1, keepdim=True).clamp(min=0)

    HdotX = (Hvec * X).sum(-1, keepdim=True)
    HdotY = (Hvec * Y).sum(-1, keepdim=True)

    LdotX = (wi * X).sum(-1, keepdim=True)
    LdotY = (wi * Y).sum(-1, keepdim=True)

    VdotX = (wo * X).sum(-1, keepdim=True)
    VdotY = (wo * Y).sum(-1, keepdim=True)

    # color
    Cdlin = mon2lin(baseColor)
    Cdlum = (0.3*Cdlin[...,0] + 0.6*Cdlin[...,1] + 0.1*Cdlin[...,2]).unsqueeze(-1)

    Ctint = torch.where(Cdlum > 0, Cdlin / Cdlum, torch.ones_like(Cdlin))

    Cspec0 = mix(specular * 0.08, specular * 0.08 * Ctint, specularTint)
    Cspec0 = mix(Cspec0, Cdlin, metallic)

    Csheen = mix(torch.ones_like(Cdlin), Ctint, sheenTint)

    # -----------------------
    # Diffuse + Subsurface
    # -----------------------

    FL = schlick_fresnel(NdotL)
    FV = schlick_fresnel(NdotV)

    Fd90 = 0.5 + 2 * LdotH**2 * roughness
    Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV)

    Fss90 = LdotH**2 * roughness
    Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV)
    ss = 1.25 * (Fss * (1 / (NdotL + NdotV + 1e-8) - 0.5) + 0.5)

    

    FH = schlick_fresnel(LdotH)
    Fsheen = FH * sheen * Csheen

    diffuse = (
        (1/PI) *
        mix(Fd, ss, subsurface) *
        Cdlin *
        (1 - metallic)
    )  + Fsheen

    # -----------------------
    # Specular (anisotropic formulation preserved)
    # -----------------------

    aspect = torch.sqrt(1 - anisotropic * 0.9)
    ax = torch.clamp(roughness**2 / aspect, min=0.001)
    ay = torch.clamp(roughness**2 * aspect, min=0.001)

    Ds = GTR2_aniso(NdotH, HdotX, HdotY, ax, ay)

    Fs = mix(Cspec0, torch.ones_like(Cspec0), FH)

    Gs = (
        smithG_GGX_aniso(NdotL, LdotX, LdotY, ax, ay) *
        smithG_GGX_aniso(NdotV, VdotX, VdotY, ax, ay)
    )

    spec = Gs * Fs * Ds

    # -----------------------
    # Clearcoat
    # -----------------------

    Dr = GTR1(NdotH, mix(0.1, 0.001, clearcoatGloss))
    Fr = mix(0.04, 1.0, FH)
    Gr = smithG_GGX(NdotL, 0.25) * smithG_GGX(NdotV, 0.25)

    clear = 0.25 * clearcoat * Gr * Fr * Dr

    return diffuse + spec + clear



if __name__ == "__main__":
    device = "cuda"

    wi, wo, N = create_rusinkiewicz_grid(device)

    params = torch.tensor(
        [[
            0.82, 0.67, 0.16,   # baseColor (R,G,B)
            0.0,                # metallic
            0.0,                # subsurface
            0.5,                # specular
            0.5,                # roughness
            0.0,                # specularTint
            0.0,                # sheen
            0.5,                # sheenTint
            0.0,                # clearcoat
            1.0                 # clearcoatGloss
        ]],
        dtype=torch.float32,
        device=device
    )

    torch_output = disney_brdf_torch(params, wi, wo, N)[0].cpu().numpy()
