import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import utilsBRDFDisney

def display_brdf_slice(brdf, phi_d_idx=90, mode="reinhard", exposure=1.0):
    """
    Display a 2D slice of the BRDF (theta_h x theta_d) as an RGB image at fixed phi_d.
    brdf: (H, D, P, 3) float16 array
    phi_d_idx: phi_d index (0 to MAX_PHI_D-1), default 90
    """

    slice_ = brdf[:, :, phi_d_idx].astype(np.float32)  # (H, D, 3)

    if mode == "reinhard":
        x = slice_ * exposure
        img = x / (1.0 + x)
    elif mode == "log":
        img = np.log1p(slice_)
        img /= img.max() if img.max() > 0 else 1.0
    elif mode == "clip":
        img = np.clip(slice_, 0, 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    img = np.clip(img, 0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, origin="lower",
               extent=[0, 90, 0, 90],  # theta_d, theta_h both 0-90°
               aspect="equal")
    plt.xlabel("theta_d (degrees)")
    plt.ylabel("theta_h (degrees)")
    plt.title(f"BRDF slice at phi_d index {phi_d_idx} [{mode}]")
    plt.colorbar(label="Tonemapped intensity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    utilsBRDFDisney.load_brdf(index=int(7))
    display_brdf_slice(utilsBRDFDisney.brdf, phi_d_idx=179, mode="reinhard")