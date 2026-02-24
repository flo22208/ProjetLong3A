import numpy as np
from merlFunctions import *
from config import Config
import argparse


def psnr(a: np.ndarray, b: np.ndarray, max_i: float) -> float:
    diff = a - b
    mse = np.mean(diff * diff)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((max_i * max_i) / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="reference MERL .binary")
    ap.add_argument("test", help="test MERL .binary")
    args = ap.parse_args()

    ref = readMERLBRDF(args.ref)
    test = readMERLBRDF(args.test)

    # Determine MAX_I
    max_i = float(np.max(ref))

    # Overall PSNR
    overall = psnr(ref, test, max_i=max_i)

    # Per-channel PSNR
    psnr_r = psnr(ref[..., 0], test[..., 0], max_i=max_i)
    psnr_g = psnr(ref[..., 1], test[..., 1], max_i=max_i)
    psnr_b = psnr(ref[..., 2], test[..., 2], max_i=max_i)

    print(f"MAX_I = {max_i}")

    print(f"PSNR overall: {overall:.4f} dB")
    print(f"PSNR R:       {psnr_r:.4f} dB")
    print(f"PSNR G:       {psnr_g:.4f} dB")
    print(f"PSNR B:       {psnr_b:.4f} dB")


if __name__ == "__main__":
    main()
