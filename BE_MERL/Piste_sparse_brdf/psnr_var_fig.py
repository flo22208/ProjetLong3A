import pandas as pd
import matplotlib.pyplot as plt

csv_path = "./output/resultats_var.csv"   # <-- ton fichier
sep = ","                    # ou ";"
encoding = "utf-8"

df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
df = df.rename(columns=str.strip)

# Convertir en float (gère aussi les virgules décimales)
df["size"] = df["size"].astype(str).str.replace(",", ".", regex=False).astype(float)
df["PSNR"] = df["PSNR"].astype(str).str.replace(",", ".", regex=False).astype(float)

# Trier par size
df = df.sort_values("size")

plt.figure(figsize=(10, 6))
plt.plot(df["size"], df["PSNR"], marker="o", linewidth=1.5)

plt.xlabel("size")
plt.ylabel("PSNR (dB)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig("./output/psnr_var_plot.png", dpi=300, bbox_inches="tight")