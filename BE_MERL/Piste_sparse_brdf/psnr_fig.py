import pandas as pd
import matplotlib.pyplot as plt

# --- Paramètres ---
csv_path = "./output/resultats.csv"          # <-- mets le chemin de ton fichier
sep = ","                      # ou ";" selon ton CSV
encoding = "utf-8"

# --- Lecture ---
df = pd.read_csv(csv_path, sep=sep, encoding=encoding)

# Vérif colonnes attendues
df = df.rename(columns=str.strip)
assert "materiau" in df.columns and "PSNR" in df.columns, df.columns

# Convert PSNR en numérique au cas où (virgules -> points)
df["PSNR"] = (
    df["PSNR"].astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

# --- Tri par PSNR (comme souvent sur ce genre de figure) ---
df_sorted = df.sort_values("PSNR", ascending=True).reset_index(drop=True)

# --- Plot ---
plt.figure(figsize=(22, 7))
plt.plot(df_sorted["PSNR"].values, marker="o", linewidth=1.5, markersize=3)

plt.ylabel("Average rendering PSNR (dB)")
plt.xticks(
    ticks=range(len(df_sorted)),
    labels=df_sorted["materiau"].astype(str).values,
    rotation=90,
    ha="center",
    fontsize=8
)

plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig("./output/psnr_plot.png", dpi=300, bbox_inches="tight")
