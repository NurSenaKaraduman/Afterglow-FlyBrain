import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# --------------------------------------------------
# DOSYA YOLLARI
# FILE PATHS
# --------------------------------------------------

folder = Path(
    "results/flypaint_training"
)

csv_path = (
    folder
    / "training_history.csv"
)

output_path = (
    folder
    / "training_curve.png"
)


# --------------------------------------------------
# EĞİTİM VERİSİNİ OKU
# LOAD THE TRAINING DATA
# --------------------------------------------------

print("📊 Eğitim verisi okunuyor...")

df = pd.read_csv(
    csv_path
)

print(
    f"✅ {len(df)} deneme bulundu."
)


# --------------------------------------------------
# TEMEL İSTATİSTİKLER
# BASIC STATISTICS
# --------------------------------------------------

first_score = df[
    "score"
].iloc[0]

last_score = df[
    "score"
].iloc[-1]

best_score = df[
    "best_score"
].max()

reward_count = (
    df["result"]
    .str.contains(
        "ODUL",
        na=False
    )
    .sum()
)


print("\n--- ÖZET ---")

print(
    f"İlk deneme puanı : "
    f"{first_score:.2f}"
)

print(
    f"Son deneme puanı : "
    f"{last_score:.2f}"
)

print(
    f"En iyi puan      : "
    f"{best_score:.2f}"
)

print(
    f"Ödüllü deneme    : "
    f"{reward_count}"
)


# --------------------------------------------------
# ÖĞRENME EĞRİSİNİ ÇİZ
# PLOT THE LEARNING CURVE
# --------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["trial"],
    df["score"],
    marker="o",
    label="Trial score"
)

plt.plot(
    df["trial"],
    df["best_score"],
    linewidth=2,
    label="Best score so far"
)


plt.xlabel(
    "Training Trial"
)

plt.ylabel(
    "Circle Score"
)

plt.title(
    "FlyPaint Learning Curve"
)

plt.ylim(
    0,
    100
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()


# --------------------------------------------------
# GRAFİĞİ KAYDET
# SAVE THE GRAPH
# --------------------------------------------------

plt.savefig(
    output_path,
    dpi=160
)

plt.show()


print(
    f"\n📈 Grafik kaydedildi: "
    f"{output_path}"
)