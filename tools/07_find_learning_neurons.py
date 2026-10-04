from flybrain import FlyBrain
import numpy as np


# --------------------------------------------------
# BEYNİ YÜKLE
# LOAD THE BRAIN
# --------------------------------------------------

print("🧠 MaleCNS yükleniyor...")

brain = FlyBrain(
    device="auto",
    seed=64,
    sensory_input=False
)

print("✅ Beyin hazır.\n")


# --------------------------------------------------
# BENZERSİZ NÖRON TİPLERİNİ AL
# GET UNIQUE NEURON TYPES
# --------------------------------------------------

cell_types = np.array(
    brain.cell_type,
    dtype=str
)

unique_types = sorted(
    set(cell_types)
)


# --------------------------------------------------
# ÖĞRENMEYLE İLGİLİ İSİMLERİ ARA
# SEARCH FOR LEARNING-RELATED NAMES
# --------------------------------------------------

keywords = [
    "KC",
    "MBON",
    "PAM",
    "PPL1",
    "DAN"
]


for keyword in keywords:

    print("=" * 60)
    print(f"🔎 {keyword}")
    print("=" * 60)

    matches = [
        cell_type
        for cell_type in unique_types
        if keyword.lower() in cell_type.lower()
    ]

    print(
        f"Bulunan tip sayısı: {len(matches)}"
    )

    for cell_type in matches[:80]:

        count = np.count_nonzero(
            cell_types == cell_type
        )

        print(
            f"{cell_type:<35} {count} nöron"
        )

    if len(matches) > 80:

        print(
            f"... ve {len(matches) - 80} tip daha"
        )

    print()