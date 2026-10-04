from flybrain import FlyBrain
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# AYARLAR
# SETTINGS
# --------------------------------------------------

SEED = 64

SETTLE_STEPS = 200
TEST_STEPS = 100

TRAINING_TRIALS = 12

LEARNING_RATE = 0.05

# Dairenin "hafıza izi" olarak kullanacağımız KC sayısı.
# Number of KCs used as the memory trace for the circle.
CIRCLE_KC_COUNT = 100


# --------------------------------------------------
# İKİ AYNI BEYİN OLUŞTUR
# CREATE TWO IDENTICAL BRAINS
# --------------------------------------------------

print("🧠 İki MaleCNS beyni hazırlanıyor...")

control_brain = FlyBrain(
    device="cpu",
    seed=SEED,
    sensory_input=False
)

learning_brain = FlyBrain(
    device="cpu",
    seed=SEED,
    sensory_input=False
)

print("✅ Beyinler hazır.")


# --------------------------------------------------
# MBON26 VE PAM05 NÖRONLARINI BUL
# FIND MBON26 AND PAM05 NEURONS
# --------------------------------------------------

mbon26 = learning_brain.cells(
    ["MBON26"]
)

pam05 = learning_brain.cells(
    ["PAM05"]
)

print("\n--- DEVRE ---")

print(
    f"MBON26 : {len(mbon26)} nöron"
)

print(
    f"PAM05  : {len(pam05)} nöron"
)


# --------------------------------------------------
# TÜM KC NÖRONLARINI BUL
# FIND ALL KC NEURONS
# --------------------------------------------------

cell_types = np.asarray(
    learning_brain.cell_type,
    dtype=str
)

kc_indices = np.flatnonzero(
    np.char.startswith(
        cell_types,
        "KC"
    )
)

print(
    f"KC     : {len(kc_indices)} nöron"
)


# --------------------------------------------------
# KC -> MBON26 BAĞLANTILARINI BUL
# FIND KC -> MBON26 CONNECTIONS
# --------------------------------------------------

mbon_mask = np.zeros(
    learning_brain.n,
    dtype=bool
)

mbon_mask[mbon26] = True


kc_connection_strength = []


for kc in kc_indices:

    start = learning_brain.indptr[kc]
    end = learning_brain.indptr[kc + 1]

    posts = learning_brain.indices[
        start:end
    ]

    weights = learning_brain.weights[
        start:end
    ]

    mask = mbon_mask[posts]

    if np.any(mask):

        total_strength = np.abs(
            weights[mask]
        ).sum()

        kc_connection_strength.append(
            (
                float(total_strength),
                int(kc)
            )
        )


# En güçlü bağlantısı olan KC'leri öne al.
# Put KCs with the strongest connections first.

kc_connection_strength.sort(
    reverse=True
)


circle_kcs = np.array(
    [
        kc
        for _, kc
        in kc_connection_strength[
            :CIRCLE_KC_COUNT
        ]
    ],
    dtype=np.int64
)


print(
    f"⭕ Daire hafızası için seçilen KC: "
    f"{len(circle_kcs)}"
)


# --------------------------------------------------
# SEÇİLEN KC -> MBON26 BAĞLANTILARINI BUL
# FIND SELECTED KC -> MBON26 SYNAPSES
# --------------------------------------------------

def find_edges(brain, pre_neurons):

    positions = []

    for pre in pre_neurons:

        start = brain.indptr[pre]
        end = brain.indptr[pre + 1]

        posts = brain.indices[
            start:end
        ]

        mask = mbon_mask[posts]

        if np.any(mask):

            edge_positions = (
                np.arange(
                    start,
                    end
                )[mask]
            )

            positions.extend(
                edge_positions.tolist()
            )

    return np.asarray(
        positions,
        dtype=np.int64
    )


learning_edges = find_edges(
    learning_brain,
    circle_kcs
)

control_edges = find_edges(
    control_brain,
    circle_kcs
)


print(
    f"🔗 Kullanılacak KC → MBON26 bağlantısı: "
    f"{len(learning_edges)}"
)


# --------------------------------------------------
# BAŞLANGIÇ AĞIRLIKLARINI KAYDET
# SAVE INITIAL WEIGHTS
# --------------------------------------------------

initial_weights = (
    learning_brain.weights[
        learning_edges
    ].copy()
)


print(
    f"Başlangıç ortalama ağırlık: "
    f"{initial_weights.mean():.6f}"
)


# --------------------------------------------------
# MBON26 CEVABINI ÖLÇ
# MEASURE MBON26 RESPONSE
# --------------------------------------------------

def measure_mbon(brain, seed):

    brain.reset(
        seed=seed
    )

    for _ in range(
        SETTLE_STEPS
    ):
        brain.step()


    spikes = 0

    mbon_set = set(
        int(x)
        for x in mbon26
    )


    for _ in range(
        TEST_STEPS
    ):

        fired = brain.step()

        fired_set = set(
            int(x)
            for x in fired
        )

        spikes += len(
            fired_set
            & mbon_set
        )


    maximum = (
        len(mbon26)
        * TEST_STEPS
    )

    percent = (
        spikes
        / maximum
        * 100
    )

    return spikes, percent


# --------------------------------------------------
# ÖĞRENME ÖNCESİ
# BEFORE LEARNING
# --------------------------------------------------

print(
    "\n🧪 Öğrenme öncesi ölçüm..."
)


control_before = measure_mbon(
    control_brain,
    seed=999
)

learning_before = measure_mbon(
    learning_brain,
    seed=999
)


print(
    f"Control MBON26 : "
    f"%{control_before[1]:.2f}"
)

print(
    f"Learning MBON26: "
    f"%{learning_before[1]:.2f}"
)


# --------------------------------------------------
# EĞİTİM
# TRAINING
# --------------------------------------------------

print(
    "\n🎓 Ödül-plastisite eğitimi başlıyor..."
)


rows = []


for trial in range(
    1,
    TRAINING_TRIALS + 1
):

    # Öğrenen beyinde seçilen KC -> MBON26
    # bağlantılarını biraz değiştir.
    #
    # Slightly modify selected KC -> MBON26
    # synapses in the learning brain.

    learning_brain.weights[
        learning_edges
    ] *= (
        1.0
        - LEARNING_RATE
    )


    # Control beyinde hiçbir ağırlığı değiştirmiyoruz.
    # No weights are modified in the control brain.


    control_result = measure_mbon(
        control_brain,
        seed=1000 + trial
    )

    learning_result = measure_mbon(
        learning_brain,
        seed=1000 + trial
    )


    mean_weight = (
        learning_brain.weights[
            learning_edges
        ].mean()
    )


    print(
        f"Deneme {trial:02d} | "
        f"Control=%{control_result[1]:6.2f} | "
        f"Learning=%{learning_result[1]:6.2f} | "
        f"Ağırlık={mean_weight:.6f}"
    )


    rows.append({
        "trial": trial,
        "control_percent":
            control_result[1],
        "learning_percent":
            learning_result[1],
        "mean_weight":
            mean_weight
    })


# --------------------------------------------------
# SON TEST
# FINAL TEST
# --------------------------------------------------

print(
    "\n🧪 Son karşılaştırma..."
)


control_after = measure_mbon(
    control_brain,
    seed=999
)

learning_after = measure_mbon(
    learning_brain,
    seed=999
)


final_weights = (
    learning_brain.weights[
        learning_edges
    ]
)


print("\n")
print("=" * 60)
print("🧠 MBON26 PLASTİSİTE SONUCU")
print("=" * 60)


print(
    f"Control önce : "
    f"%{control_before[1]:.2f}"
)

print(
    f"Control sonra: "
    f"%{control_after[1]:.2f}"
)


print(
    f"\nLearning önce : "
    f"%{learning_before[1]:.2f}"
)

print(
    f"Learning sonra: "
    f"%{learning_after[1]:.2f}"
)


print(
    f"\nBağlantı önce : "
    f"{initial_weights.mean():.6f}"
)

print(
    f"Bağlantı sonra: "
    f"{final_weights.mean():.6f}"
)


# --------------------------------------------------
# SONUCU KAYDET
# SAVE THE RESULT
# --------------------------------------------------

folder = Path(
    "results/plasticity"
)

folder.mkdir(
    parents=True,
    exist_ok=True
)

output = (
    folder
    / "mbon26_learning.csv"
)

pd.DataFrame(
    rows
).to_csv(
    output,
    index=False
)

print(
    f"\n💾 Kaydedildi: {output}"
)