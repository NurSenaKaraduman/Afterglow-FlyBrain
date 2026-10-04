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

CUE_STEPS = 50

TRAINING_TRIALS = 10

# İşareti daha hafif uyarıyoruz.
# We stimulate the cue more gently.

CUE_AMOUNT = 0.20

# Dopamin nöronlarını da tavana vurdurmuyoruz.
# We also avoid saturating the dopamine neurons.

PAM_AMOUNT = 0.25

# Her ödüllü denemeden sonra ilgili bağlantıları %5 zayıflat.
# Depress the relevant synapses by 5% after each rewarded trial.
LEARNING_RATE = 0.05


# --------------------------------------------------
# BEYNİ YÜKLE
# LOAD THE BRAIN
# --------------------------------------------------

print("🧠 MaleCNS yükleniyor...")

# CPU kullanıyoruz çünkü bağlantı ağırlıklarını değiştireceğiz.
# We use CPU because we are going to modify synaptic weights.
brain = FlyBrain(
    device="cpu",
    seed=SEED,
    sensory_input=False
)

print("✅ Beyin hazır.")


# --------------------------------------------------
# ÖĞRENME DEVRESİNİ BUL
# FIND THE LEARNING CIRCUIT
# --------------------------------------------------

cell_types = np.asarray(
    brain.cell_type,
    dtype=str
)

# Bütün Kenyon hücrelerini seç.
# Select all Kenyon cells.
kc_mask = np.char.startswith(
    cell_types,
    "KC"
)

kc_indices = np.flatnonzero(
    kc_mask
)

# MBON05'i seç.
# Select MBON05.
mbon05 = brain.cells(
    ["MBON05"]
)

# Ödül benzeri PAM08 nöronlarını seç.
# Select reward-like PAM08 neurons.
pam08 = brain.cells(
    ["PAM08"]
)

print("\n--- DEVRE ---")
print(f"KC       : {len(kc_indices)}")
print(f"MBON05   : {len(mbon05)}")
print(f"PAM08    : {len(pam08)}")


# --------------------------------------------------
# "DAİRE" İÇİN SEYREK BİR KC DESENİ OLUŞTUR
# CREATE A SPARSE KC PATTERN FOR THE "CIRCLE" CUE
# --------------------------------------------------

rng = np.random.default_rng(
    2026
)

# Şimdilik 100 KC bu işareti temsil edecek.
# For now, 100 KCs represent this cue.
circle_kcs = rng.choice(
    kc_indices,
    size=100,
    replace=False
)

circle_kcs = np.sort(
    circle_kcs
)

print(
    f"⭕ Daire işaretini temsil eden KC sayısı: "
    f"{len(circle_kcs)}"
)


# --------------------------------------------------
# KC -> MBON05 BAĞLANTILARINI BUL
# FIND KC -> MBON05 CONNECTIONS
# --------------------------------------------------

mbon_mask = np.zeros(
    brain.n,
    dtype=bool
)

mbon_mask[mbon05] = True


def find_kc_mbon_edges(pre_neurons):

    edge_positions = []
    edge_presynaptic = []

    for pre in pre_neurons:

        start = brain.indptr[pre]
        end = brain.indptr[pre + 1]

        posts = brain.indices[
            start:end
        ]

        local_mask = mbon_mask[posts]

        if not np.any(local_mask):
            continue

        positions = (
            np.arange(start, end)[local_mask]
        )

        edge_positions.extend(
            positions.tolist()
        )

        edge_presynaptic.extend(
            [int(pre)] * len(positions)
        )

    return (
        np.asarray(edge_positions, dtype=np.int64),
        np.asarray(edge_presynaptic, dtype=np.int64)
    )


circle_edges, circle_edge_pre = (
    find_kc_mbon_edges(circle_kcs)
)

print(
    f"🔗 Daire KC'lerinden MBON05'e "
    f"{len(circle_edges)} bağlantı bulundu."
)

if len(circle_edges) == 0:

    raise RuntimeError(
        "Seçilen KC grubundan MBON05'e bağlantı bulunamadı."
    )


# --------------------------------------------------
# BAŞLANGIÇ AĞIRLIKLARINI SAKLA
# SAVE THE INITIAL WEIGHTS
# --------------------------------------------------

initial_weights = brain.weights[
    circle_edges
].copy()

print(
    f"Başlangıç ortalama ağırlık: "
    f"{initial_weights.mean():.6f}"
)


# --------------------------------------------------
# BİR CUE'YA MBON05 TEPKİSİNİ ÖLÇ
# MEASURE MBON05 RESPONSE TO A CUE
# --------------------------------------------------

def measure_response(test_seed):

    # Voltajları sıfırla ama öğrenilmiş bağlantıları koru.
    # Reset voltages while preserving learned weights.
    brain.reset(
        seed=test_seed
    )

    for _ in range(SETTLE_STEPS):
        brain.step()

    mbon_spikes = 0

    kc_spikes = 0

    for _ in range(CUE_STEPS):

        fired = brain.step(
            inject=[
                (
                    circle_kcs,
                    CUE_AMOUNT
                )
            ]
        )

        fired_set = set(
            int(x) for x in fired
        )

        mbon_spikes += len(
            fired_set
            & set(int(x) for x in mbon05)
        )

        kc_spikes += len(
            fired_set
            & set(int(x) for x in circle_kcs)
        )

    return (
        mbon_spikes,
        kc_spikes
    )


# --------------------------------------------------
# ÖĞRENME ÖNCESİ ÖLÇÜM
# MEASURE BEFORE LEARNING
# --------------------------------------------------

print("\n🧪 Öğrenme öncesi test...")

before_mbon, before_kc = (
    measure_response(
        test_seed=999
    )
)

print(
    f"KC spike:     {before_kc}"
)

print(
    f"MBON05 spike: {before_mbon}"
)


# --------------------------------------------------
# EĞİTİM
# TRAINING
# --------------------------------------------------

print("\n🎓 Eğitim başlıyor...")


training_rows = []


for trial in range(
    1,
    TRAINING_TRIALS + 1
):

    # Her denemede aynı başlangıç koşullarını kullan.
    # Use the same initial conditions on every trial.
    brain.reset(
        seed=1000 + trial
    )

    for _ in range(SETTLE_STEPS):
        brain.step()

    active_kcs = set()

    pam_spikes = 0


    for _ in range(CUE_STEPS):

        # Daire işareti + PAM08 ödül sinyali.
        # Circle cue + PAM08 reward signal.
        fired = brain.step(
            inject=[
                (
                    circle_kcs,
                    CUE_AMOUNT
                ),
                (
                    pam08,
                    PAM_AMOUNT
                )
            ]
        )

        fired_set = set(
            int(x) for x in fired
        )

        # Bu denemede gerçekten ateşleyen KC'leri kaydet.
        # Record KCs that actually fired during this trial.
        active_kcs.update(
            fired_set
            & set(
                int(x)
                for x in circle_kcs
            )
        )

        pam_spikes += len(
            fired_set
            & set(
                int(x)
                for x in pam08
            )
        )


    # Dopamin nöronları gerçekten ateşlediyse öğrenme uygula.
    # Apply plasticity only if the dopamine neurons actually fired.
    changed_edges = 0

    if pam_spikes > 0:

        active_kcs_array = np.asarray(
            sorted(active_kcs),
            dtype=np.int64
        )

        active_edges, _ = (
            find_kc_mbon_edges(
                active_kcs_array
            )
        )

        # Basitleştirilmiş dopamin-gated plastisite:
        # aktif KC -> MBON05 bağlantılarını zayıflat.
        #
        # Simplified dopamine-gated plasticity:
        # depress active KC -> MBON05 synapses.
        brain.weights[
            active_edges
        ] *= (
            1.0 - LEARNING_RATE
        )

        changed_edges = len(
            active_edges
        )


    mean_weight = brain.weights[
        circle_edges
    ].mean()


    print(
        f"Deneme {trial:02d} | "
        f"PAM spike={pam_spikes:4d} | "
        f"Aktif KC={len(active_kcs):3d} | "
        f"Değişen bağlantı={changed_edges:4d} | "
        f"Ort. ağırlık={mean_weight:.6f}"
    )


    training_rows.append({
        "trial": trial,
        "pam_spikes": pam_spikes,
        "active_kcs": len(active_kcs),
        "changed_edges": changed_edges,
        "mean_weight": mean_weight
    })


# --------------------------------------------------
# ÖĞRENME SONRASI ÖLÇÜM
# MEASURE AFTER LEARNING
# --------------------------------------------------

print("\n🧪 Öğrenme sonrası test...")

after_mbon, after_kc = (
    measure_response(
        test_seed=999
    )
)

final_weights = brain.weights[
    circle_edges
]


# --------------------------------------------------
# SONUÇ
# RESULT
# --------------------------------------------------

print("\n")
print("=" * 60)
print("🧠 DOPAMİN PLASTİSİTESİ SONUCU")
print("=" * 60)

print(
    f"MBON05 öğrenme öncesi: "
    f"{before_mbon}"
)

print(
    f"MBON05 öğrenme sonrası: "
    f"{after_mbon}"
)

print(
    f"\nBağlantı ağırlığı önce: "
    f"{initial_weights.mean():.6f}"
)

print(
    f"Bağlantı ağırlığı sonra: "
    f"{final_weights.mean():.6f}"
)

weight_change = (
    (
        final_weights.mean()
        / initial_weights.mean()
    )
    - 1
) * 100

print(
    f"Ağırlık değişimi: "
    f"%{weight_change:.2f}"
)


# --------------------------------------------------
# RAPORU KAYDET
# SAVE THE REPORT
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
    / "dopamine_training.csv"
)

pd.DataFrame(
    training_rows
).to_csv(
    output,
    index=False
)

print(
    f"\n💾 Kaydedildi: {output}"
)