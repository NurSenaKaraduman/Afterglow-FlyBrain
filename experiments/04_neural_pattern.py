from flybrain import FlyBrain
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# AYARLAR
# SETTINGS
# --------------------------------------------------

SETTLE_STEPS = 200       # 4 saniye
# 4 seconds

BASELINE_STEPS = 50      # 1 saniye
# 1 second

EXPOSURE_STEPS = 50      # 1 saniye
# 1 second

AFTER_SECONDS = 3

STEPS_PER_SECOND = 50

STIMULUS_AMOUNT = 0.8


# --------------------------------------------------
# İKİ AYNI BEYİN OLUŞTUR
# CREATE TWO IDENTICAL BRAINS
# Aynı seed ile iki karşılaştırılabilir beyin oluşturuyoruz.
# Duyusal geri-besleme bu deneyde açık kalır.
# We create two comparable brains using the same seed.
# Sensory feedback remains enabled in this experiment.
# --------------------------------------------------

print("🧠 İki MaleCNS beyni hazırlanıyor...")



# İki aynı beyin oluşturuyoruz.
# We create two identical brains.

control_brain = FlyBrain(
    device="auto",
    seed=64
)

stimulus_brain = FlyBrain(
    device="auto",
    seed=64
)

print("✅ Beyinler hazır.")


# --------------------------------------------------
# STIMULUS NÖRONLARINI BUL
# FIND THE STIMULUS NEURONS
# --------------------------------------------------

lc10a = stimulus_brain.cells(
    ["LC10a"],
    side="L"
)

print(f"🪰 LC10a nöronları: {len(lc10a)}")


# --------------------------------------------------
# BEYİNLERİN OTURMASINI BEKLE
# LET THE BRAINS SETTLE
# --------------------------------------------------

print("\n1/4 Beyinler oturuyor...")

for _ in range(SETTLE_STEPS):

    control_brain.step()
    stimulus_brain.step()


# --------------------------------------------------
# NORMAL DURUM
# BASELINE STATE
# --------------------------------------------------

print("2/4 Baseline...")

for _ in range(BASELINE_STEPS):

    control_brain.step()
    stimulus_brain.step()


# --------------------------------------------------
# SADECE İKİNCİ SİNEĞE STIMULUS VER
# STIMULATE ONLY THE SECOND FLY
# --------------------------------------------------

print("3/4 Stimulus uygulanıyor...")

for _ in range(EXPOSURE_STEPS):

    control_brain.step()

    stimulus_brain.step(
        inject=[
            (lc10a, STIMULUS_AMOUNT)
        ]
    )


# --------------------------------------------------
# AFTER DÖNEMİNDE HER NÖRONU SAY
# COUNT EVERY NEURON DURING THE AFTER PERIOD
# --------------------------------------------------

print("4/4 After dönemi kaydediliyor...\n")

n = control_brain.n

control_counts = np.zeros(
    (AFTER_SECONDS, n),
    dtype=np.int16
)

stimulus_counts = np.zeros(
    (AFTER_SECONDS, n),
    dtype=np.int16
)


for second in range(AFTER_SECONDS):

    for _ in range(STEPS_PER_SECOND):

        control_fired = control_brain.step()
        stimulus_fired = stimulus_brain.step()

        # Hangi nöron kaç kez ateşledi?
        # How many times did each neuron fire?

        np.add.at(
            control_counts[second],
            control_fired,
            1
        )

        np.add.at(
            stimulus_counts[second],
            stimulus_fired,
            1
        )


# --------------------------------------------------
# SONUÇLARI KARŞILAŞTIR
# COMPARE THE RESULTS
# --------------------------------------------------

print("=" * 60)
print("🧠 NEURAL PATTERN COMPARISON")
print("=" * 60)


for second in range(AFTER_SECONDS):

    difference = (
        stimulus_counts[second]
        - control_counts[second]
    )

    changed = np.count_nonzero(difference)

    total_difference = np.abs(
        difference
    ).sum()

    percent = (
        changed / n
    ) * 100

    print(
        f"\n{second + 1}. saniye:"
    )

    print(
        f"Farklı davranan nöron: "
        f"{changed:,} / {n:,}"
    )

    print(
        f"Oran: %{percent:.2f}"
    )

    print(
        f"Toplam spike farkı: "
        f"{total_difference:,}"
    )


# --------------------------------------------------
# ÜÇ SANİYEDEKİ TOPLAM FARKI HESAPLA
# CALCULATE TOTAL DIFFERENCE ACROSS THREE SECONDS
# --------------------------------------------------

control_total = control_counts.sum(axis=0)

stimulus_total = stimulus_counts.sum(axis=0)

difference_total = (
    stimulus_total
    - control_total
)

absolute_difference = np.abs(
    difference_total
)


# --------------------------------------------------
# EN ÇOK DEĞİŞEN NÖRONLARI BUL
# FIND THE MOST CHANGED NEURONS
# --------------------------------------------------

top_indices = np.argsort(
    absolute_difference
)[::-1][:15]


print("\n")
print("=" * 60)
print("🔥 EN ÇOK DEĞİŞEN 15 NÖRON")
print("=" * 60)


rows = []


for idx in top_indices:

    row = {
        "neuron_index": int(idx),

        "cell_type":
            str(control_brain.cell_type[idx]),

        "side":
            str(control_brain.side[idx]),

        "superclass":
            str(control_brain.superclass[idx]),

        "control_spikes":
            int(control_total[idx]),

        "stimulus_spikes":
            int(stimulus_total[idx]),

        "difference":
            int(difference_total[idx])
    }

    rows.append(row)

    print(
        f"\nNöron: {idx}"
    )

    print(
        f"Tür: {row['cell_type']}"
    )

    print(
        f"Taraf: {row['side']}"
    )

    print(
        f"Sınıf: {row['superclass']}"
    )

    print(
        f"Control: {row['control_spikes']}"
    )

    print(
        f"Stimulus sonrası: "
        f"{row['stimulus_spikes']}"
    )

    print(
        f"Fark: {row['difference']:+d}"
    )


# --------------------------------------------------
# SONUÇLARI CSV OLARAK KAYDET
# SAVE THE RESULTS AS CSV
# --------------------------------------------------

df = pd.DataFrame(rows)

# İlk deneyin sonucunu ayrı kaydet.
# Save the first experiment separately.

output = Path(
    "results/top_changed_neurons_sensory_on.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output,
    index=False
)


print("\n")
print("=" * 60)
print("✅ DENEY TAMAMLANDI")
print("=" * 60)

print(
    f"💾 Kaydedildi: {output}"
)