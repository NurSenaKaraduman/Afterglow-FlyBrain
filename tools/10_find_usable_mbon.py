from flybrain import FlyBrain
import numpy as np


# --------------------------------------------------
# AYARLAR
# SETTINGS
# --------------------------------------------------

SETTLE_STEPS = 200
TEST_STEPS = 100

print("🧠 MaleCNS yükleniyor...")


# --------------------------------------------------
# BEYNİ OLUŞTUR
# CREATE THE BRAIN
# --------------------------------------------------

brain = FlyBrain(
    device="cpu",
    seed=64,
    sensory_input=False
)

print("✅ Beyin hazır.\n")


# --------------------------------------------------
# BEYNİN OTURMASINI BEKLE
# LET THE BRAIN SETTLE
# --------------------------------------------------

for _ in range(SETTLE_STEPS):
    brain.step()


# --------------------------------------------------
# TÜM MBON TİPLERİNİ BUL
# FIND ALL MBON TYPES
# --------------------------------------------------

cell_types = np.asarray(
    brain.cell_type,
    dtype=str
)

mbon_types = sorted(
    {
        cell_type
        for cell_type in cell_types
        if cell_type.startswith("MBON")
    }
)

print(
    f"Bulunan MBON tipi: {len(mbon_types)}\n"
)


# --------------------------------------------------
# HER MBON'UN NORMAL ATEŞLEME ORANINI ÖLÇ
# MEASURE BASELINE FIRING RATE OF EACH MBON
# --------------------------------------------------

groups = {}

for mbon_type in mbon_types:

    indices = brain.cells(
        [mbon_type]
    )

    if len(indices) > 0:
        groups[mbon_type] = set(
            int(x) for x in indices
        )


counts = {
    name: 0
    for name in groups
}


for _ in range(TEST_STEPS):

    fired = brain.step()

    fired_set = set(
        int(x) for x in fired
    )

    for name, neurons in groups.items():

        counts[name] += len(
            fired_set & neurons
        )


# --------------------------------------------------
# SONUÇLARI YAZDIR
# PRINT THE RESULTS
# --------------------------------------------------

results = []

for name, neurons in groups.items():

    maximum = (
        len(neurons)
        * TEST_STEPS
    )

    percent = (
        counts[name]
        / maximum
        * 100
    )

    results.append(
        (
            percent,
            name,
            len(neurons),
            counts[name],
            maximum
        )
    )


results.sort()


print("=" * 65)
print("🧠 MBON NORMAL ATEŞLEME ORANLARI")
print("=" * 65)


for percent, name, neuron_count, spikes, maximum in results:

    marker = ""

    # Orta seviyede çalışanlar bizim için daha kullanışlı.
    # Mid-range activity is more useful for our experiment.

    if 10 <= percent <= 80:
        marker = "  ⭐ UYGUN ADAY"

    print(
        f"{name:<20} "
        f"{neuron_count:2d} nöron | "
        f"{spikes:4d}/{maximum:<4d} | "
        f"%{percent:6.2f}"
        f"{marker}"
    )


print("\n✅ Tarama tamamlandı.")