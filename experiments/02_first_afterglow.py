from flybrain import FlyBrain
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# AYARLAR
# SETTINGS
# --------------------------------------------------

DT = 0.020          # Her adım 20 ms
                    # Each step is 20 ms
BASELINE_STEPS = 50 # 1 saniye
                    # 1 second
STIMULUS_STEPS = 50 # 1 saniye
                    # 1 second
AFTER_STEPS = 150   # 3 saniye
                    # 3 seconds


# --------------------------------------------------
# BEYNİ YÜKLE
# LOAD THE BRAIN
# --------------------------------------------------

print("🧠 MaleCNS yükleniyor...")

brain = FlyBrain(device="auto")

print("✅ Beyin hazır.")


# --------------------------------------------------
# UYARACAĞIMIZ NÖRONLAR
# NEURONS TO STIMULATE
# --------------------------------------------------

# LC10a gerçek erkek sinekte hareket eden küçük hedeflere
# LC10a is one of the visual projection neurons in the real male fly
# cevap veren görsel projeksiyon nöronlarından biridir.
# that responds to small moving targets.
stimulus_neurons = brain.cells(
    ["LC10a"],
    side="L"
)

print(f"🪰 LC10a nöron sayısı: {len(stimulus_neurons)}")


# --------------------------------------------------
# SONUÇLARI BURADA TOPLAYACAĞIZ
# WE WILL COLLECT THE RESULTS HERE
# --------------------------------------------------

records = []


def save_step(step_number, phase, fired):
    records.append({
        "step": step_number,
        "time_seconds": step_number * DT,
        "phase": phase,
        "spike_count": len(fired)
    })


step = 0


# --------------------------------------------------
# 1) BASELINE
# 1) BASELINE
# Sineğe hiçbir şey göstermiyoruz.
# We are not showing anything to the fly.
# --------------------------------------------------

print("\n1/3 Baseline...")

for _ in range(BASELINE_STEPS):

    fired = brain.step()

    save_step(
        step,
        "baseline",
        fired
    )

    step += 1


# --------------------------------------------------
# 2) STIMULUS
# 2) STIMULUS
# LC10a nöronlarını uyarıyoruz.
# We are stimulating the LC10a neurons.
# --------------------------------------------------

print("2/3 Stimulus...")

for _ in range(STIMULUS_STEPS):

    fired = brain.step(
        inject=[
            (stimulus_neurons, 0.8)
        ]
    )

    save_step(
        step,
        "stimulus",
        fired
    )

    step += 1


# --------------------------------------------------
# 3) AFTERGLOW
# 3) AFTERGLOW
# Uyaran tamamen kaldırıldı.
# The stimulus is completely removed.
# --------------------------------------------------

print("3/3 Afterglow...")

for _ in range(AFTER_STEPS):

    fired = brain.step()

    save_step(
        step,
        "after",
        fired
    )

    step += 1


# --------------------------------------------------
# CSV KAYDET
# SAVE AS CSV
# --------------------------------------------------

results = pd.DataFrame(records)

output = Path("results/first_afterglow.csv")
output.parent.mkdir(exist_ok=True)

results.to_csv(
    output,
    index=False
)


# --------------------------------------------------
# ÖZET
# SUMMARY
# --------------------------------------------------

print("\n-----------------------------")
print("        SONUÇ ÖZETİ")
print("-----------------------------")

print(
    results.groupby("phase")["spike_count"]
    .agg(["mean", "max", "std"])
)

print(f"\n💾 Kaydedildi: {output}")

print("\n🪰 İlk Afterglow deneyi tamamlandı.")