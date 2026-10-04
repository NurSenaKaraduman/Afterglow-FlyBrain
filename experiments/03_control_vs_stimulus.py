from flybrain import FlyBrain
import pandas as pd
import random
from pathlib import Path
import time


# ----------------------------------------
# AYARLAR
# SETTINGS
# ----------------------------------------

SETTLE_STEPS = 200      # 4 saniye: beynin oturmasını bekle
                        # 4 seconds: wait for the brain to settle
BASELINE_STEPS = 50     # 1 saniye: normal halini ölç
                        # 1 second: measure the baseline state
EXPOSURE_STEPS = 50     # 1 saniye: stimulus veya hiçbir şey
                        # 1 second: stimulus or nothing
AFTER_STEPS = 150       # 3 saniye: olay bittikten sonrası
                        # 3 seconds: after the event ends

TRIALS_PER_CONDITION = 5

STIMULUS_AMOUNT = 0.8


# ----------------------------------------
# TEK DENEY
# SINGLE TRIAL
# ----------------------------------------

def run_trial(condition, trial_number):

    print(f"\n🪰 Deney {trial_number} | {condition.upper()}")

    # Her deneyde yepyeni bir sinek beyni
    # A brand new fly brain for each trial
    brain = FlyBrain(device="auto")

    lc10a = brain.cells(
        ["LC10a"],
        side="L"
    )

    # ----------------------------------------
    # 1) BEYNİN OTURMASINI BEKLE
    # 1) WAIT FOR THE BRAIN TO SETTLE
    # ----------------------------------------

    print("   Beyin hazırlanıyor...")

    for _ in range(SETTLE_STEPS):
        brain.step()


    # ----------------------------------------
    # 2) BASELINE
    # 2) BASELINE
    # ----------------------------------------

    baseline_spikes = []

    for _ in range(BASELINE_STEPS):

        fired = brain.step()

        baseline_spikes.append(
            len(fired)
        )


    # ----------------------------------------
    # 3) CONTROL veya STIMULUS
    # 3) CONTROL OR STIMULUS
    # ----------------------------------------

    exposure_spikes = []

    for _ in range(EXPOSURE_STEPS):

        if condition == "stimulus":

            fired = brain.step(
                inject=[
                    (lc10a, STIMULUS_AMOUNT)
                ]
            )

        else:

            # CONTROL:
            # CONTROL:
            # hiçbir uyaran vermiyoruz
            # we give no stimulus
            fired = brain.step()

        exposure_spikes.append(
            len(fired)
        )


    # ----------------------------------------
    # 4) AFTER
    # 4) AFTER
    # Artık iki durumda da hiçbir uyaran yok
    # Now there is no stimulus in either condition
    # ----------------------------------------

    after_spikes = []

    for _ in range(AFTER_STEPS):

        fired = brain.step()

        after_spikes.append(
            len(fired)
        )


    # İlk 1 saniye ve son 1 saniyeyi ayrıca ölç
    # Measure the first 1 second and the last 1 second separately
    after_early = after_spikes[:50]
    after_late = after_spikes[-50:]


    result = {

        "trial": trial_number,
        "condition": condition,

        "baseline_mean":
            sum(baseline_spikes) / len(baseline_spikes),

        "exposure_mean":
            sum(exposure_spikes) / len(exposure_spikes),

        "after_mean":
            sum(after_spikes) / len(after_spikes),

        "after_early_mean":
            sum(after_early) / len(after_early),

        "after_late_mean":
            sum(after_late) / len(after_late)
    }

    print(
        f"   Baseline: {result['baseline_mean']:.1f}"
    )

    print(
        f"   Exposure: {result['exposure_mean']:.1f}"
    )

    print(
        f"   After:    {result['after_mean']:.1f}"
    )

    return result


# ----------------------------------------
# DENEYLER
# TRIALS
# ----------------------------------------

conditions = (
    ["control"] * TRIALS_PER_CONDITION
    +
    ["stimulus"] * TRIALS_PER_CONDITION
)

# Deney sırasını karıştırıyoruz.
# We shuffle the trial order.
# Hep control sonra hep stimulus olmasın.
# So it's not all control followed by all stimulus.
random.Random(42).shuffle(conditions)


results = []

start_time = time.time()


for i, condition in enumerate(conditions, start=1):

    result = run_trial(
        condition,
        i
    )

    results.append(result)


# ----------------------------------------
# DATAFRAME
# DATAFRAME
# ----------------------------------------

df = pd.DataFrame(results)


# ----------------------------------------
# KAYDET
# SAVE
# ----------------------------------------

output = Path(
    "results/control_vs_stimulus.csv"
)

output.parent.mkdir(exist_ok=True)

df.to_csv(
    output,
    index=False
)


# ----------------------------------------
# SONUÇ
# RESULT
# ----------------------------------------

print("\n")
print("=" * 55)
print("🧠 CONTROL vs STIMULUS SONUCU")
print("=" * 55)


summary = (
    df.groupby("condition")
    [
        [
            "baseline_mean",
            "exposure_mean",
            "after_mean",
            "after_early_mean",
            "after_late_mean"
        ]
    ]
    .mean()
)

print(summary)


control_after = (
    df[df["condition"] == "control"]
    ["after_mean"]
    .mean()
)

stimulus_after = (
    df[df["condition"] == "stimulus"]
    ["after_mean"]
    .mean()
)


difference = stimulus_after - control_after


print("\n-----------------------------")

print(
    f"Control AFTER:  {control_after:.2f}"
)

print(
    f"Stimulus AFTER: {stimulus_after:.2f}"
)

print(
    f"Fark:           {difference:.2f}"
)


elapsed = time.time() - start_time

print(
    f"\n⏱️ Süre: {elapsed / 60:.1f} dakika"
)

print(
    f"💾 Kaydedildi: {output}"
)