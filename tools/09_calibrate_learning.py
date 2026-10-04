from flybrain import FlyBrain
import numpy as np


# --------------------------------------------------
# AYARLAR
# SETTINGS
# --------------------------------------------------

SEED = 64

SETTLE_STEPS = 200
TEST_STEPS = 50

AMOUNTS = [
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20
]


# --------------------------------------------------
# REFERANS BEYNİ OLUŞTUR
# CREATE A REFERENCE BRAIN
# --------------------------------------------------

reference_brain = FlyBrain(
    device="cpu",
    seed=SEED,
    sensory_input=False
)

cell_types = np.asarray(
    reference_brain.cell_type,
    dtype=str
)


# --------------------------------------------------
# KC NÖRONLARINI BUL
# FIND KC NEURONS
# --------------------------------------------------

kc_indices = np.flatnonzero(
    np.char.startswith(
        cell_types,
        "KC"
    )
)

rng = np.random.default_rng(2026)

circle_kcs = np.sort(
    rng.choice(
        kc_indices,
        size=100,
        replace=False
    )
)


# --------------------------------------------------
# PAM08 VE MBON05 NÖRONLARINI BUL
# FIND PAM08 AND MBON05 NEURONS
# --------------------------------------------------

pam08 = reference_brain.cells(
    ["PAM08"]
)

mbon05 = reference_brain.cells(
    ["MBON05"]
)


print("🧠 Kalibrasyon başlıyor...\n")

print(
    f"KC: {len(circle_kcs)} | "
    f"PAM08: {len(pam08)} | "
    f"MBON05: {len(mbon05)}"
)


# --------------------------------------------------
# FARKLI UYARI GÜÇLERİNİ DENE
# TEST DIFFERENT STIMULUS STRENGTHS
# --------------------------------------------------

for amount in AMOUNTS:

    # Her güç için temiz bir beyin oluştur.
    # Create a fresh brain for every stimulus strength.

    brain = FlyBrain(
        device="cpu",
        seed=SEED,
        sensory_input=False
    )


    # Beynin normal durumuna oturmasını bekle.
    # Allow the brain to settle into its resting state.

    for _ in range(SETTLE_STEPS):
        brain.step()


    kc_spikes = 0
    pam_spikes = 0
    mbon_spikes = 0


    # Aynı anda KC ve PAM gruplarını test et.
    # Test KC and PAM groups at the same time.

    for _ in range(TEST_STEPS):

        fired = brain.step(
            inject=[
                (
                    circle_kcs,
                    amount
                ),
                (
                    pam08,
                    amount
                )
            ]
        )

        fired_set = set(
            int(x) for x in fired
        )


        kc_spikes += len(
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


        mbon_spikes += len(
            fired_set
            & set(
                int(x)
                for x in mbon05
            )
        )


    # Maksimum mümkün spike sayılarını hesapla.
    # Calculate the maximum possible spike counts.

    kc_max = (
        len(circle_kcs)
        * TEST_STEPS
    )

    pam_max = (
        len(pam08)
        * TEST_STEPS
    )

    mbon_max = (
        len(mbon05)
        * TEST_STEPS
    )


    kc_percent = (
        kc_spikes / kc_max * 100
    )

    pam_percent = (
        pam_spikes / pam_max * 100
    )

    mbon_percent = (
        mbon_spikes / mbon_max * 100
    )


    print(
        f"\nUyarı = {amount}"
    )

    print(
        f"KC     : {kc_spikes:4d}/{kc_max} "
        f"(%{kc_percent:.1f})"
    )

    print(
        f"PAM08  : {pam_spikes:4d}/{pam_max} "
        f"(%{pam_percent:.1f})"
    )

    print(
        f"MBON05 : {mbon_spikes:3d}/{mbon_max} "
        f"(%{mbon_percent:.1f})"
    )


print("\n✅ Kalibrasyon tamamlandı.")