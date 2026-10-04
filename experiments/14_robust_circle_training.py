from flybrain import FlyBrain
import pygame
import numpy as np
import pandas as pd
from pathlib import Path
import math


# --------------------------------------------------
# GENEL AYARLAR
# GENERAL SETTINGS
# --------------------------------------------------

WIDTH = 900
HEIGHT = 650

CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2

TARGET_RADIUS = 160

DRAW_STEPS = 360

SETTLE_STEPS = 50

TRAINING_TRIALS = 24


# --------------------------------------------------
# EĞİTİM VE TEST SEED'LERİ
# TRAINING AND TEST SEEDS
# --------------------------------------------------

# Her aday bu üç başlangıç durumunda sınanacak.
# Every candidate will be tested on these same three initial states.

TRAIN_SEEDS = [
    64,
    65,
    66
]

# Bunlar eğitimde ASLA kullanılmayacak.
# These seeds will NEVER be used during training.

HOLDOUT_SEEDS = [
    900,
    901,
    902
]


# --------------------------------------------------
# ÖĞRENME AYARLARI
# LEARNING SETTINGS
# --------------------------------------------------

EXPLORATION = np.array(
    [
        0.18,   # sol dönüş / left steering
        0.18,   # sağ dönüş / right steering
        0.18,   # ileri / forward
        0.18,   # geri / backward
        0.010   # dönüş biası / steering bias
    ],
    dtype=float
)

EXPLORATION_DECAY = 0.94


# --------------------------------------------------
# BEYNİ YÜKLE
# LOAD THE BRAIN
# --------------------------------------------------

print("🧠 MaleCNS hazırlanıyor...")

brain = FlyBrain(
    device="auto",
    seed=64,
    sensory_input=False
)

print("✅ Beyin hazır.")


# --------------------------------------------------
# MOTOR NÖRONLARINI BUL
# FIND MOTOR NEURONS
# --------------------------------------------------

steer_left = set(
    int(x)
    for x in brain.cells(
        ["DNa02"],
        side="L"
    )
)

steer_right = set(
    int(x)
    for x in brain.cells(
        ["DNa02"],
        side="R"
    )
)

forward = set(
    int(x)
    for x in brain.cells(
        ["DNg100"]
    )
)

backward = set(
    int(x)
    for x in brain.cells(
        ["MDN"]
    )
)


print("\n--- MOTOR DEVRESİ ---")
print("DNa02 Left :", len(steer_left))
print("DNa02 Right:", len(steer_right))
print("DNg100     :", len(forward))
print("MDN        :", len(backward))


# --------------------------------------------------
# PYGAME
# PYGAME
# --------------------------------------------------

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "Robust FlyPaint Training — MaleCNS"
)

font = pygame.font.Font(
    None,
    27
)

small_font = pygame.font.Font(
    None,
    22
)


# --------------------------------------------------
# DAİRE PUANI
# CIRCLE SCORE
# --------------------------------------------------

def calculate_circle_score(points):

    if len(points) < 10:
        return 0.0

    pts = np.asarray(
        points,
        dtype=float
    )

    dx = (
        pts[:, 0]
        - CENTER_X
    )

    dy = (
        pts[:, 1]
        - CENTER_Y
    )

    distances = np.sqrt(
        dx**2 + dy**2
    )

    # Hedef yarıçapa ne kadar yakın?
    # How close are the points to the target radius?

    radial_error = np.abs(
        distances
        - TARGET_RADIUS
    )

    normalized_error = (
        radial_error
        / TARGET_RADIUS
    )

    mean_error = float(
        np.mean(
            normalized_error
        )
    )


    # Çemberin ne kadarını dolaştı?
    # How much of the circle did it cover?

    angles = (
        np.arctan2(
            dy,
            dx
        )
        + 2 * np.pi
    ) % (
        2 * np.pi
    )

    bins = np.floor(
        angles
        / (2 * np.pi)
        * 24
    ).astype(int)

    good_points = (
        normalized_error < 0.25
    )

    covered_bins = len(
        np.unique(
            bins[good_points]
        )
    )

    coverage = (
        covered_bins
        / 24
    )


    # Şeklin hedef çembere yakınlığını ölç.
    # Measure radial shape quality.

    shape_quality = max(
        0.0,
        1.0
        - mean_error * 2.5
    )


    score = (
        100
        * shape_quality
        * coverage
    )

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


# --------------------------------------------------
# TEK ÇİZİM DENEYİ
# RUN ONE DRAWING TRIAL
# --------------------------------------------------

def run_trial(
    parameters,
    seed,
    show=False,
    show_target=True,
    label_text=""
):

    left_gain = parameters[0]
    right_gain = parameters[1]

    forward_gain = parameters[2]
    backward_gain = parameters[3]

    steering_bias = parameters[4]


    # Beyni belirli seed ile aynı başlangıç durumuna getir.
    # Reset the brain to a reproducible initial state.

    brain.reset(
        seed=seed
    )

    for _ in range(
        SETTLE_STEPS
    ):
        brain.step()


    canvas = pygame.Surface(
        (WIDTH, HEIGHT)
    )

    canvas.fill(
        (250, 248, 243)
    )


    # Kalem hedef çemberin sağından başlıyor.
    # The pen starts at the right edge of the target circle.

    x = float(
        CENTER_X
        + TARGET_RADIUS
    )

    y = float(
        CENTER_Y
    )


    # Başlangıç yönü yukarı.
    # Initial heading points upward.

    heading = (
        -math.pi / 2
    )


    points = [
        (x, y)
    ]


    for step in range(
        DRAW_STEPS
    ):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit


        fired = brain.step()

        fired_set = set(
            int(n)
            for n in fired
        )


        left_count = len(
            fired_set
            & steer_left
        )

        right_count = len(
            fired_set
            & steer_right
        )

        forward_count = len(
            fired_set
            & forward
        )

        backward_count = len(
            fired_set
            & backward
        )


        # Motor çıktısından dönüş üret.
        # Convert motor activity into steering.

        steering = (
            right_count
            * right_gain

            -

            left_count
            * left_gain

            +

            steering_bias
        )


        steering = float(
            np.clip(
                steering,
                -0.10,
                0.10
            )
        )

        heading += steering


        # Motor çıktısından hız üret.
        # Convert motor activity into movement speed.

        speed = (
            2.0

            +

            forward_count
            * forward_gain

            -

            backward_count
            * backward_gain
        )


        speed = float(
            np.clip(
                speed,
                0.3,
                4.5
            )
        )


        old_x = x
        old_y = y


        x += (
            math.cos(
                heading
            )
            * speed
        )

        y += (
            math.sin(
                heading
            )
            * speed
        )


        x = float(
            np.clip(
                x,
                5,
                WIDTH - 5
            )
        )

        y = float(
            np.clip(
                y,
                5,
                HEIGHT - 5
            )
        )


        pygame.draw.line(
            canvas,
            (30, 30, 30),
            (
                int(old_x),
                int(old_y)
            ),
            (
                int(x),
                int(y)
            ),
            2
        )


        points.append(
            (
                x,
                y
            )
        )


        if show:

            screen.fill(
                (250, 248, 243)
            )

            screen.blit(
                canvas,
                (0, 0)
            )


            # Eğitim sırasında hedefi göster.
            # Show the target during training previews.

            if show_target:

                pygame.draw.circle(
                    screen,
                    (205, 205, 205),
                    (
                        CENTER_X,
                        CENTER_Y
                    ),
                    TARGET_RADIUS,
                    1
                )


            pygame.draw.circle(
                screen,
                (180, 70, 70),
                (
                    int(x),
                    int(y)
                ),
                6
            )


            title = font.render(
                label_text,
                True,
                (40, 40, 40)
            )

            screen.blit(
                title,
                (15, 15)
            )


            subtitle_text = (
                "Target visible"
                if show_target
                else
                "TARGET HIDDEN"
            )

            subtitle = small_font.render(
                subtitle_text,
                True,
                (100, 100, 100)
            )

            screen.blit(
                subtitle,
                (15, 45)
            )


            pygame.display.flip()


    score = calculate_circle_score(
        points
    )


    return {
        "score": score,
        "canvas": canvas,
        "points": points
    }


# --------------------------------------------------
# PARAMETRELERİ AYNI SEED'LERDE DEĞERLENDİR
# EVALUATE PARAMETERS ON THE SAME SEEDS
# --------------------------------------------------

def evaluate_parameters(
    parameters,
    seeds
):

    scores = []

    for seed in seeds:

        result = run_trial(
            parameters,
            seed=seed,
            show=False
        )

        scores.append(
            result["score"]
        )


    return {
        "mean": float(
            np.mean(scores)
        ),

        "std": float(
            np.std(scores)
        ),

        "scores": scores
    }


# --------------------------------------------------
# SONUÇ KLASÖRÜ
# RESULT FOLDER
# --------------------------------------------------

folder = Path(
    "results/robust_flypaint"
)

folder.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# BAŞLANGIÇ PARAMETRELERİ
# INITIAL PARAMETERS
# --------------------------------------------------

best_parameters = np.array(
    [
        0.12,
        0.12,
        0.15,
        0.15,
        0.000
    ],
    dtype=float
)


rng = np.random.default_rng(
    2026
)


# --------------------------------------------------
# İLK DEĞERLENDİRME
# INITIAL EVALUATION
# --------------------------------------------------

print(
    "\n🧪 Başlangıç motor ayarı değerlendiriliyor..."
)


best_eval = evaluate_parameters(
    best_parameters,
    TRAIN_SEEDS
)


best_score = best_eval[
    "mean"
]


print(
    f"Başlangıç ortalaması: "
    f"{best_score:.2f}"
)

print(
    f"Seed puanları: "
    f"{best_eval['scores']}"
)


# --------------------------------------------------
# EĞİTİM
# TRAINING
# --------------------------------------------------

history = []

current_exploration = (
    EXPLORATION.copy()
)


for trial in range(
    1,
    TRAINING_TRIALS + 1
):

    # Mevcut en iyi parametrelerin yakınında yeni aday oluştur.
    # Create a new candidate near the current best parameters.

    mutation = (
        rng.normal(
            0,
            1,
            size=5
        )
        * current_exploration
    )


    candidate_parameters = (
        best_parameters
        + mutation
    )


    candidate_parameters[
        :4
    ] = np.clip(
        candidate_parameters[:4],
        0.01,
        1.20
    )


    candidate_parameters[
        4
    ] = np.clip(
        candidate_parameters[4],
        -0.05,
        0.05
    )


    print(
        f"\n🪰 Deneme {trial}/{TRAINING_TRIALS}"
    )


    # Adayı HER ZAMAN aynı seed'lerde test et.
    # ALWAYS test the candidate on the same seeds.

    candidate_eval = evaluate_parameters(
        candidate_parameters,
        TRAIN_SEEDS
    )


    candidate_score = (
        candidate_eval["mean"]
    )


    print(
        f"Aday ortalama : "
        f"{candidate_score:.2f}"
    )

    print(
        f"Aday puanları : "
        f"{candidate_eval['scores']}"
    )

    print(
        f"Mevcut en iyi : "
        f"{best_score:.2f}"
    )


    # Adayın ortalaması gerçekten daha iyiyse kabul et.
    # Accept the candidate only if its mean is genuinely better.

    if candidate_score > best_score:

        outcome = "ODUL / REWARD"

        best_parameters = (
            candidate_parameters.copy()
        )

        best_score = (
            candidate_score
        )

        best_eval = (
            candidate_eval
        )


        print(
            "✅ ÖDÜL — yeni ayar kabul edildi."
        )


        # Kabul edilen ayarı bir kez görsel olarak göster.
        # Visually preview each accepted candidate once.

        preview = run_trial(
            best_parameters,
            seed=TRAIN_SEEDS[0],
            show=True,
            show_target=True,
            label_text=(
                f"Accepted trial {trial} | "
                f"Mean score {best_score:.1f}"
            )
        )


        pygame.image.save(
            preview["canvas"],
            folder
            / f"accepted_{trial:02d}.png"
        )


    else:

        outcome = "CEZA / PENALTY"

        print(
            "❌ CEZA — aday reddedildi."
        )


    history.append({
        "trial":
            trial,

        "candidate_mean":
            candidate_score,

        "candidate_std":
            candidate_eval["std"],

        "best_mean":
            best_score,

        "result":
            outcome,

        "seed_64":
            candidate_eval["scores"][0],

        "seed_65":
            candidate_eval["scores"][1],

        "seed_66":
            candidate_eval["scores"][2],

        "left_gain":
            candidate_parameters[0],

        "right_gain":
            candidate_parameters[1],

        "forward_gain":
            candidate_parameters[2],

        "backward_gain":
            candidate_parameters[3],

        "steering_bias":
            candidate_parameters[4]
    })


    current_exploration *= (
        EXPLORATION_DECAY
    )


# --------------------------------------------------
# EĞİTİMDE GÖRÜLMEMİŞ SEED'LERLE TEST
# TEST ON UNSEEN SEEDS
# --------------------------------------------------

print("\n")
print("=" * 60)
print("🧪 HOLDOUT TEST")
print("=" * 60)


holdout_eval = evaluate_parameters(
    best_parameters,
    HOLDOUT_SEEDS
)


print(
    f"Eğitim seed ortalaması : "
    f"{best_score:.2f}"
)

print(
    f"Eğitim seed puanları   : "
    f"{best_eval['scores']}"
)

print(
    f"\nYeni seed ortalaması   : "
    f"{holdout_eval['mean']:.2f}"
)

print(
    f"Yeni seed puanları     : "
    f"{holdout_eval['scores']}"
)


# --------------------------------------------------
# TAMAMEN GİZLİ FİNAL TEST
# FINAL TEST WITH TARGET COMPLETELY HIDDEN
# --------------------------------------------------

print(
    "\n🎨 Hedef gizli final çizimi..."
)


final_result = run_trial(
    best_parameters,
    seed=9999,
    show=True,
    show_target=False,
    label_text=(
        "FINAL HOLDOUT TEST — unseen seed"
    )
)


final_score = (
    final_result["score"]
)


pygame.image.save(
    final_result["canvas"],
    folder
    / "final_hidden_target.png"
)


# --------------------------------------------------
# SONUÇLARI KAYDET
# SAVE RESULTS
# --------------------------------------------------

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    folder
    / "robust_training_history.csv",
    index=False
)


np.save(
    folder
    / "best_parameters.npy",
    best_parameters
)


# --------------------------------------------------
# ÖĞRENME GRAFİĞİ
# LEARNING CURVE
# --------------------------------------------------

import matplotlib.pyplot as plt


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    history_df["trial"],
    history_df["candidate_mean"],
    marker="o",
    label="Candidate mean score"
)

plt.plot(
    history_df["trial"],
    history_df["best_mean"],
    linewidth=2,
    label="Best mean score"
)

plt.xlabel(
    "Training Trial"
)

plt.ylabel(
    "Mean Circle Score"
)

plt.title(
    "Robust FlyPaint Learning Curve"
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


curve_path = (
    folder
    / "robust_training_curve.png"
)

plt.savefig(
    curve_path,
    dpi=160
)

plt.show()


# --------------------------------------------------
# SON ÖZET
# FINAL SUMMARY
# --------------------------------------------------

print("\n")
print("=" * 60)
print("🎨 ROBUST FLYPAINT RESULT")
print("=" * 60)

print(
    f"En iyi training ortalaması : "
    f"{best_score:.2f}"
)

print(
    f"Holdout ortalaması          : "
    f"{holdout_eval['mean']:.2f}"
)

print(
    f"Hedef gizli final seed      : "
    f"{final_score:.2f}"
)

print(
    "\nEn iyi motor parametreleri:"
)

print(
    best_parameters
)

print(
    f"\n📁 Sonuçlar: {folder}"
)


pygame.quit()