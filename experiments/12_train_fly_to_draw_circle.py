from flybrain import FlyBrain
from flybrain.eyes import FeatureDetectors
import pygame
import numpy as np
import pandas as pd
from pathlib import Path
import math
import time


# --------------------------------------------------
# GENEL AYARLAR
# GENERAL SETTINGS
# --------------------------------------------------

WIDTH = 900
HEIGHT = 650

SEED = 64

CENTER_X = WIDTH // 2
CENTER_Y = HEIGHT // 2

TARGET_RADIUS = 160

# Toplam eğitim denemesi.
# Total number of training trials.
TRAINING_TRIALS = 22

# Her çizim için daha uzun süre veriyoruz.
# Give each drawing more time to complete.

DRAW_STEPS = 360

# Her denemeden önce beynin kısa süre oturmasını bekle.
# Let the brain settle briefly before each trial.
SETTLE_STEPS = 50

# İlk başta sineğe kaç tur dairesel hedef gösterelim?
# Number of circular target laps shown before training.
WATCH_LAPS = 5

STEPS_PER_LAP = 100


# --------------------------------------------------
# ÖĞRENME AYARLARI
# LEARNING SETTINGS
# --------------------------------------------------

# Yeni motor ayarlarının ne kadar değişebileceği.
# Amount of exploration applied to motor parameters.
EXPLORATION = np.array([
    0.20,   # sol dönüş kazancı / left steering gain
    0.20,   # sağ dönüş kazancı / right steering gain
    0.20,   # ileri kazancı / forward gain
    0.20,   # geri kazancı / backward gain
    0.012   # hafif dönüş biası / small steering bias
])

# Keşif zamanla biraz azalacak.
# Exploration gradually decreases.
EXPLORATION_DECAY = 0.94


# --------------------------------------------------
# BEYNİ YÜKLE
# LOAD THE BRAIN
# --------------------------------------------------

print("🧠 MaleCNS hazırlanıyor...")

brain = FlyBrain(
    device="auto",
    seed=SEED,
    sensory_input=False
)

vision = FeatureDetectors(brain)

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
# PYGAME BAŞLAT
# START PYGAME
# --------------------------------------------------

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "FlyPaint Learning — MaleCNS"
)

font = pygame.font.Font(
    None,
    27
)

small_font = pygame.font.Font(
    None,
    22
)

clock = pygame.time.Clock()


# --------------------------------------------------
# ÖNCE SİNEĞE DAİRE HAREKETİNİ GÖSTER
# FIRST SHOW THE FLY A CIRCULAR MOTION
# --------------------------------------------------

print("\n👁️ Sinek dairesel hedefi izliyor...")


for step in range(
    WATCH_LAPS * STEPS_PER_LAP
):

    angle = (
        2
        * math.pi
        * (step % STEPS_PER_LAP)
        / STEPS_PER_LAP
    )

    # Görsel encoder yatay konum ve görünür boyut kullanıyor.
    # The visual encoder uses horizontal position and apparent size.

    x_target = math.cos(angle)

    y_target = math.sin(angle)

    horizontal_position = (
        x_target * 80
    )

    apparent_size = (
        14
        + (y_target + 1) * 10
    )

    injections = vision.inject(
        opp=(
            horizontal_position,
            apparent_size
        )
    )

    brain.step(
        inject=injections
    )


print("✅ Dairesel gösterim tamamlandı.")


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

    # Noktaların hedef çembere uzaklığı.
    # Distance of points from the target circle.

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


    # Çemberin kaç farklı bölümünü gezdi?
    # How much of the circle was covered?

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

    # Sadece çembere yeterince yakın noktaları say.
    # Count only points reasonably close to the circle.

    good_points = (
        normalized_error < 0.25
    )

    covered_bins = len(
        np.unique(
            bins[good_points]
        )
    )

    coverage = (
        covered_bins / 24
    )


    # Daireye yakınlık kalitesi.
    # Quality of radial accuracy.

    shape_quality = max(
        0.0,
        1.0
        - mean_error * 2.5
    )


    # Hem yuvarlak olmalı hem çemberin çoğunu gezmeli.
    # It must stay near the ring and cover most of it.

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
# TEK BİR ÇİZİM DENEMESİ
# RUN ONE DRAWING TRIAL
# --------------------------------------------------

def run_trial(
    parameters,
    trial_number,
    show=True,
    show_target=True
):

    left_gain = parameters[0]
    right_gain = parameters[1]

    forward_gain = parameters[2]
    backward_gain = parameters[3]

    steering_bias = parameters[4]


    # Her denemede aynı temel başlangıç durumundan başla.
    # Begin each trial from a comparable neural state.

    brain.reset(
        seed=SEED + trial_number
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


    # Kalem hedef dairenin sağ kenarından başlıyor.
    # The pen begins on the right edge of the target circle.

    x = float(
        CENTER_X
        + TARGET_RADIUS
    )

    y = float(
        CENTER_Y
    )

    # Yukarı doğru bakarak başla.
    # Begin facing upward.

    heading = (
        -math.pi / 2
    )


    points = [
        (x, y)
    ]


    total_left = 0
    total_right = 0

    total_forward = 0
    total_backward = 0


    running = True


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


        total_left += left_count
        total_right += right_count

        total_forward += forward_count
        total_backward += backward_count


        # MaleCNS motor çıktısını dönüş miktarına çevir.
        # Convert MaleCNS motor output into steering.

        steering = (
            right_count
            * right_gain

            -

            left_count
            * left_gain

            +

            steering_bias
        )


        # Çok sert dönüşü engelle.
        # Prevent extreme turns.

        steering = float(
            np.clip(
                steering,
                -0.10,
                0.10
            )
        )


        heading += steering


        # İleri ve geri nöronları hızı etkiliyor.
        # Forward and backward neurons affect movement speed.

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


        # Tuval dışına çıkmayı engelle.
        # Keep the pen inside the canvas.

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


            # Gri çizgi yalnızca hedefi göstermek için.
            # The gray ring only displays the target.

            # Eğitim sırasında hedef çemberi göster.
# Show the target circle during training.

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


            text = font.render(
                (
                    f"Trial {trial_number} | "
                    f"Step {step}/{DRAW_STEPS}"
                ),
                True,
                (40, 40, 40)
            )

            screen.blit(
                text,
                (15, 15)
            )


            # Ekranda deney durumunu belirt.
            # Display the current experiment state.

            if show_target:

                label_text = (
                    "Gray circle = target only"
                )

            else:

                label_text = (
                    "FINAL TEST — target hidden"
                )


            label = small_font.render(
                label_text,
                True,
                (100, 100, 100)
            )

            screen.blit(
                label,
                (15, 45)
            )


            pygame.display.flip()


    score = calculate_circle_score(
        points
    )


    return {
        "score": score,

        "points": points,

        "canvas": canvas,

        "left_spikes": total_left,
        "right_spikes": total_right,

        "forward_spikes": total_forward,
        "backward_spikes": total_backward
    }


# --------------------------------------------------
# SONUÇ KLASÖRÜ
# RESULT FOLDER
# --------------------------------------------------

folder = Path(
    "results/flypaint_training"
)

folder.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# BAŞLANGIÇ MOTOR AYARLARI
# INITIAL MOTOR PARAMETERS
# --------------------------------------------------

rng = np.random.default_rng(
    2026
)


# [sol, sağ, ileri, geri, dönüş biası]
# [left, right, forward, backward, steering bias]

best_parameters = np.array([
    0.12,
    0.12,
    0.15,
    0.15,
    0.000
], dtype=float)


# --------------------------------------------------
# İLK ÇİZİM
# INITIAL DRAWING
# --------------------------------------------------

print("\n🎨 Eğitim öncesi çizim...")

initial = run_trial(
    best_parameters,
    trial_number=0
)

best_score = initial[
    "score"
]


pygame.image.save(
    initial["canvas"],
    folder
    / "trial_00_before_training.png"
)


print(
    f"Başlangıç Circle Score: "
    f"{best_score:.2f}"
)


# --------------------------------------------------
# ÖDÜL / CEZA İLE EĞİTİM
# REWARD / PENALTY TRAINING
# --------------------------------------------------

history = []


for trial in range(
    1,
    TRAINING_TRIALS + 1
):

    # Mevcut en iyi motor ayarının biraz değiştirilmiş halini dene.
    # Try a slightly mutated version of the current best decoder.

    mutation = (
        rng.normal(
            0,
            1,
            size=5
        )
        * EXPLORATION
    )


    candidate_parameters = (
        best_parameters
        + mutation
    )


    # Motor kazançlarını mantıklı sınırlar içinde tut.
    # Keep motor gains within reasonable limits.

    candidate_parameters[:4] = np.clip(
        candidate_parameters[:4],
        0.01,
        1.20
    )

    candidate_parameters[4] = np.clip(
        candidate_parameters[4],
        -0.05,
        0.05
    )


    print(
        f"\n🪰 Deneme {trial}/{TRAINING_TRIALS}"
    )


    result = run_trial(
        candidate_parameters,
        trial_number=trial
    )


    score = result[
        "score"
    ]


    # Öncekinden daha iyiyse ödüllendir ve yeni ayarı sakla.
    # If it performs better, reward it and keep the new parameters.

    if score > best_score:

        outcome = "ODUL / REWARD"

        best_score = score

        best_parameters = (
            candidate_parameters.copy()
        )

        pygame.image.save(
            result["canvas"],
            folder
            / "best_drawing.png"
        )

        print(
            f"✅ {outcome}"
        )

        print(
            f"Yeni en iyi puan: "
            f"{best_score:.2f}"
        )


    else:

        # Daha kötüyse bu değişiklik kalıcı olmuyor.
        # If it performs worse, the mutation is discarded.

        outcome = "CEZA / PENALTY"

        print(
            f"❌ {outcome}"
        )

        print(
            f"Puan: {score:.2f} | "
            f"En iyi: {best_score:.2f}"
        )


    # Her denemenin çizimini kaydet.
    # Save every trial drawing.

    pygame.image.save(
        result["canvas"],
        folder
        / f"trial_{trial:02d}.png"
    )


    history.append({
        "trial": trial,

        "score": score,

        "best_score": best_score,

        "result": outcome,

        "left_gain":
            candidate_parameters[0],

        "right_gain":
            candidate_parameters[1],

        "forward_gain":
            candidate_parameters[2],

        "backward_gain":
            candidate_parameters[3],

        "steering_bias":
            candidate_parameters[4],

        "left_spikes":
            result["left_spikes"],

        "right_spikes":
            result["right_spikes"],

        "forward_spikes":
            result["forward_spikes"],

        "backward_spikes":
            result["backward_spikes"]
    })


    # Eğitim ilerledikçe değişiklikleri küçült.
    # Reduce exploration as training progresses.

    EXPLORATION *= (
        EXPLORATION_DECAY
    )


# --------------------------------------------------
# SON DEĞERLENDİRME
# FINAL EVALUATION
# --------------------------------------------------

print(
    "\n🧪 Öğrenilmiş motor ayarıyla son çizim..."
)


# Son testte hedef daireyi gizliyoruz.
# Hide the target circle during the final test.

final_result = run_trial(
    best_parameters,
    trial_number=999,
    show_target=False
)


pygame.image.save(
    final_result["canvas"],
    folder
    / "final_after_training.png"
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
    / "training_history.csv",
    index=False
)


np.save(
    folder
    / "best_parameters.npy",
    best_parameters
)


print("\n")
print("=" * 60)
print("🎨 FLYPAINT TRAINING RESULT")
print("=" * 60)

print(
    f"Eğitim öncesi puan : "
    f"{initial['score']:.2f}"
)

print(
    f"Eğitimde en iyi    : "
    f"{best_score:.2f}"
)

print(
    f"Son test puanı     : "
    f"{final_result['score']:.2f}"
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