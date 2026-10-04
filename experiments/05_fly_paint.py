from flybrain import FlyBrain
import pygame
import numpy as np
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# AYARLAR
# SETTINGS
# --------------------------------------------------

WIDTH = 900
HEIGHT = 650

SIM_STEPS_PER_FRAME = 5

MAX_SIMULATION_STEPS = 1500

MOVE_SCALE = 3

SEED = 64


# --------------------------------------------------
# BEYNİ OLUŞTUR
# CREATE THE BRAIN
# --------------------------------------------------

print("🧠 MaleCNS hazırlanıyor...")

brain = FlyBrain(
    device="auto",
    seed=SEED,
    sensory_input=False
)

print("✅ Beyin hazır.")


# --------------------------------------------------
# HAREKET NÖRONLARINI BUL
# FIND THE MOTOR NEURONS
# --------------------------------------------------

steer_left = brain.cells(
    ["DNa02"],
    side="L"
)

steer_right = brain.cells(
    ["DNa02"],
    side="R"
)

forward = brain.cells(
    ["DNg100"]
)

backward = brain.cells(
    ["MDN"]
)


print("\nMotor nöronları:")
print(f"DNa02 Left  : {len(steer_left)}")
print(f"DNa02 Right : {len(steer_right)}")
print(f"DNg100      : {len(forward)}")
print(f"MDN         : {len(backward)}")


# --------------------------------------------------
# MOTOR GRUPLARINI SET'E ÇEVİR
# CONVERT MOTOR GROUPS TO SETS
# --------------------------------------------------

steer_left_set = set(
    int(x) for x in steer_left
)

steer_right_set = set(
    int(x) for x in steer_right
)

forward_set = set(
    int(x) for x in forward
)

backward_set = set(
    int(x) for x in backward
)


# --------------------------------------------------
# BEYNİN OTURMASINI BEKLE
# LET THE BRAIN SETTLE
# --------------------------------------------------

print("\n🪰 Beyin oturuyor...")

for _ in range(200):
    brain.step()

print("🎨 Çizim başlıyor!")


# --------------------------------------------------
# PYGAME
# PYGAME
# --------------------------------------------------

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "FlyPaint — MaleCNS"
)

canvas = pygame.Surface(
    (WIDTH, HEIGHT)
)

canvas.fill(
    (250, 248, 243)
)

font = pygame.font.Font(
    None,
    26
)


# --------------------------------------------------
# KALEMİN BAŞLANGIÇ NOKTASI
# INITIAL PEN POSITION
# --------------------------------------------------

x = WIDTH // 2
y = HEIGHT // 2

previous_x = x
previous_y = y


# --------------------------------------------------
# DENEY VERİLERİ
# EXPERIMENT DATA
# --------------------------------------------------

records = []

simulation_step = 0

running = True


# --------------------------------------------------
# ANA DÖNGÜ
# MAIN LOOP
# --------------------------------------------------

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    # Bu karedeki motor aktivitelerini say.
    # Count motor activity during this frame.

    left_count = 0
    right_count = 0

    forward_count = 0
    backward_count = 0


    for _ in range(SIM_STEPS_PER_FRAME):

        fired = brain.step()

        fired_set = set(
            int(x) for x in fired
        )

        left_count += len(
            fired_set & steer_left_set
        )

        right_count += len(
            fired_set & steer_right_set
        )

        forward_count += len(
            fired_set & forward_set
        )

        backward_count += len(
            fired_set & backward_set
        )

        simulation_step += 1


    # Sağ-sol nöron farkını yatay harekete çevir.
    # Convert left-right neural difference into horizontal movement.

    horizontal = (
        right_count - left_count
    )


    # İleri-geri nöron farkını dikey harekete çevir.
    # Convert forward-backward neural difference into vertical movement.

    vertical = (
        backward_count - forward_count
    )


    dx = horizontal * MOVE_SCALE

    dy = vertical * MOVE_SCALE


    # Çok büyük sıçramaları engelle.
    # Prevent excessively large jumps.

    dx = int(
        np.clip(dx, -12, 12)
    )

    dy = int(
        np.clip(dy, -12, 12)
    )


    previous_x = x
    previous_y = y


    x += dx
    y += dy


    # Kalemi tuvalin içinde tut.
    # Keep the pen inside the canvas.

    x = int(
        np.clip(x, 5, WIDTH - 5)
    )

    y = int(
        np.clip(y, 5, HEIGHT - 5)
    )


    # Sinek hareket ettiyse çizgi çiz.
    # Draw a line if the fly moved.

    if x != previous_x or y != previous_y:

        pygame.draw.line(
            canvas,
            (35, 35, 35),
            (previous_x, previous_y),
            (x, y),
            2
        )


    # Verileri kaydet.
    # Record the data.

    records.append({

        "step": simulation_step,

        "x": x,
        "y": y,

        "left_spikes": left_count,
        "right_spikes": right_count,

        "forward_spikes": forward_count,
        "backward_spikes": backward_count
    })


    # Tuvali ekrana çiz.
    # Draw the canvas on the screen.

    screen.blit(
        canvas,
        (0, 0)
    )


    # Sineğin mevcut konumunu göster.
    # Show the fly's current position.

    pygame.draw.circle(
        screen,
        (180, 80, 80),
        (x, y),
        6
    )


    # Bilgi yazısını göster.
    # Display status information.

    status = (
        f"Step: {simulation_step} | "
        f"L:{left_count} "
        f"R:{right_count} "
        f"F:{forward_count} "
        f"B:{backward_count}"
    )

    text = font.render(
        status,
        True,
        (40, 40, 40)
    )

    screen.blit(
        text,
        (15, 15)
    )


    pygame.display.flip()


    # Belirlenen süre dolunca bitir.
    # Stop when the requested simulation length is reached.

    if simulation_step >= MAX_SIMULATION_STEPS:
        running = False


# --------------------------------------------------
# SONUÇ DOSYALARINI HAZIRLA
# PREPARE RESULT FILES
# --------------------------------------------------

results_folder = Path(
    "results/flypaint"
)

results_folder.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# ÇİZİMİ PNG OLARAK KAYDET
# SAVE THE DRAWING AS PNG
# --------------------------------------------------

image_path = (
    results_folder
    / "flypaint_no_stimulus.png"
)

pygame.image.save(
    canvas,
    image_path
)


# --------------------------------------------------
# HAREKET VERİSİNİ CSV OLARAK KAYDET
# SAVE MOVEMENT DATA AS CSV
# --------------------------------------------------

df = pd.DataFrame(
    records
)

csv_path = (
    results_folder
    / "flypaint_no_stimulus.csv"
)

df.to_csv(
    csv_path,
    index=False
)


pygame.quit()


print("\n🎨 FlyPaint tamamlandı!")

print(
    f"🖼️ Resim: {image_path}"
)

print(
    f"📊 Veri: {csv_path}"
)