from flybrain import FlyBrain
from flybrain.eyes import FeatureDetectors
import pygame
import numpy as np
import pandas as pd
from pathlib import Path
import math


# --------------------------------------------------
# AYARLAR
# SETTINGS
# --------------------------------------------------

WIDTH = 900
HEIGHT = 650

SEED = 64

SETTLE_STEPS = 200

# Sineğin daireyi kaç tur izleyeceği.
# Number of circle laps the fly will watch.
CIRCLE_LAPS = 5

# Bir turun kaç simülasyon adımı süreceği.
# Number of simulation steps per lap.
STEPS_PER_LAP = 100

# Çizim aşamasının uzunluğu.
# Length of the drawing phase.
DRAW_STEPS = 1500

MOVE_SCALE = 3


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

vision = FeatureDetectors(brain)

print("✅ Beyin hazır.")


# --------------------------------------------------
# MOTOR NÖRONLARI
# MOTOR NEURONS
# --------------------------------------------------

steer_left = set(
    int(x)
    for x in brain.cells(["DNa02"], side="L")
)

steer_right = set(
    int(x)
    for x in brain.cells(["DNa02"], side="R")
)

forward = set(
    int(x)
    for x in brain.cells(["DNg100"])
)

backward = set(
    int(x)
    for x in brain.cells(["MDN"])
)


print("\nMotor nöronları:")
print("DNa02 L:", len(steer_left))
print("DNa02 R:", len(steer_right))
print("DNg100 :", len(forward))
print("MDN    :", len(backward))


# --------------------------------------------------
# BEYNİN OTURMASINI BEKLE
# LET THE BRAIN SETTLE
# --------------------------------------------------

print("\n🪰 Beyin oturuyor...")

for _ in range(SETTLE_STEPS):
    brain.step()


# --------------------------------------------------
# DAİRE ŞEKLİNDE HAREKET EDEN HEDEFİ GÖSTER
# SHOW A TARGET MOVING IN A CIRCLE
# --------------------------------------------------

print("\n👁️ Sinek daire hareketini izliyor...")

total_watch_steps = (
    CIRCLE_LAPS * STEPS_PER_LAP
)

for step in range(total_watch_steps):

    # Dairenin o andaki açısı.
    # Current angle of the circle.
    angle = (
        2 * math.pi
        * (step % STEPS_PER_LAP)
        / STEPS_PER_LAP
    )

    # Yatay konum.
    # Horizontal position.
    x_target = math.cos(angle)

    # Dikey konumu doğrudan göremediğimiz için
    # hedefin boyutuna/distance cue'ya çeviriyoruz.
    # Since vertical position is not directly represented,
    # encode it as apparent target size / distance.
    y_target = math.sin(angle)

    dx = x_target * 80

    size = (
        14
        + (y_target + 1) * 10
    )

    injections = vision.inject(
        opp=(dx, size)
    )

    brain.step(
        inject=injections
    )


print("✅ Görsel hedef kaldırıldı.")


# --------------------------------------------------
# KISA BİR SESSİZ DÖNEM
# SHORT QUIET PERIOD
# --------------------------------------------------

# Artık hiçbir görsel giriş yok.
# There is now no visual input.

for _ in range(25):
    brain.step()


# --------------------------------------------------
# PAINT'İ BAŞLAT
# START PAINT
# --------------------------------------------------

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "FlyPaint — After Watching a Circle"
)

canvas = pygame.Surface(
    (WIDTH, HEIGHT)
)

canvas.fill(
    (250, 248, 243)
)

font = pygame.font.Font(
    None,
    25
)


x = WIDTH // 2
y = HEIGHT // 2

records = []

running = True
simulation_step = 0


print("\n🎨 Şimdi sinek serbestçe çiziyor...")


# --------------------------------------------------
# ÇİZİM DÖNGÜSÜ
# DRAWING LOOP
# --------------------------------------------------

while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


    fired = brain.step()

    fired_set = set(
        int(n)
        for n in fired
    )


    # Motor nöronlarının o adımda ateşleyip
    # ateşlemediğini ölç.
    # Measure whether each motor group fired
    # during this step.

    left_count = len(
        fired_set & steer_left
    )

    right_count = len(
        fired_set & steer_right
    )

    forward_count = len(
        fired_set & forward
    )

    backward_count = len(
        fired_set & backward
    )


    horizontal = (
        right_count - left_count
    )

    vertical = (
        backward_count - forward_count
    )


    dx = int(
        np.clip(
            horizontal * MOVE_SCALE,
            -12,
            12
        )
    )

    dy = int(
        np.clip(
            vertical * MOVE_SCALE,
            -12,
            12
        )
    )


    old_x = x
    old_y = y

    x += dx
    y += dy


    # Kalemi tuval sınırlarında tut.
    # Keep the pen inside the canvas.

    x = int(
        np.clip(x, 5, WIDTH - 5)
    )

    y = int(
        np.clip(y, 5, HEIGHT - 5)
    )


    if (x, y) != (old_x, old_y):

        pygame.draw.line(
            canvas,
            (30, 30, 30),
            (old_x, old_y),
            (x, y),
            2
        )


    records.append({
        "step": simulation_step,
        "x": x,
        "y": y,
        "left": left_count,
        "right": right_count,
        "forward": forward_count,
        "backward": backward_count
    })


    screen.blit(
        canvas,
        (0, 0)
    )


    # Kırmızı nokta sineğin/kalemin mevcut konumu.
    # The red dot is the fly/pen's current position.

    pygame.draw.circle(
        screen,
        (180, 70, 70),
        (x, y),
        6
    )


    status = font.render(
        f"After circle | Step {simulation_step}",
        True,
        (40, 40, 40)
    )

    screen.blit(
        status,
        (15, 15)
    )


    pygame.display.flip()

    simulation_step += 1


    if simulation_step >= DRAW_STEPS:
        running = False


# --------------------------------------------------
# SONUÇLARI KAYDET
# SAVE RESULTS
# --------------------------------------------------

folder = Path(
    "results/flypaint"
)

folder.mkdir(
    parents=True,
    exist_ok=True
)


image_path = (
    folder
    / "flypaint_after_circle.png"
)

pygame.image.save(
    canvas,
    image_path
)


csv_path = (
    folder
    / "flypaint_after_circle.csv"
)

pd.DataFrame(
    records
).to_csv(
    csv_path,
    index=False
)


pygame.quit()


print("\n🎨 Tamamlandı!")

print(
    f"🖼️ Çizim: {image_path}"
)

print(
    f"📊 Veri: {csv_path}"
)