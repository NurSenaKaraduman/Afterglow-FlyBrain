import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    "data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

print("MaleCNS yükleniyor...")

brain = pd.read_feather(DATA_PATH)

print("\n--- BOYUT ---")
print(brain.shape)

print("\n--- SÜTUNLAR ---")
print(brain.columns.tolist())

print("\n--- İLK 5 SATIR ---")
print(brain.head())

print("\n--- VERİ TİPLERİ ---")
print(brain.dtypes)

print("\nHazır 🪰")