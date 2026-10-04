# Afterglow-FlyBrain

A small experimental project exploring neural activity, short-term aftereffects, plasticity, and simple motor-learning behavior using the **FlyBrain MaleCNS connectome simulation**.

The project gradually moves from basic connectome inspection to controlled stimulus experiments, neural-pattern analysis, dopamine-gated plasticity, and a simple FlyPaint learning task.

## Project Goals

- Inspect the MaleCNS connectome
- Compare baseline, stimulus, and post-stimulus neural activity
- Analyze neurons whose firing patterns change after stimulation
- Explore simplified dopamine-gated plasticity
- Map motor-neuron activity to a drawing task
- Test simple learning and generalization on unseen simulation seeds

## Project Structure

```text
Afterglow-FlyBrain/
├── analysis/
├── data/
├── experiments/
├── results/
├── tools/
├── .gitignore
├── README.md
└── requirements.txt
```

### `tools/`

Utility and calibration scripts.

- 01_check_connectome.py — inspects the connectome dataset
- 07_find_learning_neurons.py — searches for KC, MBON, PAM, PPL1 and DAN neuron types
- 09_calibrate_learning.py — tests different stimulation strengths
- 10_find_usable_mbon.py — measures baseline firing activity of MBON types

### `experiments/`

Main experimental scripts.

- 02_first_afterglow.py — baseline → stimulus → after-stimulus experiment
- 03_control_vs_stimulus.py — compares control and stimulus conditions
- 04_neural_pattern.py — analyzes neurons showing the largest post-stimulus differences
- 05_fly_paint.py — maps motor-neuron activity to a drawing canvas
- 06_watch_and_draw_circle.py — visual exposure followed by free drawing
- 08_dopamine_plasticity_experiment.py — simplified dopamine-gated synaptic plasticity
- 11_mbon26_weight_depression_experiment.py — controlled KC → MBON26 weight-depression experiment
- 12_train_fly_to_draw_circle.py — baseline parameter-learning experiment
- 14_robust_circle_training.py — multi-seed training with holdout evaluation

### `analysis/`

- 13_plot_training_curve.py — plots the FlyPaint training history

## Experimental Progression

The project was developed step by step:
1. Connectome inspection
2. First afterglow experiment
3. Control vs stimulus comparison
4. Neural-pattern analysis
5. Motor-neuron driven FlyPaint
6. Visual exposure experiments
7. Learning-circuit exploration
8. Simplified dopamine-gated plasticity
9. MBON-based experiments
10. Circle-drawing training
11. Robust multi-seed training
12. Holdout testing on unseen seeds

## Robust Training

The robust training experiment evaluates candidate motor parameters on multiple training seeds:
64, 65, 66

The final learned parameters are then evaluated on unseen holdout seeds:
900, 901, 902

This helps test whether improvement is specific to one simulation initialization or generalizes to new starting conditions.
## Results

The results/ directory contains:
- control vs stimulus measurements
- post-stimulus neural activity comparisons
- FlyPaint drawings
- plasticity experiment CSV files
- learning curves
- accepted training stages
- final holdout drawings
Example outputs include:
- results/robust_flypaint/robust_training_curve.png
- results/robust_flypaint/final_hidden_target.png
- results/flypaint_training/training_curve.png

## Data

Large raw connectome files are not included in the repository.
They are excluded through .gitignore because some files are hundreds of megabytes or larger.
Expected raw-data location:
data/raw/

## Installation

Create a virtual environment and install dependencies:
python -m venv .venv

Activate it, then run:
pip install -r requirements.txt

## Running an Experiment

Run scripts from the project root.
Example:
python experiments/03_control_vs_stimulus.py

or:
python experiments/14_robust_circle_training.py

## Notes

This project is exploratory.
The plasticity and learning mechanisms used here are simplified experimental models and should not be interpreted as a complete biological model of learning in Drosophila.
The goal is to experiment with neural dynamics, plasticity concepts, and simple learning behavior using connectome-based simulation.