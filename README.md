# Is the brain re-learning?

Decoding and tracking recovery in stroke motor-imagery BCI.
Entry by Sudip Sharma in the data analysis track of the
[BR41N.IO hackathon at IEEE SMC 2026](https://www.gtec.at/hackathon/ieee-smc-2026/), using the stroke-rehab dataset provided by g.tec.

![Patient 1: which side of the motor cortex engages, session 1 vs session 25](figures/6_P1_flip.gif)

## The data

Three chronic stroke patients did 25 sessions of motor-imagery BCI therapy with functional electrical stimulation (FES).
In each 8-second trial the patient imagines a left- or right-hand movement (cue at 2 s) and receives FES and visual feedback from 3.5 s.
Each session has an 80-trial calibration run and an 80-trial test run, recorded with 16 EEG channels at 256 Hz.
The dataset contains session 1 ("pre") and session 25 ("post") for each patient.

## What we asked

1. Can an open-source pipeline match the reference accuracies?
2. Is the classifier reading the brain, or the stimulation it triggers?
3. Does a patient need 80 calibration trials every session?
4. What changed in the brain between session 1 and session 25?

## Results

Accuracy is one decision per trial on the test run, with the model fitted on the calibration run.

| Session | Ref. CSP+LDA | Ref. PCA+TVLDA | Ours | Zero calibration |
|---|---|---|---|---|
| P1 pre | 77.1 | 92.9 | 91.2 | 76.2 |
| P1 post | 93.9 | 97.0 | 98.7 | 62.0 |
| P2 pre | 68.4 | 72.4 | 78.8 | 88.8 |
| P2 post | 96.1 | 97.4 | 100.0 | 93.8 |
| P3 pre | 74.4 | 93.6 | 91.2 | 93.8 |
| P3 post | 79.7 | 100.0 | 96.2 | 97.5 |
| **Mean** | 81.6 | 92.2 | **92.7** | 85.3 |

- **Decoding.** A filter-bank Riemannian classifier with causal online recentering reaches 92.7 % (444 of 479 trials, 95 % CI 90.0–94.7), level with the best reference. The CSP+LDA reference is the best time point of a time-resolved accuracy; on that metric ours is 90.0 % against 81.6 %.
- **Brain, not stimulation.** The information sits in the beta band and rises before the stimulation starts. The 55–95 Hz band is weaker but not at chance during feedback.
- **Zero calibration.** A model from the same patient's other session, with no calibration data from the day, reaches 85.3 %. It fails for one session (P1 post, 62 %).
- **Same accuracy gain, three different brains.** Using only 2–3.5 s (after the cue, before any feedback):

| Patient | Imagery-only accuracy, session 1 → 25 | Beta desynchronisation over motor cortex |
|---|---|---|
| P1 | 72.5 → 91.1 % | Session 1: right-hand imagery engages the same-side hemisphere, left-hand imagery has no side. Session 25: left-hand imagery clearly engages the opposite hemisphere and the same-side response is gone. The shift holds in 27 of 27 analysis variants. |
| P2 | 52.5 (chance) → 76.2 % | The right hemisphere responds for both hands; the left stays silent. |
| P3 | 72.5 → 73.8 % | No change. |

## Limits

- Three patients, two sessions each: the per-patient patterns are descriptive.
- Lesion side and clinical scores are not part of the dataset, so any link to motor recovery is a hypothesis.
- Once feedback is on, stimulation-evoked activity cannot be fully separated from imagery.
- The zero-calibration result includes using the later session to decode the earlier one.

## Reproduce

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
scripts/download_data.sh      # fetches the dataset from g.tec into data/
scripts/run_all.sh            # results/ and figures/
```

| File | What it does |
|---|---|
| `src/mi.py` | Loading, epoching, filter-bank features, classifier pipelines |
| `src/benchmark.py` | Six pipelines against the reference accuracies |
| `src/adapt.py` | Online recentering and zero-calibration transfer |
| `src/deeper.py` | Imagery-only decoding, reference-style metric, confidence intervals, per-hemisphere desynchronisation |
| `src/robust.py` | Lateralisation across 27 analysis variants |
| `src/figures_decoding.py`, `src/figures_imagery.py`, `src/animate.py` | Figures and the animation |

The dataset belongs to g.tec medical engineering and is not redistributed here.
