# What each result can establish

The original numerical results and committed figures are retained. This review
changes interpretation, input validation and portability, and adds a separate
training-only mu-power analysis. It does not establish clinical efficacy.

## Fitting and temporal scope

Within-session models fit the training run and score the separate test run.
The benchmark includes training-run CV. Further comparisons of adaptation,
bands or time windows use already exposed test outcomes and remain exploratory.
A maximum over scored test time points is a descriptive maximum, not an
independent estimate for a time point selected without those labels.

`adapt.py` cross-session transfer pools labelled training and test runs from
the source sessions. The target session's training labels are excluded, but
source labels are required. Same-person transfer includes both PRE→POST and
POST→PRE. Only the first follows the release's chronology; neither evaluates
a live calibration-free rehabilitation deployment.

`rc_online` updates its reference using covariances of the current and earlier
completed trials without labels. The input covariances still derive from
continuous zero-phase filtering. This is trial-ordered recentering in an
offline pipeline, not causal acquisition-to-decision replay.

## Early windows and the animation

`mi.epoch` applies `sosfiltfilt` before selecting epochs. `deeper.py` and
`robust.py` compute Hilbert power across complete epochs, and `animate.py`
adds centred 0.5-second smoothing. Later samples can therefore affect early
estimates. The documentation's approximate +3.5 s boundary is not an actual
per-trial feedback timestamp. Hand-specific cues, FES, artifacts and reference
effects remain alternative explanations of class differences.

The liked original animation is preserved. It uses median scalp beta power,
baseline normalisation, and subtraction of the across-sensor mean **after**
conversion to dB. This is not common-average referencing of raw EEG. Blue
denotes lower relative power; it need not be an absolute decrease. Standard
montage interpolation supplies display coordinates, not patient anatomy.
The generator's revised labels are intended for a future explicitly regenerated
figure; the checked-in original GIF retains its historical labels.

The 27 band/reference/window variants reuse the same data; nine use the
documented feedback window. They are sensitivity checks, not 27 independent
confirmations of a recovery mechanism. Legacy `imagery_*` result keys remain
for compatibility and refer to the early post-cue analysis.

## Counts and uncertainty

The loader retains the original full −2 to +8 s epoch boundary rule, with
optional metadata reporting exclusions. P1 POST excludes one test trial whose
baseline is outside the recording; original pooled decoding uses 479 trials.
Trial-wise Wilson intervals and within-session bootstrap intervals assume
sampling properties that do not establish uncertainty across new patients.
There are three participants, with no supplied lesion-side or clinical outcome
labels. A session difference is not evidence of treatment causality.

## Added analysis

The mu notebook uses all six training runs, including one-second pre-attention
baselines, C3/C4, and fixed 8–13 Hz window-local Welch estimates. No classifier
or test recording is used. It has a different band, baseline and estimator
from the original beta/Hilbert plots, so it is not a direct causal-filter
comparison. Parameters, input hashes and checks accompany its outputs.
