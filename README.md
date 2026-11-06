# VigilEye

Detects fatigue from a webcam by watching eyelid and eye movement. Video is
analysed on this computer and never stored, and session logs hold numbers only.

**Network use:** VigilEye sends nothing while running. The only download is the face
model on first run, if it is missing. MediaPipe is pinned to 0.10.21 on purpose:
later versions include Google usage logging with no opt-out
([issue 6291](https://github.com/google-ai-edge/mediapipe/issues/6291)), so do not
upgrade it without checking this first.

**This is a prototype, not a safety device.** On a small test it told drowsy from
alert clips better than chance, but half of the alert clips set off a false alarm
and a third of the drowsy clips set off none ([results](#results)). It has never
been tested in a car. Do not rely on it to stay safe while driving. If you feel
tired, stop and rest.

## Run it

```bash
.venv/bin/python -m vigileye
```

To set it up from scratch:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

macOS asks for camera permission for the app you launch it from, such as
Terminal. If the camera will not open, allow that app in System Settings >
Privacy & Security > Camera, then restart it.

### Calibration (about 11 seconds)

1. Look straight ahead with your eyes open until the bar fills.
2. At the chime, close your eyes.
3. At the second chime, open them.

The score appears after about 20 seconds of data. Microsleep alarms work
straight away. Recalibrate after moving your seat or the camera.

Keys: `q` quit, `c` recalibrate, `m` mute.

Options: `--source FILE` analyses a video, `--no-window`, `--mute`, `--no-log`.

## What it measures

Eyelid closure is measured against your own calibrated open and closed eyes,
using the 3D eye aspect ratio (EAR).

| Signal | What it means | Score weight |
| --- | --- | --- |
| PERCLOS | Share of the last 60 s with eyes at least 80% closed | 35% |
| Blink length | Blinks get longer with fatigue | 20% |
| Long blinks | Closures of 0.5 s or more, per minute | 20% |
| Eyelid reopening | Lids reopen more slowly with fatigue | 10% |
| Gaze slowing | Drop in saccade rate against your own first 3 minutes | 10% |
| Blink rate | Unusually high rate only | 5% |

PERCLOS alone can raise the score to 70, because it is the strongest single sign.

| Level | Trigger | Response |
| --- | --- | --- |
| ALERT | Score under 30 | None |
| TIRED | Score 30 or more | On-screen only |
| DROWSY | Score 55 or more | Chime every 20 s, banner |
| CRITICAL | Eyes shut for 1 s or more | Spoken alarm until the eyes reopen, then 5 s more |

CRITICAL does not wait for a score, and it fires while the eyes are still shut.

When your face is lost or your head is turned away, analysis pauses and the score
holds. Looking away never counts as recovering. Looking down is not paused,
because that is how a head nods off.

Every threshold is in [vigileye/config.py](vigileye/config.py). Each session
writes a CSV to `logs/`, which is useful for tuning them to you.

## Limitations

- Effectiveness is only partly tested. The core measures (PERCLOS and blink
  duration) have published support, but the score weights and most cut-offs were
  chosen by judgement. Only the 80% closure level and 60 s window follow the
  published PERCLOS definition. See [Results](#results) for the alarm's error rates.
- Studies suggest PERCLOS is less reliable over short windows and for individual
  people, and false alarms are a known weakness. VigilEye uses a 60 s window.
- Saccades are counted, not measured. A 30 fps webcam is too slow to capture
  their peak speed, which is the better fatigue marker. The saccade threshold has
  not been tuned against live webcam noise, so it carries only 10% of the score.
- Lowering your eyes, for example to a dashboard, partly closes the lids and can
  read as closure.
- There is no infrared, so it needs enough light on your face. It will not work
  at night in a dark car. Sunglasses block it, and glasses with strong glare may
  reduce accuracy.
- It tracks one face. In a car, make sure the camera does not pick up a passenger.

## Tests

```bash
.venv/bin/python -m pytest tests
```

`tests/test_fatigue.py` runs simulated eyelid traces with known outcomes at 15,
30 and 60 fps. `tests/test_real_face.py` runs the full app on a real photo with
the eyes edited shut. It needs the photo from scikit-image, and is skipped
without it.

## Measuring accuracy

`evaluation/` scores VigilEye against labelled videos. It is built for
[UTA-RLDD](https://sites.google.com/view/utarldd/home): 180 clips of 60 people, each
self-rated alert, low vigilant or drowsy. Unzip the folds into one folder, then:

```bash
.venv/bin/python -m evaluation extract --data PATH/TO/VIDEOS
```

```bash
.venv/bin/python -m evaluation report
```

`extract` is slow and can be stopped and rerun. It saves numbers only to
`evaluation/cache/`, so the videos can be deleted afterwards. `report` takes seconds,
scores every cached clip and writes to `evaluation/results/`. Use `--folds 1` to run
part of the data, which needs `--data` and the videos.

How the test works:

- **No training:** the shipped thresholds are scored as they are.
- **Main result:** AUC of the mean score, drowsy against alert clips. 0.5 is guessing.
  Intervals resample people, not clips.
- **Alarm result:** sensitivity, specificity and false alarms per hour.
- **Calibration:** these clips have no closed-eye cue, so open EAR comes from the first
  60 s of the person's alert clip, and closed EAR uses the fallback ratio. Use
  `--calibration same-clip` to take it from each clip's own start instead.
- **Fair scoring:** the first 60 s of every clip is skipped, and a clip with a face in
  under half its frames is excluded and named in the report.
- **Tuning:** if thresholds are changed, tune on some folds and report on the others.

Limits of this test: labels are self-rated and cover whole clips, so a drowsy clip
can contain alert moments. The clips were filmed at home, not while driving. A good
result would support the method, but it would not prove the system keeps drivers safe.

The dataset's authors reached 65.2% three-class accuracy with a trained model, and
human judges reached 57.8%
([Ghoddoosian et al., 2019](https://arxiv.org/abs/1904.07312)). Any published result
from this dataset must cite that paper.

## Results

Run on UTA-RLDD Fold 1: 12 people, 36 clips, 616,732 frames, with no clip excluded
and the shipped thresholds unchanged. Results are in `evaluation/results/`.
`same-clip` is the safer column: the open-eye reading shifted between one person's
recordings, by up to 29%, and `alert-clip` can mistake that for closure. It may also
understate drowsiness, because it calibrates on the clip being judged.

| Measure | `alert-clip` | `same-clip` |
| --- | --- | --- |
| AUC, drowsy against alert (0.5 = guessing) | 0.76 (95% CI 0.59 to 0.92) | 0.74 (0.56 to 0.90) |
| Drowsy clip scored above the same person's alert clip | 83% (58% to 100%) | 83% (58% to 100%) |
| Drowsy clips with an alarm | 67% (42% to 92%) | 67% (42% to 92%) |
| Alert clips with no alarm | 50% (25% to 75%) | 50% (25% to 75%) |
| False alarms per hour on alert clips | 6.0 | 6.0 |
| Three-class accuracy (guessing = 33.3%) | 47.2% | 41.7% |

What this shows:

- The score ranks drowsy clips above alert clips better than chance, but 12 people
  leave a wide interval, and the low end is close to guessing.
- The alarm is not good enough to rely on. Every alert clip with an alarm also had a
  detected closure of 1 s or more. The videos were not reviewed, so it is unknown
  whether those were real closures or lowered eyes.
- PERCLOS (AUC 0.83, or 0.80 with `same-clip`) and blink length (0.83, or 0.82) did at
  least as well as the combined score, so the weights may be adding noise. These
  are not tuned results, and changing the score should be tested on other folds.
- Saccade rate did not fall in drowsy clips (AUC 0.38, or 0.40). This gives no support
  for the gaze slowing signal.
- The three-class figure uses alarm cut-offs that were not designed for sorting clips,
  and it covers one fold, so it does not compare directly with the 65.2% above.
- These clips have no closed-eye cue, so the app's measured calibration was not tested.

## References

- Soukupova & Cech (2016), *Real-Time Eye Blink Detection using Facial Landmarks*
- Wierwille et al. (1994), research behind PERCLOS as a drowsiness measure
- Caffier et al. (2003), *Experimental evaluation of eye-blink parameters as a drowsiness measure*
