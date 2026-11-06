# VigilEye accuracy report

Data: data/utarldd/videos, folds 1. Calibration: same-clip.

36 clips scored, 12 people with both an alert and a drowsy clip, 0 clips unusable.

## Drowsy against alert

| Measure | Result |
| --- | --- |
| AUC of mean score (0.5 = guessing) | 0.74 (95% CI 0.56 to 0.90) |
| Drowsy clip beat the same person's alert clip | 0.83 (95% CI 0.58 to 1.00) |
| Sensitivity (drowsy clips with an alarm) | 0.67 (95% CI 0.42 to 0.92) |
| Specificity (alert clips with no alarm) | 0.50 (95% CI 0.25 to 0.75) |
| False alarms per hour on alert clips | 6.0 (11 in 1.8 h) |

## Each signal alone

| Signal | AUC |
| --- | --- |
| Fatigue score | 0.74 (95% CI 0.56 to 0.90) |
| PERCLOS | 0.80 (95% CI 0.67 to 0.94) |
| Blink length | 0.82 (95% CI 0.62 to 0.97) |
| Long blinks | 0.74 (95% CI 0.54 to 0.91) |
| Eyelid reopening | 0.56 (95% CI 0.38 to 0.75) |
| Blink rate | 0.65 (95% CI 0.47 to 0.83) |
| Saccade rate (lower = more tired) | 0.40 (95% CI 0.24 to 0.56) |

## By label

| Label | Clips | Mean score | Clips with an alarm |
| --- | --- | --- | --- |
| Alert | 12 | 12.8 | 50% |
| Low vigilant | 12 | 22.9 | 75% |
| Drowsy | 12 | 28.2 | 67% |

Three-class accuracy from mean score: 41.7% (guessing: 33.3%).
