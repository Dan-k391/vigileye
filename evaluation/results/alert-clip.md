# VigilEye accuracy report

Data: data/utarldd/videos, folds 1. Calibration: alert-clip.

36 clips scored, 12 people with both an alert and a drowsy clip, 0 clips unusable.

## Drowsy against alert

| Measure | Result |
| --- | --- |
| AUC of mean score (0.5 = guessing) | 0.76 (95% CI 0.59 to 0.92) |
| Drowsy clip beat the same person's alert clip | 0.83 (95% CI 0.58 to 1.00) |
| Sensitivity (drowsy clips with an alarm) | 0.67 (95% CI 0.42 to 0.92) |
| Specificity (alert clips with no alarm) | 0.50 (95% CI 0.25 to 0.75) |
| False alarms per hour on alert clips | 6.0 (11 in 1.8 h) |

## Each signal alone

| Signal | AUC |
| --- | --- |
| Fatigue score | 0.76 (95% CI 0.59 to 0.92) |
| PERCLOS | 0.83 (95% CI 0.69 to 0.95) |
| Blink length | 0.83 (95% CI 0.69 to 0.96) |
| Long blinks | 0.79 (95% CI 0.59 to 0.95) |
| Eyelid reopening | 0.66 (95% CI 0.53 to 0.83) |
| Blink rate | 0.62 (95% CI 0.45 to 0.79) |
| Saccade rate (lower = more tired) | 0.38 (95% CI 0.22 to 0.54) |

## By label

| Label | Clips | Mean score | Clips with an alarm |
| --- | --- | --- | --- |
| Alert | 12 | 12.8 | 50% |
| Low vigilant | 12 | 32.3 | 92% |
| Drowsy | 12 | 35.2 | 67% |

Three-class accuracy from mean score: 47.2% (guessing: 33.3%).
