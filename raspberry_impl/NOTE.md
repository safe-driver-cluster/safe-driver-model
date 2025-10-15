### detect drowsiness based on eye closure frequency
more than 4 times with score > 0.4 in 10 seconds

### differentiate between quick blinks and drowsy eye closures
Filter quick blinks: Only counts closures that last ≥ 0.4 seconds
Quick blinks (< 0.4s) are ignored for drowsy closure count
Drowsy closures (~0.5s or more) are counted

## PERCLOS (Percentage of Eye Closure), which is a key metric for drowsiness detection.