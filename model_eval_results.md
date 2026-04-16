# Grad-CAM Analysis

## Overview

Evaluation includes:
- Samples for both correct and incorrect predictions
- Confidence scores
- Full test set accuracy

---

## CORRECT CASES

True NORMAL:

- Idx: 0, True: NORMAL, Prediction: NORMAL, Confidence: 0.951
- Idx: 1, True: NORMAL, Prediction: NORMAL, Confidence: 0.995
- Idx: 2, True: NORMAL, Prediction: NORMAL, Confidence: 0.984
- Idx: 3, True: NORMAL, Prediction: NORMAL, Confidence: 0.748
- Idx: 4, True: NORMAL, Prediction: NORMAL, Confidence: 0.965

True PNEUMONIA:

- Idx: 234, True: PNEUMONIA, Prediction: PNEUMONIA, Confidence: 0.994
- Idx: 235, True: PNEUMONIA, Prediction: PNEUMONIA, Confidence: 0.997
- Idx: 236, True: PNEUMONIA, Prediction: PNEUMONIA, Confidence: 1.000
- Idx: 237, True: PNEUMONIA, Prediction: PNEUMONIA, Confidence: 0.999
- Idx: 238, True: PNEUMONIA, Prediction: PNEUMONIA, Confidence: 0.999

---

## INCORRECT CASES

True NORMAL:

- Idx: 8, True: NORMAL, Prediction: PNEUMONIA, Confidence: 0.779
- Idx: 10, True: NORMAL, Prediction: PNEUMONIA, Confidence: 0.565
- Idx: 11, True: NORMAL, Prediction: PNEUMONIA, Confidence: 0.861
- Idx: 14, True: NORMAL, Prediction: PNEUMONIA, Confidence: 0.992
- Idx: 16, True: NORMAL, Prediction: PNEUMONIA, Confidence: 0.993

True PNEUMONIA:

- Idx: 391, True: PNEUMONIA, Prediction: NORMAL, Confidence: 0.848
- Idx: 473, True: PNEUMONIA, Prediction: NORMAL, Confidence: 0.606
- Idx: 496, True: PNEUMONIA, Prediction: NORMAL, Confidence: 0.696
- Idx: 599, True: PNEUMONIA, Prediction: NORMAL, Confidence: 0.639
- (only 4 cases where a PNEUMONIA patient was predicted as NORMAL)

---

## Full Test Set Evaluation

- Correct Predictions: 553  
- Wrong Predictions: 71  
- Total Predictions: 624  

---

## Final Performance

Test Accuracy: 0.8862 (88.62%)
