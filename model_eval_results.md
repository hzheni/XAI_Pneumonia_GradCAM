# Pneumonia Detection Model - Evaluation Results

## Overview
This document summarizes the performance of a ResNet18-based model trained to classify chest X-ray images into two categories:

- NORMAL
- PNEUMONIA

Evaluation includes:
- Sample predictions for both classes
- Confidence scores
- Full test set accuracy

---

## NORMAL CASES

- True: NORMAL, Pred: NORMAL, Confidence: 0.951  
- True: NORMAL, Pred: NORMAL, Confidence: 0.995  
- True: NORMAL, Pred: NORMAL, Confidence: 0.984  
- True: NORMAL, Pred: NORMAL, Confidence: 0.748  
- True: NORMAL, Pred: NORMAL, Confidence: 0.965  

---

## PNEUMONIA CASES

- True: PNEUMONIA, Pred: PNEUMONIA, Confidence: 0.994  
- True: PNEUMONIA, Pred: PNEUMONIA, Confidence: 0.997  
- True: PNEUMONIA, Pred: PNEUMONIA, Confidence: 1.000  
- True: PNEUMONIA, Pred: PNEUMONIA, Confidence: 0.999  
- True: PNEUMONIA, Pred: PNEUMONIA, Confidence: 0.999  

---

## Full Test Set Evaluation

- Correct Predictions: 553  
- Wrong Predictions: 71  
- Total Predictions: 624  

---

## Final Performance

Test Accuracy: 0.8862 (88.62%)

---

## Notes

- The model performs well on both classes.
- Predictions show consistently high confidence.
- Grad-CAM was used separately to visualize model focus areas on chest X-rays.
- Misclassified cases should be further analyzed using Grad-CAM outputs.

---

## Files in this project

- gradcam.py: Grad-CAM implementation and inference
- best_model.pth: trained ResNet18 model weights
- outputs/: Grad-CAM visualizations
- results.md: evaluation summary
