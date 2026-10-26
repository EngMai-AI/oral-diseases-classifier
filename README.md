# 🦷 Oral Diseases Image Classification

<img width="1536" height="1024" alt="WhatsApp Image 2026-10-26 at 4 16 52 PM" src="https://github.com/user-attachments/assets/3d795789-878b-452b-b572-907f52914431" />

End-to-end deep learning project that classifies oral images into 5 conditions:
**Calculus, Gingivitis, Hypodontia, Mouth ulcer, Tooth discoloration**, with a Gradio web app.

> ⚠️ Educational project only. Not a medical diagnosis tool.

## Pipeline
1. **Data audit & cleaning:** removed non-disease folders and 2,022 duplicate/conflicting images (13,862 → 3,322 usable images for the 5 classes).
2. **Custom CNN** trained from scratch.
3. **Transfer learning:** MobileNetV2, EfficientNetB0, DenseNet121 (frozen head + fine-tuning).
4. **Hyperparameter tuning:** random search (lr, dropout, units, fine-tuned layers).
5. **Comparative analysis** and per-class error analysis.
6. **Deployment** with Gradio.

## Results (test set, 499 images)
| Model | Accuracy | Macro-F1 | Params |
|---|---|---|---|
| **MobileNetV2** | **95.6%** | **0.932** | 2.59M |
| MobileNetV2 (tuned) | 94.6% | 0.921 | 2.92M |
| DenseNet121 | 94.4% | 0.920 | 7.30M |
| EfficientNetB0 | 94.0% | 0.896 | 4.38M |
| Custom CNN | 68.1% | 0.639 | 1.24M |

## Run the app
```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:7860

## Limitations
- Small, imbalanced dataset (173 to 1,239 images per class).
- Only exact duplicates were removed; flipped/rotated copies may remain across train/test, which can inflate results.
- Tested on a few external images, the model made mistakes with high confidence, so confidence is not a reliable correctness signal.
- The model has no "not teeth" class and always picks one of the 5 conditions.

## Dataset
Oral diseases dataset on Kaggle (salmansajid05). Images are not included in this repo.

## Screenshots
<img width="1091" height="630" alt="image" src="https://github.com/user-attachments/assets/f666bf46-2bb8-4e33-8d23-b70609e62f8a" />
