# 🧠 CNN Emotion Classifier (FER2013)

This project implements a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify human facial emotions using the FER2013 dataset. It includes preprocessing, model training, evaluation, and prediction on custom images.

---

## 📁 Project Structure

```
CNN/
├── data/
│   ├── train/           # Training images (by emotion)
│   ├── test/            # Testing images (by emotion)
│   └── random/          # Custom images for manual testing
├── models/
│   └── ImageClassification.keras   # Saved model
├── logs/                # TensorBoard logs
├── image_classification.ipynb      # Main notebook
└── README.md
```

---

## 😄 Emotion Categories

The model classifies faces into 7 categories:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## ⚙️ How to Use

### 1. Clone the repository

```bash
git clone https://github.com/AymenOML/cnn-image-classifier.git
cd cnn-image-classifier
```

### 2. Setup virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you're using a Mac with Apple Silicon:
```bash
pip install tensorflow-macos tensorflow-metal
```

### 3. Prepare the data

Download the dataset using:
```python
import kagglehub
kagglehub.dataset_download("msambare/fer2013")
```

Or manually place the folders `train/` and `test/` in the `data/` directory.

### 4. Run the notebook

Open `image_classification.ipynb` and run all cells:
- Load & preprocess the data
- Build the CNN model
- Train with callbacks (EarlyStopping, TensorBoard, ModelCheckpoint)
- Evaluate precision, recall, accuracy
- Predict custom images

---

## 📈 Sample Evaluation

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.52   |
| Precision  | 0.86   |
| Recall     | 0.74   |

> Note: Precision and recall are high despite lower accuracy due to class imbalance. Use confusion matrix for deeper analysis.

---

## 🔍 Predicting Custom Images

Place your image in `data/random/` (e.g., `sadtest.jpg`) and use the pipeline to:
- Convert to grayscale
- Resize to 48×48
- Normalize and reshape
- Predict and display label

---

## 📊 TensorBoard Support

To visualize training:

```bash
tensorboard --logdir=logs
```

---

## 💾 Save & Reload Model

```python
model.save('models/ImageClassification.keras')
model = load_model('models/ImageClassification.keras')
```

---

## 📦 Requirements

- Python 3.9+
- TensorFlow 2.x
- numpy, matplotlib, opencv-python
- kagglehub (for dataset download)

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Aymen Oumali**  
GitHub: [@AymenOML](https://github.com/AymenOML)

---

## ✅ To Do

- [ ] Add confusion matrix
- [ ] Visualize per-class accuracy
- [ ] Add webcam/live prediction mode
