---
# Animal Image Classification

A deep learning web application that classifies animal images into 15 categories using a fine-tuned EfficientNetB0 model. Built with TensorFlow and Streamlit, and deployable on Hugging Face Spaces.

---

## Overview

This project demonstrates image classification using transfer learning. Users can upload an image, and the app predicts the animal category with a confidence score.

---

## Features

- **Transfer Learning**: EfficientNetB0 model trained on a labeled animal image dataset.
- **Streamlit Web App**: Simple and intuitive interface for users to upload and classify images.
- **Ready for Deployment**: Easily deployable to Hugging Face Spaces.

---

## Supported Animal Classes

- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

---

## Project Structure

```
animal-image-classification/
├──Dataset                         # Folder containing 15 categories of animal images
├── app.py                         # Streamlit application
├── efficientnetb0_best.h5         # Pretrained model (EfficientNetB0)
├── mobilenetv2_best.h5            # Optional pretrained model
├── ml animal classification.ipynb # Training notebook
├── requirements.txt               # Dependencies
├── Image Classification of animals.pdf  # Project summary
└── README.md                      # Project documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hurmath123/animal-image-classifier.git
cd animal-image-classifier
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## Deploy to Hugging Face

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **Create new Space** → Choose **Streamlit** and **Python**
3. Upload:
   - `app.py`
   - `efficientnetb0_best.h5`
   - `requirements.txt`

4. The app will auto-deploy once all files are uploaded.

---

## 📁 Dataset

The model was trained on [Frough11/animal_classes](https://huggingface.co/datasets/Frough11/animal_classes), a folder-based image classification dataset.

---

## Author
Frough Hurmath S
**Hurmath123**  
[GitHub Profile](https://github.com/Hurmath123)

---

## License

This project is licensed under the [MIT License](LICENSE).
