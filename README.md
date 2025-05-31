# 🦠 Malaria Detection Web App

This is a Flask-based web application that uses a trained deep learning model to detect malaria parasites in blood smear images.

🔗 **Live App:** [malaria-detectio-ba2fff198a09.herokuapp.com](https://malaria-detectio-ba2fff198a09.herokuapp.com/)  
📂 **Dataset Used:** [Malaria Detection Dataset on Kaggle](https://www.kaggle.com/datasets/shahriar26s/malaria-detection/data)

---

## 💡 Features

- Upload a blood smear image and get real-time predictions.
- Detects whether malaria parasites are present or not.
- Deep learning model built with PyTorch (ResNet18 architecture).
- Clean, responsive user interface.
- Deployed on Heroku.

---

## 🧠 Model Summary

- **Architecture:** ResNet18 (pretrained on ImageNet)
- **Training Framework:** PyTorch
- **Output Labels:**
  - `Malaria parasite detected`
  - `No malaria parasite found`
- **Dataset Source:** [Shahriar26s on Kaggle](https://www.kaggle.com/datasets/shahriar26s/malaria-detection/data)
- **Input Shape:** 224x224 RGB images

You can find the full training pipeline in [`Malaria Detection.ipynb`](./model_training.ipynb).

---

## 🛠 Tech Stack

- **Backend:** Python, Flask
- **ML Model:** PyTorch, torchvision
- **Deployment:** Heroku
- **Frontend:** HTML (Jinja2 templating)

---

## 🚀 How to Run Locally

### 1. Clone this Repository

```bash
git clone https://github.com/your-username/malaria-detection-app.git
cd malaria-detection-app
````

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App Locally

```bash
# Set the Flask app
# Windows
set FLASK_APP=app.py

# macOS/Linux
export FLASK_APP=app.py

# Then run
flask run
```

Open your browser and go to:
**[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📁 Project Structure

```
malaria-detection-app/
├── app.py                   # Main Flask application
├── malaria_resnet18.pth     # Trained PyTorch model
├── model_training.ipynb     # Jupyter notebook for training the model
├── requirements.txt         # Python dependencies
├── Procfile                 # Heroku deployment config
├── .python-version          # Python version specification
├── templates/
│   └── index.html           # Frontend HTML
└── README.md
```

---

## 📓 `model_training.ipynb`

The notebook includes:

* Loading and preprocessing the Kaggle dataset.
* Building and training a transfer learning model using **ResNet18**.
* Evaluation on validation data.
* Saving the final trained model (`malaria_resnet18.pth`) for deployment.

---

## 🔎 Labels and Prediction

* **Labels Used in Classification:**

  * `Malaria parasite detected`
  * `No malaria parasite found`

The model outputs a binary classification based on the blood smear image uploaded by the user.

---

## 🙏 Acknowledgements

* Dataset: [Malaria Detection | Kaggle](https://www.kaggle.com/datasets/shahriar26s/malaria-detection/data)
* Model Architecture: [ResNet18 - torchvision](https://pytorch.org/vision/stable/models.html)
* Deployment Platform: [Heroku](https://www.heroku.com)

---

## 📬 Contact

Feel free to raise issues or contribute by submitting a pull request.
Star ⭐ the repo if you find it useful!

```

 
