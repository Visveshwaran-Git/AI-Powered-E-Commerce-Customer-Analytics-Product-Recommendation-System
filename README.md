<h1 align="center">
  🤖 AI-Powered E-Commerce Insights & Recommendation System
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Flask-3.0.3-black.svg" alt="Flask">
  <img src="https://img.shields.io/badge/TensorFlow-2.16-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4.2-yellow.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/UI-Bootstrap%205-purple.svg" alt="Bootstrap 5">
</p>

<p align="center">
  A complete, end-to-end Machine Learning web application designed to predict customer purchasing behavior and recommend products using Collaborative and Content-Based Filtering.
</p>

---

## 🌟 Features

* **Purchase Prediction Engine:** Predicts the probability of a user making a purchase based on their age, income, and browsing habits.
* **Multi-Model Comparison:** Trains and compares Logistic Regression, Random Forest, and Deep Learning (TensorFlow/Keras) models.
* **Explainable AI (XAI):** Integrated with SHAP (SHapley Additive exPlanations) to provide waterfall plots explaining exactly *why* a customer is predicted to buy (or not buy).
* **Dual Recommendation System:** 
  * **Content-Based Filtering** (TF-IDF) to recommend similar products.
  * **Collaborative Filtering** to recommend products based on similar user behaviors.
* **Analytics Dashboard:** Automatically generates and displays data visualizations (Seaborn/Matplotlib) for age, income distributions, and feature importance.
* **Modern Web Interface:** Fully responsive UI built with Bootstrap 5, featuring a sleek Dark Mode and glassmorphism design.
* **Secure Authentication:** User registration and login utilizing Flask-Login and Bcrypt.

## 🛠️ Technology Stack

* **Backend & Web Framework:** Python, Flask, Flask-SQLAlchemy, SQLite
* **Machine Learning & AI:** TensorFlow/Keras, Scikit-Learn, Pandas, NumPy, SHAP
* **Data Visualization:** Matplotlib, Seaborn
* **Frontend:** HTML5, Vanilla CSS, Bootstrap 5
* **Deployment:** Docker, Gunicorn, Render (`render.yaml`)

## 📂 Project Structure

```text
Ecommerce/
│
├── data/                      # Synthetic datasets and training histories
│   ├── customer_data.csv      # Auto-generated customer interaction data
│   └── product_catalog.csv    # Auto-generated e-commerce product catalog
│
├── models/                    # Saved ML Models and Preprocessors
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── tensorflow_model.h5
│   └── scaler.pkl             # StandardScaler for data normalization
│
├── static/
│   ├── css/style.css          # Custom dark mode UI styling
│   └── charts/                # Auto-generated visualization plots
│
├── templates/                 # Flask HTML templates
│   ├── base.html              
│   ├── dashboard.html         # Analytics dashboard
│   ├── prediction.html        # ML Prediction UI with SHAP plots
│   └── recommendation.html    # Product Recommendation UI
│
├── app.py                     # Main Flask Application
├── data_generator.py          # Script to generate mock data if none exists
├── db.py                      # Database models (User Auth)
├── explainability.py          # SHAP integration for ML interpretability
├── recommendation.py          # Recommendation engine logic
├── train_models.py            # ML Pipeline: Preprocess, Train, Evaluate, Save
├── visualization.py           # Generates Seaborn/Matplotlib charts
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Containerization instructions
└── render.yaml                # Infrastructure as Code for Render deployment
```

## 🚀 Local Setup & Installation

### Prerequisites
* Python 3.9+
* Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ecommerce-ai-system.git
   cd ecommerce-ai-system
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Data and Train Models**
   Run the training script. This will automatically generate the synthetic `customer_data.csv`, preprocess the data, train the Machine Learning models, and generate the data visualizations.
   ```bash
   python train_models.py
   python visualization.py
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the App**
   Open your browser and navigate to `http://localhost:5000`

## 🐳 Docker Deployment

You can run the entire application, including the training pipeline, inside a Docker container:

```bash
# Build the image
docker build -t ecommerce-ai .

# Run the container
docker run -p 5000:5000 ecommerce-ai
```

## 🌐 Cloud Deployment (Render / Railway)

This repository is configured for easy deployment on [Render.com](https://render.com). 
1. Connect your GitHub account to Render.
2. Click **New > Blueprint**.
3. Select this repository. The `render.yaml` file will automatically configure the Python environment, install dependencies, train the models, and launch the Gunicorn server.

## 📸 Screenshots

*(Add screenshots of your Dashboard, Prediction Page, and Recommendation Page here)*

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

#Project Scrrenshots
<img width="1897" height="852" alt="Screenshot 2026-06-21 124838" src="https://github.com/user-attachments/assets/0658874c-5ff8-4dc3-bfe6-57ba902dc0bf" />
<img width="1868" height="766" alt="Screenshot 2026-06-21 125015" src="https://github.com/user-attachments/assets/bad14e89-cb59-4feb-8f8e-17bcba34355d" />
<img width="1423" height="846" alt="Screenshot 2026-06-21 125035" src="https://github.com/user-attachments/assets/d5b8f06c-5d17-401f-a7ec-a7bcf6286b86" />
<img width="1476" height="821" alt="Screenshot 2026-06-21 125047" src="https://github.com/user-attachments/assets/f64c4419-f0b0-4afe-82c4-f4ddb3812cf0" />
<img width="1860" height="808" alt="Screenshot 2026-06-21 125121" src="https://github.com/user-attachments/assets/fbcd583f-b3ae-4acc-8e36-c4b9bbee31ce" />





