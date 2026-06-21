from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from db import db, User
from recommendation import RecommendationSystem
from explainability import generate_shap_explanation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ecommerce_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ecommerce.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Models (Lazy loading)
rf_model = None
lr_model = None
tf_model = None
scaler = None
label_encoder = None
rec_system = None

def load_models():
    global rf_model, lr_model, tf_model, scaler, label_encoder, rec_system
    if os.path.exists('models/random_forest.pkl'):
        rf_model = joblib.load('models/random_forest.pkl')
    if os.path.exists('models/logistic_regression.pkl'):
        lr_model = joblib.load('models/logistic_regression.pkl')
    if os.path.exists('models/tensorflow_model.h5'):
        tf_model = tf.keras.models.load_model('models/tensorflow_model.h5')
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
    if os.path.exists('models/label_encoder.pkl'):
        label_encoder = joblib.load('models/label_encoder.pkl')
    if os.path.exists('data/product_catalog.csv'):
        rec_system = RecommendationSystem()

with app.app_context():
    db.create_all()
    load_models()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Check if user exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'warning')
            return redirect(url_for('register'))
            
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    prediction_result = None
    shap_image = None
    input_data_dict = {}
    
    if request.method == 'POST':
        if not rf_model or not scaler or not label_encoder:
            flash('Models not loaded. Please train models first.', 'danger')
            return redirect(url_for('prediction'))
            
        try:
            age = float(request.form['Age'])
            income = float(request.form['Income'])
            category = request.form['FavoriteCategory']
            browsing_time = float(request.form['BrowsingTimeMins'])
            pages_visited = int(request.form['PagesVisited'])
            past_purchases = int(request.form['PastPurchases'])
            selected_model = request.form['ModelSelection']
            
            # Encode category
            try:
                category_encoded = label_encoder.transform([category])[0]
            except ValueError:
                category_encoded = 0 # default if unknown
                
            input_data_dict = {
                'Age': age,
                'Income': income,
                'FavoriteCategory_Encoded': category_encoded,
                'BrowsingTimeMins': browsing_time,
                'PagesVisited': pages_visited,
                'PastPurchases': past_purchases
            }
            
            # Scale input
            df_input = pd.DataFrame([input_data_dict])
            scaled_input = scaler.transform(df_input)
            
            # Predict based on selected model
            if selected_model == 'rf':
                prob = rf_model.predict_proba(scaled_input)[0][1]
            elif selected_model == 'lr':
                prob = lr_model.predict_proba(scaled_input)[0][1]
            elif selected_model == 'tf':
                prob = tf_model.predict(scaled_input)[0][0]
            else:
                prob = rf_model.predict_proba(scaled_input)[0][1]
                
            prediction_result = {
                'probability': round(prob * 100, 2),
                'will_purchase': prob > 0.5,
                'model_used': selected_model.upper()
            }
            
            # Generate SHAP explanation using RF model
            shap_image = generate_shap_explanation(input_data_dict)
            if shap_image:
                shap_image = 'charts/shap_explanation.png' # Relative to static
                
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'danger')

    return render_template('prediction.html', prediction_result=prediction_result, shap_image=shap_image, input_data=input_data_dict)

@app.route('/export_prediction')
@login_required
def export_prediction():
    # Simple endpoint to export last made prediction if needed (for simplicity we just export a template CSV here)
    data = request.args.get('data', 'No data')
    csv_content = f"Age,Income,BrowsingTime,PagesVisited,PastPurchases,Probability\n{data}"
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=prediction.csv"}
    )

@app.route('/recommendation', methods=['GET', 'POST'])
@login_required
def recommendation():
    content_recs = []
    collab_recs = []
    
    if request.method == 'POST':
        if not rec_system:
            flash('Recommendation system not initialized.', 'danger')
            return redirect(url_for('recommendation'))
            
        rec_type = request.form.get('rec_type')
        if rec_type == 'content':
            product_id = int(request.form.get('product_id', 101))
            content_recs = rec_system.content_based_recommendation(product_id)
        elif rec_type == 'collab':
            # In a real app we'd use current_user.id, here we use random or input
            user_id = int(request.form.get('user_id', current_user.id))
            collab_recs = rec_system.collaborative_filtering_recommendation(user_id)
            
    # Load some popular products for display
    popular = rec_system.get_popular_products() if rec_system else []
            
    return render_template('recommendation.html', content_recs=content_recs, collab_recs=collab_recs, popular=popular)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
