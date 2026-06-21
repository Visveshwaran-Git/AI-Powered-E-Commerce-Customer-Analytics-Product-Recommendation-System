import shap
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def generate_shap_explanation(input_data):
    """
    Generates a SHAP waterfall plot for a specific prediction using the Random Forest model.
    """
    rf_path = 'models/random_forest.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if not os.path.exists(rf_path) or not os.path.exists(scaler_path):
        return None
        
    model = joblib.load(rf_path)
    scaler = joblib.load(scaler_path)
    
    # input_data is a dictionary, convert to dataframe
    # Expected keys: ['Age', 'Income', 'FavoriteCategory_Encoded', 'BrowsingTimeMins', 'PagesVisited', 'PastPurchases']
    df = pd.DataFrame([input_data])
    
    # Scale input
    scaled_input = scaler.transform(df)
    
    # Explainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)
    
    # For classification, shap_values is a list of arrays (one for each class). We want the positive class [1]
    # Sometimes it returns just an array, checking shape
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        sv = shap_values[0, :, 1] if len(shap_values.shape) > 2 else shap_values[0]
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    os.makedirs('static/charts', exist_ok=True)
    
    # Create matplotlib figure
    plt.figure(figsize=(10, 6))
    
    # Create the SHAP explanation object for waterfall plot
    exp = shap.Explanation(values=sv, 
                           base_values=base_value, 
                           data=df.iloc[0].values, 
                           feature_names=df.columns)
                           
    shap.waterfall_plot(exp, show=False)
    plt.tight_layout()
    
    # Save the plot
    output_path = 'static/charts/shap_explanation.png'
    plt.savefig(output_path, bbox_inches='tight', transparent=False)
    plt.close()
    
    return output_path
