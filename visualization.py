import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Use Agg backend to avoid GUI issues in server environments
import matplotlib
matplotlib.use('Agg')

def generate_visualizations():
    """
    Generates all requested data visualizations and saves them to static/charts/
    """
    os.makedirs('static/charts', exist_ok=True)
    
    data_path = 'data/customer_data.csv'
    if not os.path.exists(data_path):
        print("Data file not found. Run train_models.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Modern styling
    plt.style.use('dark_background') # Align with dark mode UI preference
    sns.set_palette("husl")
    
    # 1. Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=30, kde=True, color='cyan')
    plt.title('Customer Age Distribution', color='white', pad=15)
    plt.xlabel('Age', color='white')
    plt.ylabel('Count', color='white')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/charts/age_distribution.png', transparent=True)
    plt.close()
    
    # 2. Income Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Income'], bins=30, kde=True, color='magenta')
    plt.title('Customer Income Distribution', color='white', pad=15)
    plt.xlabel('Income', color='white')
    plt.ylabel('Count', color='white')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/charts/income_distribution.png', transparent=True)
    plt.close()
    
    # 3. Product Popularity (Category based on data)
    plt.figure(figsize=(8, 5))
    sns.countplot(y='FavoriteCategory', data=df, order=df['FavoriteCategory'].value_counts().index, palette='viridis')
    plt.title('Favorite Product Categories', color='white', pad=15)
    plt.xlabel('Count', color='white')
    plt.ylabel('Category', color='white')
    plt.tight_layout()
    plt.savefig('static/charts/product_popularity.png', transparent=True)
    plt.close()
    
    # 4. Purchase Trends (Purchased vs Browsing Time)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Purchased', y='BrowsingTimeMins', data=df, palette='Set2')
    plt.title('Browsing Time vs Purchase Decision (0=No, 1=Yes)', color='white', pad=15)
    plt.xlabel('Purchased', color='white')
    plt.ylabel('Browsing Time (mins)', color='white')
    plt.tight_layout()
    plt.savefig('static/charts/purchase_trends.png', transparent=True)
    plt.close()
    
    # 5. Model Accuracy Comparison
    acc_path = 'data/model_accuracies.csv'
    if os.path.exists(acc_path):
        acc_df = pd.read_csv(acc_path)
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Model', y='Accuracy', data=acc_df, palette='coolwarm')
        plt.title('Model Accuracy Comparison', color='white', pad=15)
        plt.ylim(0, 1.0)
        for index, row in acc_df.iterrows():
            plt.text(index, row.Accuracy + 0.02, round(row.Accuracy, 4), color='white', ha="center")
        plt.tight_layout()
        plt.savefig('static/charts/model_comparison.png', transparent=True)
        plt.close()
        
    # 6. Feature Importance (Random Forest)
    rf_path = 'models/random_forest.pkl'
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        features = ['Age', 'Income', 'FavoriteCategory_Encoded', 'BrowsingTimeMins', 'PagesVisited', 'PastPurchases']
        importances = rf_model.feature_importances_
        
        fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        fi_df = fi_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='plasma')
        plt.title('Feature Importance (Random Forest)', color='white', pad=15)
        plt.tight_layout()
        plt.savefig('static/charts/feature_importance.png', transparent=True)
        plt.close()

    # 7. TF Model Training History
    tf_history_path = 'data/tf_history.csv'
    if os.path.exists(tf_history_path):
        hist_df = pd.read_csv(tf_history_path)
        plt.figure(figsize=(8, 5))
        plt.plot(hist_df['accuracy'], label='Train Accuracy', color='cyan')
        plt.plot(hist_df['val_accuracy'], label='Validation Accuracy', color='magenta')
        plt.title('TensorFlow Model Accuracy over Epochs', color='white', pad=15)
        plt.xlabel('Epoch', color='white')
        plt.ylabel('Accuracy', color='white')
        plt.legend()
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        plt.savefig('static/charts/tf_accuracy.png', transparent=True)
        plt.close()

if __name__ == "__main__":
    generate_visualizations()
    print("Visualizations generated and saved to static/charts/")
