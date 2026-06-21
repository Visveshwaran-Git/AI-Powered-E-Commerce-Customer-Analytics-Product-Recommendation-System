import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from data_generator import generate_product_catalog, generate_user_interactions

class RecommendationSystem:
    def __init__(self):
        self.catalog_path = 'data/product_catalog.csv'
        self.interactions_path = 'data/user_interactions.csv'
        
        # Ensure data exists
        if not os.path.exists(self.catalog_path):
            generate_product_catalog(self.catalog_path)
        if not os.path.exists(self.interactions_path):
            generate_user_interactions(self.interactions_path)
            
        self.products_df = pd.read_csv(self.catalog_path)
        self.interactions_df = pd.read_csv(self.interactions_path)
        
        # Compute similarities
        self._compute_content_similarity()

    def _compute_content_similarity(self):
        """
        Creates a TF-IDF matrix based on product name and category, 
        then computes cosine similarity.
        """
        self.products_df['combined_features'] = self.products_df['Category'] + " " + self.products_df['Name']
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.products_df['combined_features'])
        self.content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def content_based_recommendation(self, product_id, top_n=5):
        """
        Recommend products similar to a given product ID based on content.
        """
        try:
            # Find index of product
            idx = self.products_df.index[self.products_df['ProductID'] == product_id].tolist()[0]
        except IndexError:
            return [] # Product not found
            
        sim_scores = list(enumerate(self.content_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip the first one (itself)
        sim_scores = sim_scores[1:top_n+1]
        
        product_indices = [i[0] for i in sim_scores]
        recommendations = self.products_df.iloc[product_indices][['ProductID', 'Name', 'Category', 'Price']].to_dict('records')
        return recommendations

    def collaborative_filtering_recommendation(self, user_id, top_n=5):
        """
        Recommend products based on similar users (Item-based collaborative approach using pivot table)
        """
        # Create user-item matrix
        user_item_matrix = self.interactions_df.pivot_table(
            index='CustomerID', 
            columns='ProductID', 
            values='InteractionScore'
        ).fillna(0)
        
        if user_id not in user_item_matrix.index:
            # Fallback to popular products if user has no history
            return self.get_popular_products(top_n)
            
        # Compute user similarity (cosine similarity between users)
        user_sim = cosine_similarity(user_item_matrix)
        user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)
        
        # Get similar users
        similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:6] # Top 5 similar users
        
        # Get products bought/interacted by similar users
        similar_users_interactions = user_item_matrix.loc[similar_users]
        
        # Calculate mean score for each product across similar users
        product_scores = similar_users_interactions.mean().sort_values(ascending=False)
        
        # Filter out products the user has already interacted with
        user_interactions = user_item_matrix.loc[user_id]
        already_interacted = user_interactions[user_interactions > 0].index
        
        recommendations = product_scores.drop(already_interacted, errors='ignore').head(top_n).index
        
        # Fetch product details
        rec_details = self.products_df[self.products_df['ProductID'].isin(recommendations)][['ProductID', 'Name', 'Category', 'Price']].to_dict('records')
        return rec_details

    def get_popular_products(self, top_n=5):
        """
        Returns top N popular products based on average rating
        """
        popular = self.products_df.sort_values(by='Rating', ascending=False).head(top_n)
        return popular[['ProductID', 'Name', 'Category', 'Price']].to_dict('records')
