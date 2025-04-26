# Import necessary Flask modules
from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
import traceback
import os
import numpy as np

# Import your recommender code
from research import JournalRecommender

# Create a Blueprint for the recommender API
recommender_api = Blueprint('recommender_api', __name__)

# Initialize recommender
recommender = JournalRecommender()

# Helper function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj

@recommender_api.route('/api/recommend-journals', methods=['POST', 'GET'])
def recommend_journals():
    """API endpoint to recommend journals based on paper details"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'title' not in data or 'abstract' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters (title, abstract)'
            }), 400
        
        # Extract parameters
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        top_k = data.get('top_k', 5)
        
        # Validate parameters
        if not title or not abstract:
            return jsonify({
                'success': False,
                'error': 'Title and abstract cannot be empty'
            }), 400
        
        # Get recommendations using the recommender
        recommendations = recommender.recommend(title, abstract, top_k=top_k)
        
        # Convert NumPy types to Python native types for JSON serialization
        serializable_recommendations = convert_numpy_types(recommendations)
        
        # Return recommendations as JSON
        return jsonify({
            'success': True,
            'recommendations': serializable_recommendations
        })
        
    except Exception as e:
        # Log the error
        print(f"Error in journal recommendation: {str(e)}")
        print(traceback.format_exc())
        
        # Return error response
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }), 500


# How to register this blueprint in your main Flask app:

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(recommender_api)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 