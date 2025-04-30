from flask import Flask, Blueprint, request, jsonify, render_template, send_file
from flask_cors import CORS
import traceback
import os
import numpy as np
import tempfile
from werkzeug.utils import secure_filename

# Import your recommender code
from research import JournalRecommender
from recommender_with_report import EnhancedJournalRecommender, ReportVisualizer, ReportGenerator

# Import chatbot routes
from chatbot import register_chatbot_routes, chatbot_api

# Create a Blueprint for the recommender API
recommender_api = Blueprint('recommender_api', __name__)

# Initialize recommenders
basic_recommender = JournalRecommender()
enhanced_recommender = EnhancedJournalRecommender()

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

@recommender_api.route('/api/recommend-journals', methods=['POST'])
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
        
        # Get recommendations using the basic recommender
        recommendations = basic_recommender.recommend(title, abstract, top_k=top_k)
        
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

@recommender_api.route('/api/generate-report', methods=['POST'])
def generate_report():
    """API endpoint to generate a detailed PDF report with journal recommendations"""
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
        
        # Create a temporary file for the PDF
        temp_dir = tempfile.gettempdir()
        safe_title = secure_filename(title)
        filename = f"ResearchSaathi_Report_{safe_title[:30]}.pdf"
        filepath = os.path.join(temp_dir, filename)
        
        print(f"Generating report to be saved at: {filepath}")
        
        # Use the enhanced recommender to generate the report with explicit output path
        journals = enhanced_recommender.generate_report(
            title=title, 
            abstract=abstract, 
            top_k=top_k,
            output_path=filepath
        )
        
        # Verify the file exists
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'Report generation failed. The file was not created.'
            }), 500
            
        print(f"Report file exists: {os.path.exists(filepath)}")
        print(f"File size: {os.path.getsize(filepath)} bytes")
        
        # Return file download link
        return jsonify({
            'success': True,
            'message': 'Report generated successfully',
            'download_url': f'/api/download-report/{filename}'
        })
        
    except Exception as e:
        # Log the error
        print(f"Error in report generation: {str(e)}")
        print(traceback.format_exc())
        
        # Return error response
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }), 500

@recommender_api.route('/api/download-report/<filename>', methods=['GET'])
def download_report(filename):
    """Download the generated PDF report"""
    try:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, secure_filename(filename))
        
        print(f"Attempting to download file from: {filepath}")  # Debug log
        print(f"File exists: {os.path.exists(filepath)}")  # Debug check
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'Report not found. It may have expired or been removed.'
            }), 404
        
        return send_file(
            filepath,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"Error downloading report: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }), 500

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(recommender_api)
    register_chatbot_routes(app)  # Register chatbot routes
    
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)