"""
Embedding Service - Flask API for generating embeddings
"""
from flask import Flask, request, jsonify
from mistralai import Mistral
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize Mistral client
api_key = os.getenv('MISTRAL_API_KEY')
if not api_key:
    print("Warning: MISTRAL_API_KEY not found in environment variables")

client = Mistral(api_key=api_key)

# Configuration
EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIMENSION = 1024

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION
    })

@app.route('/embed', methods=['POST'])
def create_embedding():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        if not text or not text.strip():
            return jsonify({"error": "Text cannot be empty"}), 400
        
        # Mistral API call
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=[text]  # Mistral expects a list
        )
        
        embedding = response.data[0].embedding
        
        return jsonify({
            "embedding": embedding,
            "dimension": len(embedding),
            "model": EMBEDDING_MODEL
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/embed/batch', methods=['POST'])
def create_embeddings_batch():
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400
        
        texts = data['texts']
        
        # Mistral API call
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        
        return jsonify({
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "model": EMBEDDING_MODEL
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)