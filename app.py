import numpy as np
from flask import Flask, request, render_template
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

app = Flask(__name__)

# Load the NearestNeighbors model
try:
    model = pickle.load(open('nearest_neighbors_model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

# Load the Universal Sentence Encoder
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed_model = hub.load(model_url)
print('Embedding model loaded')

def embed(texts):
    return embed_model(texts)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the single string input from the form
    input_string = request.form['input_string']
    
    # Make the prediction
    try:
        # Preprocess the input as needed by your model
        processed_input = preprocess_input(input_string)
        
        # Embed the input string
        embedded_input = embed([processed_input])
        
        # Find the nearest neighbors
        distances, indices = model.kneighbors(embedded_input)
        
        # Assuming df is loaded with destinations and descriptions
        df = pd.read_excel("Tourist places.xlsx")
        df = df[["Destination", "description"]].dropna().reset_index()
        
        # Get the recommended destinations
        recommended_destinations = df['Destination'].iloc[indices[0]].tolist()
        
        # Format the output
        output = recommended_destinations if recommended_destinations else ["No recommendations found"]
    except Exception as e:
        output = [f"Error making prediction: {e}"]
    
    # Render the template with the prediction result
    return render_template('index.html', prediction_text=output)

def preprocess_input(input_string):
    # Example preprocessing, modify as needed for your model
    return input_string

if __name__ == "__main__":
    app.run(debug=True)