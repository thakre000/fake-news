from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}
MODEL_FILE = 'TrainedModel/model.pkl'
VECTORIZER_FILE = 'TrainedModel/vectorizer.pkl'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Utility function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load and preprocess the data
            data = pd.read_csv(filepath)
            
            # Validate file structure
            if not all(col in data.columns for col in ['text', 'label']):
                return jsonify({'error': 'CSV file must contain "text" and "label" columns.'})

            # Drop rows with missing text
            data = data.dropna(subset=['text'])

            # Map labels to binary values
            data['label'] = data['label'].apply(lambda x: 1 if x.upper() == 'REAL' else 0)

            # Get label distribution
            label_dist = data['label'].value_counts().to_dict()

            # Split the data
            X = data['text']
            y = data['label']

            # Text vectorization
            vectorizer = TfidfVectorizer(max_features=5000)
            X_vectorized = vectorizer.fit_transform(X)

            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

            # Train the model
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Save the model and vectorizer
            joblib.dump(model, MODEL_FILE)
            joblib.dump(vectorizer, VECTORIZER_FILE)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            return jsonify({
                'message': 'Model trained successfully',
                'accuracy': accuracy,
                'label_distribution': label_dist
            })

        except Exception as e:
            return jsonify({'error': f'Error processing the file: {str(e)}'})

    else:
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'})

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form.get('news')
    if not news:
        return jsonify({'error': 'No news article provided.'})
    
    try:
        # Load the trained model and vectorizer
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)

        # Transform the input text
        news_vectorized = vectorizer.transform([news])

        # Make the prediction
        prediction = model.predict(news_vectorized)[0]
        prediction_label = 'Real' if prediction == 1 else 'Fake'

        return jsonify({'prediction': prediction_label})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
