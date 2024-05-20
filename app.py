from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename

# Load the trained model
model = joblib.load('model.pkl')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title')
    abstract = request.form.get('abstract')
    
    if not title or not abstract:
        return jsonify({'error': 'Title and abstract are required'}), 400
    
    text = title + ' ' + abstract
    prediction = model.predict([text])[0]
    
    topics = {
        'Computer Science': bool(prediction[0]),
        'Physics': bool(prediction[1]),
        'Mathematics': bool(prediction[2]),
        'Statistics': bool(prediction[3]),
        'Quantitative Biology': bool(prediction[4]),
        'Quantitative Finance': bool(prediction[5])
    }
    
    return render_template('result.html', topics=topics)

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        data = pd.read_csv(filepath)
        if 'TITLE' not in data.columns or 'ABSTRACT' not in data.columns:
            return jsonify({'error': 'CSV file must contain TITLE and ABSTRACT columns'}), 400
        
        data['TEXT'] = data['TITLE'] + ' ' + data['ABSTRACT']
        predictions = model.predict(data['TEXT'])
        
        results = []
        for i, prediction in enumerate(predictions):
            topics = {
                'Computer Science': bool(prediction[0]),
                'Physics': bool(prediction[1]),
                'Mathematics': bool(prediction[2]),
                'Statistics': bool(prediction[3]),
                'Quantitative Biology': bool(prediction[4]),
                'Quantitative Finance': bool(prediction[5])
            }
            results.append({
                'title': data['TITLE'][i],
                'abstract': data['ABSTRACT'][i],
                'topics': topics
            })
        
        return render_template('bulk_result.html', results=results)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

