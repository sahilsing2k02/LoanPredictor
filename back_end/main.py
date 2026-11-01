"""
Loan Default Prediction Backend API & Fullstack Server
Author: Sahil Singh
"""

import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

candidate_model_paths = [
    os.path.join(BASE_DIR, 'model.txt'),
    os.path.join(os.path.dirname(BASE_DIR), 'api', 'model.txt'),
    os.path.join(os.path.dirname(BASE_DIR), 'back_end', 'model.txt'),
    os.path.join(os.getcwd(), 'api', 'model.txt'),
    os.path.join(os.getcwd(), 'back_end', 'model.txt'),
    'model.txt'
]
MODEL_FILE = next((p for p in candidate_model_paths if os.path.exists(p)), candidate_model_paths[0])

candidate_frontend_paths = [
    os.path.abspath(os.path.join(BASE_DIR, '..', 'front_end')),
    os.path.abspath(os.path.join(os.getcwd(), 'front_end')),
    os.path.abspath(os.path.join(os.path.dirname(BASE_DIR), 'front_end'))
]
FRONTEND_DIR = next((p for p in candidate_frontend_paths if os.path.exists(p)), candidate_frontend_paths[0])

bst = lgb.Booster(model_file=MODEL_FILE)

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    index_file = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_file):
        return send_from_directory(FRONTEND_DIR, 'index.html')
    return jsonify({'status': 'running', 'message': 'Loan Default Prediction API is live.'})

@app.route('/<path:filename>', methods=['GET'])
def static_files(filename):
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({'error': 'Not found'}), 404

@app.route('/', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}

    def get_or_none(dictionary, key):
        val = dictionary.get(key)
        return val if val != '' else None

    Gender = get_or_none(data, 'Gender')
    Married = get_or_none(data, 'Married')
    Dependents = str(get_or_none(data, 'Dependents')) if get_or_none(data, 'Dependents') is not None else None
    Education = get_or_none(data, 'Education')
    Self_employed = get_or_none(data, 'Self_employed')

    def to_float(val):
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    ApplicantIncome = to_float(get_or_none(data, 'ApplicantIncome'))
    CoapplicantIncome = to_float(get_or_none(data, 'CoapplicantIncome'))
    LoanAmount = to_float(get_or_none(data, 'LoanAmount'))
    Loan_Amount_Term = to_float(get_or_none(data, 'Loan_Amount_Term'))
    Credit_History = to_float(get_or_none(data, 'Credit_History'))
    Property_Area = get_or_none(data, 'Property_Area')

    features = [[
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        int(Gender == 'Female'),
        int(Gender == 'Male'),
        int(Married == 'No'),
        int(Married == 'Yes'),
        int(Dependents == '0'),
        int(Dependents == '1'),
        int(Dependents == '2'),
        int(Dependents == '3+'),
        int(Education == 'Graduate'),
        int(Education in ('Not Graduate', 'Under Graduate')),
        int(Self_employed == 'No'),
        int(Self_employed == 'Yes'),
        int(Property_Area == 'Rural'),
        int(Property_Area == 'Semiurban'),
        int(Property_Area == 'Urban')
    ]]

    res = bst.predict(features)
    return jsonify({'result': float(res[0])})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1')
    app.run(debug=debug, host='0.0.0.0', port=port)
