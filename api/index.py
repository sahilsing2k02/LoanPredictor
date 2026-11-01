"""
CrediAI - Vercel Serverless Function & Local API Entrypoint
Author: Sahil Singh
"""

import os
import sys

# Ensure local api folder and root are on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

for p in (CURRENT_DIR, ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from . import predictor_pure
except (ImportError, ValueError):
    try:
        import predictor_pure
    except ImportError:
        from back_end import predictor_pure

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Resolve front_end directory candidates
candidate_dirs = [
    os.path.join(ROOT_DIR, 'front_end'),
    os.path.join(CURRENT_DIR, 'front_end'),
    os.path.join(ROOT_DIR, 'public'),
    os.path.join(os.getcwd(), 'front_end'),
    os.path.join(os.getcwd(), 'public')
]
FRONTEND_DIR = next((d for d in candidate_dirs if os.path.exists(d)), candidate_dirs[0])

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home():
    index_file = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_file):
        return send_from_directory(FRONTEND_DIR, 'index.html')
    return jsonify({
        'status': 'running',
        'service': 'CrediAI ML Engine (Vercel Serverless)',
        'author': 'Sahil Singh'
    })

@app.route('/api', methods=['GET'])
@app.route('/api/', methods=['GET'])
@app.route('/status', methods=['GET'])
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'running',
        'service': 'CrediAI ML Engine (Vercel Serverless)',
        'author': 'Sahil Singh'
    })

@app.route('/<path:filename>', methods=['GET'])
def static_files(filename):
    if filename in ('api', 'api/', 'status', 'api/status'):
        return status()
    file_path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(FRONTEND_DIR, filename)
    index_file = os.path.join(FRONTEND_DIR, 'index.html')
    if os.path.exists(index_file):
        return send_from_directory(FRONTEND_DIR, 'index.html')
    return jsonify({'error': f'Not found: {filename}'}), 404

@app.route('/', methods=['POST'])
@app.route('/api', methods=['POST'])
@app.route('/api/', methods=['POST'])
@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
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
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    ApplicantIncome = to_float(get_or_none(data, 'ApplicantIncome'))
    CoapplicantIncome = to_float(get_or_none(data, 'CoapplicantIncome'))
    LoanAmount = to_float(get_or_none(data, 'LoanAmount'))
    Loan_Amount_Term = to_float(get_or_none(data, 'Loan_Amount_Term'))
    Credit_History = to_float(get_or_none(data, 'Credit_History'))
    Property_Area = get_or_none(data, 'Property_Area')

    features = [
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
    ]

    res = predictor_pure.score(features)
    prob = float(res[1] if isinstance(res, (list, tuple)) else res)
    return jsonify({'result': prob})
