"""
CrediAI - Vercel Serverless Prediction API
Author: Sahil Singh
"""

import os
import sys

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

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def handler(path):
    if request.method == 'GET':
        return jsonify({
            'status': 'running',
            'service': 'CrediAI Loan Prediction API',
            'author': 'Sahil Singh'
        })

    data = request.get_json(silent=True) or {}

    def val(key):
        v = data.get(key)
        return v if v != '' else None

    def num(key):
        try:
            v = val(key)
            return float(v) if v is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    Gender = val('Gender')
    Married = val('Married')
    Dependents = str(val('Dependents')) if val('Dependents') is not None else None
    Education = val('Education')
    Self_employed = val('Self_employed')
    Property_Area = val('Property_Area')

    features = [
        num('ApplicantIncome'),
        num('CoapplicantIncome'),
        num('LoanAmount'),
        num('Loan_Amount_Term'),
        num('Credit_History'),
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
