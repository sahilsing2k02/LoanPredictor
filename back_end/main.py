"""
Loan Default Prediction Backend API
Author: Sahil Singh
"""

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import lightgbm as lgb

MODEL_FILE = os.path.join(os.path.dirname(__file__), 'model.txt')
bst = lgb.Booster(model_file=MODEL_FILE)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return jsonify({'status': 'running', 'message': 'Loan Default Prediction API is live.'})

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
    app.run(debug=True, host='0.0.0.0', port=5000)
