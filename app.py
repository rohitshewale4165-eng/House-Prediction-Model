from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Generate synthetic training data ────────────────────────────────────────
np.random.seed(42)
n = 1500

area         = np.random.randint(500, 5000, n)
bedrooms     = np.random.randint(1, 7, n)
bathrooms    = np.random.randint(1, 5, n)
floors       = np.random.randint(1, 4, n)
age          = np.random.randint(0, 50, n)
garage       = np.random.randint(0, 4, n)
location_num = np.random.randint(0, 5, n)   # 0=rural … 4=prime
condition    = np.random.randint(1, 6, n)   # 1=poor … 5=excellent

# realistic price formula
price = (
    area          * 120
    + bedrooms    * 15_000
    + bathrooms   * 20_000
    + floors      * 10_000
    - age         * 2_000
    + garage      * 12_000
    + location_num* 40_000
    + condition   * 18_000
    + np.random.normal(0, 25_000, n)
)
price = np.maximum(price, 50_000)

df = pd.DataFrame({
    'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
    'floors': floors, 'age': age, 'garage': garage,
    'location': location_num, 'condition': condition, 'price': price
})

# ─── Feature engineering ─────────────────────────────────────────────────────
def engineer(df):
    d = df.copy()
    d['price_per_sqft_proxy'] = d['area'] / (d['bedrooms'] + 1)
    d['total_rooms']          = d['bedrooms'] + d['bathrooms']
    d['age_condition']        = d['age'] * (6 - d['condition'])
    d['luxury_score']         = d['location'] * d['condition']
    return d

df = engineer(df)
FEATURES = ['area','bedrooms','bathrooms','floors','age','garage',
            'location','condition','price_per_sqft_proxy',
            'total_rooms','age_condition','luxury_score']

X, y = df[FEATURES], df['price']

# ─── Train pipeline ──────────────────────────────────────────────────────────
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05,
        max_depth=5, min_samples_split=5,
        subsample=0.8, random_state=42
    ))
])
model_pipeline.fit(X, y)

# Cross-validation score
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {cv_scores}")
print(f"Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

LOCATION_MAP = {0:'Rural',1:'Suburban',2:'Town',3:'City',4:'Prime'}
CONDITION_MAP = {1:'Poor',2:'Fair',3:'Good',4:'Very Good',5:'Excellent'}

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        row = pd.DataFrame([{
            'area':      float(data['area']),
            'bedrooms':  int(data['bedrooms']),
            'bathrooms': int(data['bathrooms']),
            'floors':    int(data['floors']),
            'age':       int(data['age']),
            'garage':    int(data['garage']),
            'location':  int(data['location']),
            'condition': int(data['condition']),
        }])
        row = engineer(row)
        pred = model_pipeline.predict(row[FEATURES])[0]

        # confidence band ± 8 %
        low  = pred * 0.92
        high = pred * 1.08

        # feature importance
        importances = model_pipeline.named_steps['model'].feature_importances_
        fi = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)

        return jsonify({
            'predicted_price': round(pred, 2),
            'price_range': {'low': round(low,2), 'high': round(high,2)},
            'model_accuracy': round(cv_scores.mean() * 100, 1),
            'feature_importance': [{'feature':f,'importance':round(float(i),4)} for f,i in fi[:6]],
            'location_name':  LOCATION_MAP.get(int(data['location']),'N/A'),
            'condition_name': CONDITION_MAP.get(int(data['condition']),'N/A'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/model-info')
def model_info():
    return jsonify({
        'algorithm': 'Gradient Boosting Regressor',
        'n_estimators': 300,
        'cv_r2_mean': round(float(cv_scores.mean()), 4),
        'cv_r2_std':  round(float(cv_scores.std()),  4),
        'training_samples': n,
        'features': FEATURES,
        'preprocessing': 'StandardScaler + Feature Engineering',
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
