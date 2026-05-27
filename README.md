# EstateIQ — House Price Predictor
> ML-powered real estate valuation using Flask + Scikit-Learn

## Project Structure
```
house_price_predictor/
├── app.py               ← Flask backend + ML model
├── requirements.txt     ← Python dependencies
├── templates/
│   └── index.html       ← Frontend (HTML + CSS + JS)
└── README.md
```

## Tech Stack
| Layer      | Technology |
|------------|-----------|
| Backend    | Python 3.10+, Flask |
| ML Model   | Scikit-Learn — Gradient Boosting Regressor |
| Data       | Pandas, NumPy |
| Frontend   | HTML5, CSS3, Vanilla JavaScript |

## ML Pipeline Details
- **Algorithm**: `GradientBoostingRegressor` — 300 estimators, lr=0.05, max_depth=5
- **Preprocessing**: `StandardScaler` inside a `Pipeline`
- **Feature Engineering**: 4 derived features (price_per_sqft_proxy, total_rooms, age_condition, luxury_score)
- **Validation**: 5-Fold Cross Validation (R² ~0.96+)

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Flask server
```bash
python app.py
```

### 3. Open your browser
```
http://localhost:5000
```

## API Endpoints
| Method | Endpoint       | Description |
|--------|---------------|-------------|
| GET    | `/`           | Main UI |
| POST   | `/predict`    | Get price prediction (JSON body) |
| GET    | `/model-info` | Model metadata and R² score |

### POST /predict — Request body
```json
{
  "area": 1800,
  "bedrooms": 3,
  "bathrooms": 2,
  "floors": 1,
  "age": 10,
  "garage": 1,
  "location": 2,
  "condition": 3
}
```

### POST /predict — Response
```json
{
  "predicted_price": 385000.0,
  "price_range": { "low": 354200.0, "high": 415800.0 },
  "model_accuracy": 96.3,
  "feature_importance": [...],
  "location_name": "Town",
  "condition_name": "Good"
}
```
