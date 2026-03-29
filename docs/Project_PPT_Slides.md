# Crop Intelligence Platform

## PPT Slide Content

### Slide 1: Title

Crop Intelligence Platform  
Real-Time Crop Recommendation, Yield Prediction, Profit Estimation, and Risk-Aware Ranking

### Slide 2: Problem Statement

- Farmers need better crop decisions
- Soil, climate, yield, and profit must be considered together
- Single-label prediction is not enough

### Slide 3: Objectives

- Recommend best crop
- Predict yield
- Estimate profit
- Measure risk
- Measure sustainability
- Return top 3 crops

### Slide 4: Final Output

- Best crop
- Top 3 crops
- Yield
- Profit
- Risk
- Sustainability
- Map-based area and weather autofill

### Slide 5: Datasets Used

- `real_crop_recommendation_13.csv`
- `real_crop_production_13.csv`
- `real_crop_reference_profiles_13.csv`
- `real_market_lookup_13.csv`
- `real_project_master_13.csv`

### Slide 6: Crop Scope

13 crops:

- Apple
- Banana
- Blackgram
- Coconut
- Coffee
- Grapes
- Jute
- Lentil
- Maize
- Mango
- Orange
- Papaya
- Rice

### Slide 7: Architecture

Show:

- user input
- map and weather
- API
- classifiers
- stacking
- yield regressor
- ranking engine

### Slide 8: Features Used

Classification:

- nitrogen
- phosphorous
- potassium
- temperature
- humidity
- pH
- rainfall

Regression:

- crop year
- area
- ideal crop profile values
- price
- crop
- state
- district
- season

### Slide 9: Models Used

- TabNet slot
- LightGBM classifier
- CatBoost classifier
- Logistic Regression stacking model
- LightGBM regressor

### Slide 10: Why These Models

- TabNet: tabular learning
- LightGBM: strong structured-data performance
- CatBoost: comparison benchmark
- Logistic Regression: simple meta learner
- LightGBM Regressor: yield prediction

### Slide 11: Training Pipeline

- load datasets
- preprocess
- train classifiers
- train stacking model
- train yield regressor
- save artifacts

### Slide 12: Stacking

Base models:

- TabNet slot
- LightGBM

Meta model:

- Logistic Regression

### Slide 13: Yield Prediction

- candidate crop chosen
- regression row built
- yield predicted

### Slide 14: Profit, Risk, Sustainability

- profit = yield × area × price
- risk = climate mismatch
- sustainability = soil/profile closeness

### Slide 15: Ranking Formula

`Final Score = 0.4 Yield + 0.3 Profit + 0.2 Risk + 0.1 Sustainability`

### Slide 16: Map Integration

- draw land
- calculate area
- get coordinates
- auto-fill weather

### Slide 17: Rainfall Logic

- daily rain alone is weak
- use climate-aware rainfall
- combine live weather context and historical rainfall support

### Slide 18: API Endpoints

- `GET /api/health`
- `GET /api/metadata`
- `POST /api/predict`
- `POST /api/train`

### Slide 19: Final Results

- Stacking accuracy: `1.0000`
- LightGBM accuracy: `0.9962`
- CatBoost accuracy: `0.9885`
- Yield R²: `0.8581`
- Top-3 accuracy: `1.0000`

### Slide 20: Strengths

- real data
- ensemble learning
- yield and profit support
- map and weather
- deployable architecture

### Slide 21: Limitations

- only 13 crops
- rainfall interpretation complexity
- market lookup is not full live mandi system

### Slide 22: Future Scope

- more crops
- live mandi APIs
- better climate baselines
- SHAP explanation dashboard
- mobile support

### Slide 23: Conclusion

This is a complete crop intelligence decision-support system, not just a basic crop classifier.

### Slide 24: Viva Quick Answers

- Why stacking: better combined performance
- Why LightGBM: strong tabular model
- Why regression: need yield, not just crop label
- Why map: realistic area and weather capture
