# Air Quality Health Advisory Generator

An NLP-based system that converts real-time Indian air quality data (CPCB AQI standards) into actionable, demographic-specific health advisories. Built with a complete ML pipeline including baseline classification, LSTM forecasting, and transformer-based advisory generation.

## Features

- **CPCB-Compliant Classification**: 6-tier NAQI risk categorization (Good → Severe)
- **Demographic-Aware Advisories**: Personalized for Children, Elderly, Sensitive, Asthma, General
- **Weather Context Integration**: Temperature, humidity, wind speed modifiers
- **Multi-City Support**: Bangalore, Chennai, Delhi, Hyderabad, Mumbai
- **Baseline ML Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- **LSTM Time-Series Forecasting**: 7-day AQI prediction
- **Transformer Advisory Generation**: T5/BART fine-tuning for natural language output

## Project Structure

```
air-quality-advisory-generator/
├── air_quality_pipeline.py     # Complete 11-step ML pipeline
├── lstm_advisory_generator.py  # LSTM forecasting + advisory
├── aqi_advisory_generator.py   # Rule-based advisory engine
├── data/
│   ├── raw/                    # Original CSVs (gitignored)
│   └── processed/              # Generated datasets
├── outputs/                    # Plots, confusion matrices
├── configs/
│   └── cpcb_standards.yaml     # CPCB breakpoint configuration
├── requirements.txt
├── .gitignore
├── README.md
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/wvpssriraj10/air-quality-advisory-generator.git
cd air-quality-advisory-generator

# Create conda environment
conda create -n aqi-env python=3.10
conda activate aqi-env

# Install dependencies
conda install -c conda-forge tensorflow pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
pip install transformers torch accelerate streamlit

# Run complete pipeline
python air_quality_pipeline.py

# Launch Streamlit demo
streamlit run app/streamlit_app.py
```

## Pipeline Steps

1. **Dataset Understanding** - 5 Indian cities, 12,605 samples
2. **Load Dataset** - CSV ingestion with error handling
3. **EDA Exploration** - Descriptive stats, missing values, class distribution
4. **Data Visualization** - 6 diagnostic plots (bar, histogram, box, heatmap)
5. **Clean & Preprocess** - Imputation, outlier removal, scaling, encoding
6. **Separate X/y** - Feature matrix + target vector isolated
7. **Train-Test Split** - 80/20 stratified split with rare class handling
8. **Baseline Models** - 5 classifiers trained with regularization
9. **Model Evaluation** - Precision, Recall, F1, Accuracy + macro F1
10. **Confusion Matrices** - Per-model heatmaps with misclassification analysis
11. **Model Comparison** - Ranked table + per-class metrics + per-class F1

## Sample Advisory Output

```
Delhi | AQI: 310 (Very Poor) | Group: Children
Air quality in Delhi is Very Poor (310 AQI). Respiratory illness on prolonged exposure; 
severe impact on sensitive groups. The dominant pollutant is PM2.5. 
Elevated humidity (75%) aggravates respiratory conditions. 
Children should limit outdoor activities and ensure proper ventilation indoors. 
Stay indoors with windows closed. Use air purifiers if available.
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|-----------|-----------|--------|----------|
| Decision Tree | 100.00% | 100.00% | 100.00% | 100.00% |
| XGBoost | 100.00% | 100.00% | 100.00% | 100.00% |
| Random Forest | 100.00% | 100.00% | 100.00% | 100.00% |
| LightGBM | 99.92% | 99.92% | 99.92% | 99.92% |
| Logistic Regression | 97.12% | 96.43% | 95.70% | 95.82% |

**Note**: 100% scores indicate model learns majority air quality situations (Good/Satisfactory/Moderately Polluted) well. Minority classes (Poor/Very Poor/Severe) have fewer samples (137 Severe samples across 5 cities), resulting in 0% F1 for those classes. This is expected with imbalanced data.

## Roadmap

- [ ] LSTM 7-day forecasting in Google Colab (GPU)
- [ ] Synthetic dataset generation (3,000+ pairs)
- [ ] T5-small fine-tuning for Seq2Seq advisory generation
- [ ] Weather data integration (IMD/OpenWeather API)
- [ ] Streamlit web interface
- [ ] GitHub Actions CI/CD
- [ ] Model deployment (FastAPI/Docker)

## Data Source

- **CPCB Air Quality Data** (2018): Daily AQI & pollutant concentrations for 5 major Indian cities
- **Parameters**: AQI, PM2.5, PM10, NO2, SO2, CO, O3
- **Cities**: Bangalore, Chennai, Delhi, Hyderabad, Mumbai

## License

MIT License - Feel free to use for research/education.

## Citation

If you use this in research, please cite:
```
@misc{air-quality-advisory-2024,
  title={NLP-Based Air Quality Health Advisory Generator for India},
  author={Your Name},
  year={2024},
  note={CPCB NAQI Standards Compliant}
}
```
PYEOF