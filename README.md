# SynTwin AI

### Domain-Adaptive Business Intelligence & Decision Intelligence Platform

SynTwin AI is an AI-powered business analytics and decision intelligence platform that takes a structured CSV or Excel dataset and automatically transforms it into actionable business insights.

Instead of requiring a predefined dataset or fixed business domain, SynTwin analyzes the uploaded data, identifies important patterns and variables, and provides:

- Business intelligence dashboards
- Data quality and anomaly analysis
- Automated diagnosis and KPI discovery
- Predictive machine learning
- SHAP-based explainability
- Time-series forecasting
- Digital Twin what-if simulations
- Genetic Algorithm optimization
- Reinforcement Learning decision optimization
- RAG-powered knowledge grounding
- Gemini-powered AI Assistant

The goal is simple:

**Upload business data → Understand → Diagnose → Predict → Explain → Forecast → Simulate → Optimize → Decide**

---

## Key Features

### 1. Data Ingestion & Profiling
Upload CSV or Excel datasets.

SynTwin automatically detects:
- Numeric variables
- Categorical variables
- Dates
- Boolean variables
- Missing values
- Duplicates
- High-cardinality columns
- Basic data-quality issues

The system is designed to work with previously unseen datasets.

### 2. Business Diagnosis
Automatically identifies useful business metrics and patterns.

Includes:
- KPI discovery
- Correlation analysis
- Temporal patterns
- Outlier detection
- Data-health indicators

### 3. Predictive Analytics
Automatically recommends suitable prediction targets and trains appropriate models.

Supports:
- Regression
- Classification
- Automatic preprocessing
- Model evaluation
- Representative sampling for large datasets

### 4. Explainable AI
Uses SHAP (SHapley Additive exPlanations) to explain model predictions.

Provides:
- Global feature importance
- Individual prediction explanations
- Strongest model drivers

### 5. Forecasting
Detects suitable date and business-metric columns and generates future projections using time-series forecasting techniques.

### 6. Digital Twin
The Digital Twin module creates a model-based representation of the business system.

Users can modify relevant variables and perform:

**"What happens if this variable changes?"**

The system compares the baseline and simulated outcomes.

### 7. Decision Intelligence
SynTwin converts analytical results into decision recommendations using:

- Genetic Algorithm optimization
- Reinforcement Learning with PPO
- Custom Gymnasium simulation environments

These methods search for better values of controllable variables according to the selected business objective.

### 8. AI Assistant
The AI Assistant allows users to ask natural-language questions about the current analysis.

Examples:

- "Summarize the current situation."
- "What are the biggest problems?"
- "What are the most important factors?"
- "Why did the model make this prediction?"
- "What does the forecast indicate?"
- "What happens if I change this variable?"
- "What action is recommended?"

The assistant uses the current SynTwin analytical context, including available:

- Dataset information
- Data quality
- Diagnosis
- Prediction results
- SHAP explanations
- Forecasts
- Digital Twin results
- Decision/optimization results
- RAG knowledge

Gemini is used for natural-language response generation when configured.

A local fallback is available when the LLM is unavailable.

---

## Technology Stack

| Area | Technology |
|---|---|
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainability | SHAP |
| Forecasting | Statsmodels |
| Reinforcement Learning | Gymnasium, Stable-Baselines3 |
| Optimization | Genetic Algorithm |
| Generative AI | Google Gemini API |
| RAG | TF-IDF / cosine similarity |
| Documents | PyPDF, python-docx |
| Testing | Pytest |
| Configuration | python-dotenv |

---

## Architecture

```text
                  ┌─────────────────────┐
                  │   CSV / Excel Data  │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Ingestion & Profile │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Diagnosis & Quality │
                  └──────────┬──────────┘
                             ↓
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
        Prediction       Forecasting      Anomaly
              ↓              ↓              ↓
           SHAP        Digital Twin       Insights
              └──────────────┼──────────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Decision Intelligence│
                  │       GA + RL       │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │    AI Assistant     │
                  │       Gemini        │
                  │    + RAG Context    │
                  └─────────────────────┘