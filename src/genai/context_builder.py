import pandas as pd
from typing import Dict, Any, Optional

def build_context(df: Optional[pd.DataFrame] = None, 
                  profile: Optional[Dict[str, Any]] = None, 
                  st_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compiles a structured, JSON-serializable dictionary with all available information
    across the active SynTwin AI pipeline steps.
    """
    context = {}
    st_state = st_state or {}
    
    # 1. Dataset Ingestion & Profiling
    if profile:
        overview = profile.get("overview", {})
        context["dataset"] = {
            "num_rows": overview.get("num_rows", 0),
            "num_columns": overview.get("num_columns", 0),
            "warnings_count": len(profile.get("warnings", [])),
            "quality_status": profile.get("overall_status", "Unknown"),
            "columns": list(profile.get("column_profiles", {}).keys())
        }
        
    # 2. Diagnosis Inferences
    # In Streamlit, we run KPI, pattern, and anomaly engines. If we store them in session state or run them.
    # To keep this clean, check if st_state has active values.
    # We can pull:
    # "discovered_kpis"
    # "detected_patterns"
    # "detected_anomalies"
    # Let's check for these keys in st_state or provide defaults if not loaded yet.
    context["diagnosis"] = {}
    if "discovered_kpis" in st_state:
        context["diagnosis"]["kpis"] = [
            {"name": k["name"], "value": k["value"], "interpretation": k["interpretation"]}
            for k in st_state["discovered_kpis"][:5]
        ]
    if "detected_patterns" in st_state:
        pats = st_state["detected_patterns"]
        context["diagnosis"]["correlations"] = [
            {"col1": c["col1"], "col2": c["col2"], "coefficient": c["coefficient"]}
            for c in pats.get("correlations", [])[:3]
        ]
        context["diagnosis"]["temporal_trends"] = [
            {"date_column": t["date_column"], "value_column": t["value_column"], "trend_type": t["trend_type"]}
            for t in pats.get("temporal_patterns", [])[:2]
        ]
    if "detected_anomalies" in st_state:
        anoms = st_state["detected_anomalies"]
        context["diagnosis"]["anomalies"] = {
            "anomaly_ratio": anoms.get("anomaly_percentage", 0.0),
            "anomaly_count": anoms.get("total_anomalies", 0)
        }

    # 3. Model Predictions
    trained_targets = [k.replace("model_", "") for k in st_state.keys() if k.startswith("model_")]
    if trained_targets:
        target = trained_targets[0]
        meta = st_state.get(f"meta_{target}", {})
        metrics = st_state.get(f"metrics_{target}", {})
        
        context["prediction"] = {
            "selected_target": target,
            "best_model": meta.get("best_name", "N/A"),
            "test_sample_size": meta.get("test_samples", 0),
            "metrics": {k: float(v) for k, v in metrics.items()}
        }
        
        # 4. SHAP Explainability
        # Check global importance
        shap_cache_key = f"shap_vals_{target}_1"
        if shap_cache_key not in st_state:
            shap_cache_key = f"shap_vals_{target}_0"
            
        if shap_cache_key in st_state:
            context["explainability"] = {
                "has_shap_values": True,
                "target_explained": target
            }
            # If global drivers were cached
            if f"global_importance_{target}" in st_state:
                context["explainability"]["global_drivers"] = [
                    {"feature": d["feature"], "relative_importance": d.get("relative_importance", 0.0)}
                    for d in st_state[f"global_importance_{target}"][:5]
                ]

    # 5. Forecasting
    forecast_keys = [k for k in st_state.keys() if k.startswith("forecast_res_")]
    if forecast_keys:
        f_data = st_state[forecast_keys[0]]["results"]
        f_df = f_data.get("forecast_df", None)
        context["forecast"] = {
            "best_model": f_data.get("best_model", "N/A"),
            "horizon_periods": len(f_df) if f_df is not None else 0,
        }
        if f_df is not None:
            # Send mean projected values
            context["forecast"]["average_projection"] = float(f_df["Forecast"].mean())
            context["forecast"]["lower_bound_min"] = float(f_df["Lower Bound"].min())
            context["forecast"]["upper_bound_max"] = float(f_df["Upper Bound"].max())

    # 6. Digital Twin Scenario Builder
    if "last_sim_res" in st_state:
        sim = st_state["last_sim_res"]
        res = sim["res"]
        context["digital_twin"] = {
            "scenario_name": sim["name"],
            "target": sim["target"],
            "baseline_prediction": res["baseline_prediction"],
            "scenario_prediction": res["scenario_prediction"],
            "absolute_change": res.get("abs_difference", 0.0),
            "percentage_change": res.get("pct_difference", 0.0),
            "out_of_range": res.get("out_of_range", False),
            "warnings": res.get("warnings", [])
        }

    # 7. Decision Recommendations
    if "last_ga_res" in st_state:
        ga = st_state["last_ga_res"]
        context["decision"] = {
            "objective_mode": ga.get("objective_mode", "N/A"),
            "ga_status": "success",
            "ga_baseline_prediction": ga.get("baseline_prediction"),
            "ga_optimized_prediction": ga.get("optimized_prediction"),
            "ga_predicted_improvement": ga.get("predicted_improvement"),
            "ga_recommended_values": ga.get("recommended_values", {})
        }
        
    if "last_rl_res" in st_state:
        rl = st_state["last_rl_res"]
        if "decision" not in context:
            context["decision"] = {}
        context["decision"]["rl_status"] = rl.get("status", "failed")
        if rl.get("status") == "success":
            context["decision"]["rl_baseline_prediction"] = rl.get("baseline_prediction")
            context["decision"]["rl_optimized_prediction"] = rl.get("optimized_prediction")
            context["decision"]["rl_predicted_improvement"] = rl.get("predicted_improvement")
            context["decision"]["rl_recommended_values"] = rl.get("recommended_values", {})
            
    return context

def format_context_to_text(context: Dict[str, Any]) -> str:
    """
    Serializes context details into a highly structured markdown prompt block.
    """
    lines = []
    lines.append("## SynTwin Analytical Context\n")
    
    if "dataset" in context:
        d = context["dataset"]
        lines.append(f"### Dataset Profile")
        lines.append(f"- Ingested Size: {d['num_rows']:,} rows, {d['num_columns']} columns")
        lines.append(f"- Data Quality Status: {d['quality_status']} ({d['warnings_count']} quality issues flagged)")
        lines.append(f"- Available Columns: {', '.join(d['columns'][:25])}")
        if len(d["columns"]) > 25:
            lines.append("  (remaining columns truncated...)")
        lines.append("")
        
    if "diagnosis" in context:
        diag = context["diagnosis"]
        lines.append(f"### Diagnostic Inferences")
        if "kpis" in diag:
            lines.append("- Discovered KPIs:")
            for k in diag["kpis"]:
                lines.append(f"  * {k['name']}: {k['value']} ({k['interpretation']})")
        if "correlations" in diag:
            lines.append("- Correlation Patterns:")
            for c in diag["correlations"]:
                lines.append(f"  * '{c['col1']}' / '{c['col2']}': coefficient {c['coefficient']:.3f}")
        if "temporal_trends" in diag:
            lines.append("- Temporal Trend Inferences:")
            for t in diag["temporal_trends"]:
                lines.append(f"  * Trend of '{t['value_column']}' across periods of '{t['date_column']}': {t['trend_type']}")
        if "anomalies" in diag:
            an = diag["anomalies"]
            lines.append(f"- Outlier Records: {an['anomaly_count']:,} anomalies detected ({an['anomaly_ratio']:.2f}% of dataset)")
        lines.append("")
        
    if "prediction" in context:
        p = context["prediction"]
        lines.append(f"### Predictive Models")
        lines.append(f"- Selected Target Objective: '{p['selected_target']}'")
        lines.append(f"- Best Performing Algorithm: {p['best_model']}")
        lines.append(f"- Test Evaluation Sample Size: {p['test_sample_size']:,}")
        lines.append("- Validation Performance Scores:")
        for m_name, m_val in p["metrics"].items():
            lines.append(f"  * {m_name}: {m_val:.4f}")
        lines.append("")
        
    if "explainability" in context:
        exp = context["explainability"]
        lines.append(f"### Model Drivers (SHAP Explainability)")
        lines.append(f"- Feature impact analysis calculated for target '{exp['target_explained']}'")
        if "global_drivers" in exp:
            lines.append("- Strongest Global Drivers (Mean absolute impact):")
            for d in exp["global_drivers"]:
                lines.append(f"  * feature '{d['feature']}': relative impact {d['relative_importance']*100:.1f}%")
        lines.append("")
        
    if "forecast" in context:
        f = context["forecast"]
        lines.append(f"### Forecasting Projection")
        lines.append(f"- Algorithm: {f['best_model']}")
        lines.append(f"- Horizon: {f['horizon_periods']} future periods projected")
        lines.append(f"- Estimated Average: {f['average_projection']:,.2f}")
        lines.append(f"- Uncertainty Range (95% CI): {f['lower_bound_min']:,.2f} to {f['upper_bound_max']:,.2f}")
        lines.append("")
        
    if "digital_twin" in context:
        dt = context["digital_twin"]
        lines.append(f"### Digital Twin Simulation Scenario")
        lines.append(f"- Active Scenario: '{dt['scenario_name']}' (Target: '{dt['target']}')")
        lines.append(f"- Baseline Prediction: {dt['baseline_prediction']}")
        lines.append(f"- Scenario Prediction: {dt['scenario_prediction']}")
        if isinstance(dt["absolute_change"], (int, float)):
            lines.append(f"- Prediction Change: {dt['absolute_change']:+,.2f} ({dt['percentage_change']:+.2f}%)")
        else:
            lines.append(f"- Prediction Change: Outcome Shifted")
        if dt["out_of_range"]:
            lines.append("- Out-of-bounds safety warning triggered (parameters outside historical limits)")
        lines.append("")
        
    if "decision" in context:
        dec = context["decision"]
        lines.append(f"### Decision Recommendations")
        lines.append(f"- Business Objective Mode: {dec.get('objective_mode')}")
        
        if dec.get("ga_status") == "success":
            lines.append("- Genetic Algorithm Search Recommendation:")
            lines.append(f"  * Baseline prediction outcome: {dec.get('ga_baseline_prediction')}")
            lines.append(f"  * Optimized prediction outcome: {dec.get('ga_optimized_prediction')}")
            lines.append(f"  * Estimated Improvement Margin: {dec.get('ga_predicted_improvement')}")
            lines.append("  * Recommended controllable parameters values:")
            for name, val in dec.get("ga_recommended_values", {}).items():
                lines.append(f"    - '{name}': {val:.4f}")
                
        if dec.get("rl_status") == "success":
            lines.append("- Reinforcement Learning Policy Recommendation:")
            lines.append(f"  * Baseline prediction outcome: {dec.get('rl_baseline_prediction')}")
            lines.append(f"  * Optimized policy prediction outcome: {dec.get('rl_optimized_prediction')}")
            lines.append(f"  * Estimated Improvement Margin: {dec.get('rl_predicted_improvement')}")
            lines.append("  * Policy target values:")
            for name, val in dec.get("rl_recommended_values", {}).items():
                lines.append(f"    - '{name}': {val:.4f}")
        lines.append("")
        
    return "\n".join(lines)
