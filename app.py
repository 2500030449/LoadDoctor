
import os
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="LoopDoctor | AI Monitor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "Random Forest": os.path.join(
        BASE_DIR, "models", "random_forest_model.pkl"
    ),
    "KNN": os.path.join(
        BASE_DIR, "models", "knn_model.pkl"
    )
}

DEFAULT_FEATURES = [
    "feedback",
    "error",
    "feedback_change",
    "absolute_error"
]

# =====================================================
# 2. ANIMATED CSS AND BUTTON DESIGN
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #172d50 0%, transparent 40%),
        radial-gradient(ellipse at 95% 10%, #30204c 0%, transparent 35%),
        linear-gradient(145deg, #080d1b, #0b1325 60%, #10182c);
    color: #edf4ff;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    z-index: 999999;
    background: linear-gradient(
        90deg, #36e4f5, #8274ff, #ff69b4, #36e4f5
    );
    background-size: 300% 100%;
    animation: colorFlow 7s linear infinite;
    pointer-events: none;
}

@keyframes colorFlow {
    0% { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes buttonGradient {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
    animation: fadeUp 0.6s ease-out;
}

h1, h2, h3, p, label {
    color: #edf4ff;
}

h1 {
    font-weight: 900 !important;
    letter-spacing: -1px;
}

h2, h3 {
    font-weight: 750 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #0b1223, #101c33 55%, #17152f
    );
    border-right: 1px solid #526b923b;
}

[data-testid="stSidebar"] * {
    color: #e8f2ff;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border: 1px solid transparent;
    border-radius: 12px;
    padding: 10px 12px;
    transition: all 0.25s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(54, 228, 245, 0.12);
    border-color: rgba(54, 228, 245, 0.35);
    transform: translateX(4px);
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #16243a, #101a2d);
    border: 1px solid #6d8bb73b;
    border-radius: 18px;
    padding: 20px;
    animation: fadeUp 0.6s ease-out;
    transition: transform 0.25s ease, border-color 0.25s ease,
                box-shadow 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #36e4f5a0;
    box-shadow: 0 10px 30px #36e4f51b;
}

[data-testid="stMetricLabel"] {
    color: #a6b9d7 !important;
}

[data-testid="stMetricValue"] {
    color: #f2f7ff !important;
    font-weight: 800 !important;
}

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #101b30 !important;
    color: #f1f7ff !important;
    border-color: #526b92 !important;
    border-radius: 10px !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: #36e4f5 !important;
    box-shadow: 0 0 0 2px #36e4f526 !important;
}

[data-testid="stNumberInput"] button {
    background: #1b2a43 !important;
    color: #eaf2ff !important;
    border: 1px solid #344866 !important;
    border-radius: 8px !important;
}

[data-testid="stNumberInput"] button:hover {
    background: #155e75 !important;
    border-color: #36e4f5 !important;
}

/* Form */
[data-testid="stForm"] {
    background: linear-gradient(145deg, #14223a, #10192b);
    border: 1px solid #6b86ae50;
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 12px 38px #00000026;
}

/* Animated gradient buttons */
.stButton button,
.stFormSubmitButton button,
[data-testid="stDownloadButton"] button {
    background: linear-gradient(
        110deg, #06b6d4 0%, #3b82f6 48%, #8b5cf6 100%
    ) !important;
    background-size: 200% 200% !important;
    color: #ffffff !important;
    border: 1px solid rgba(125, 211, 252, 0.65) !important;
    border-radius: 14px !important;
    min-height: 46px;
    padding: 11px 20px;
    font-size: 15px !important;
    font-weight: 750 !important;
    box-shadow: 0 5px 16px rgba(59, 130, 246, 0.22);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: buttonGradient 6s ease infinite;
}

.stButton button:hover,
.stFormSubmitButton button:hover,
[data-testid="stDownloadButton"] button:hover {
    transform: translateY(-3px);
    border-color: #a5f3fc !important;
    box-shadow: 0 8px 28px rgba(6, 182, 212, 0.30);
}

.stButton button:active,
.stFormSubmitButton button:active {
    transform: scale(0.98);
}

.stFormSubmitButton button {
    min-height: 54px !important;
    font-size: 16px !important;
}

/* Graphs */
[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg, #111c30, #0e1729);
    border: 1px solid #617ba63b;
    border-radius: 18px;
    padding: 8px;
    margin-top: 8px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
}

[data-testid="stPlotlyChart"]:hover {
    border-color: #36e4f56b;
    box-shadow: 0 8px 28px #36e4f510;
}

[data-testid="stDataFrame"] {
    border: 1px solid #617ba63b;
    border-radius: 14px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: #6b83a42e !important;
}

.footer {
    text-align: center;
    color: #8295b5;
    font-size: 12px;
    padding: 28px 0 10px;
    letter-spacing: 0.5px;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD BOTH MODELS
# =====================================================
@st.cache_resource
def load_saved_model(path, modified_time):
    saved = joblib.load(path)

    if isinstance(saved, dict):
        model = saved["model"]
        features = saved.get("features", DEFAULT_FEATURES)
    else:
        model = saved
        features = DEFAULT_FEATURES

    return model, features


available_models = {}
model_errors = {}

for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            available_models[name] = load_saved_model(
                path, os.path.getmtime(path)
            )
        except Exception as exc:
            model_errors[name] = str(exc)


# =====================================================
# 4. SESSION STATE
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# =====================================================
# 5. PREDICTION HELPER
# =====================================================
def predict_with_model(model, features, feedback, setpoint,
                       feedback_change):
    error = setpoint - feedback

    values = {
        "feedback": float(feedback),
        "setpoint": float(setpoint),
        "error": float(error),
        "absolute_error": float(abs(error)),
        "feedback_change": float(feedback_change)
    }

    missing = [f for f in features if f not in values]
    if missing:
        raise ValueError("Missing supported features: " + str(missing))

    input_df = pd.DataFrame([
        {feature: values[feature] for feature in features}
    ])

    prediction = str(model.predict(input_df)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(max(probabilities)) * 100

    return prediction, error, abs(error), confidence


def make_record(sample, model_name, prediction, setpoint,
                feedback, feedback_change, error, absolute_error,
                confidence):
    return {
        "sample": sample,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "setpoint": float(setpoint),
        "feedback": float(feedback),
        "feedback_change": float(feedback_change),
        "error": float(error),
        "absolute_error": float(absolute_error),
        "prediction": prediction,
        "confidence_percent": confidence
    }


# =====================================================
# 6. GRAPH STYLE
# =====================================================
def style_chart(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF2FF", family="Inter"),
        height=height,
        margin=dict(l=20, r=20, t=65, b=30),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hovermode="x unified"
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(150,170,210,0.12)"
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(150,170,210,0.12)"
    )
    return fig


# =====================================================
# 7. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("# 🩺 LoopDoctor")
    st.caption("AI Feedback-Noise Monitor")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Predict Diagnosis",
            "Compare Models",
            "Live Graphs",
            "Prediction History",
            "About Project"
        ]
    )

    st.divider()
    st.markdown("### MODEL STATUS")

    for model_name, path in MODEL_PATHS.items():
        if model_name in available_models:
            st.success(f"{model_name}: Loaded")
        elif model_name in model_errors:
            st.warning(f"{model_name}: Load error")
        else:
            st.error(f"{model_name}: File missing")

    st.caption("Manual input / simulation mode")


# =====================================================
# 8. HEADER
# =====================================================
st.markdown("""
<h1 style="
    background: linear-gradient(90deg, #ffffff, #36e4f5, #a99aff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 38px;
    margin-bottom: 0;">
    LoopDoctor AI Monitor
</h1>
<p style="color:#9aacc8; margin-top:8px;">
    Intelligent feedback analysis • Machine learning diagnosis • Visual analytics
</p>
""", unsafe_allow_html=True)

st.divider()


# =====================================================
# 9. DASHBOARD
# =====================================================
if page == "Dashboard":
    st.subheader("✨ System Overview")

    history = pd.DataFrame(st.session_state.history)
    total = len(history)

    if total:
        noise_count = int(
            (history["prediction"] == "feedback_noise").sum()
        )
        normal_count = int(
            (history["prediction"] == "normal").sum()
        )
        latest = history.iloc[-1]
    else:
        noise_count = normal_count = 0
        latest = None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", total)
    c2.metric("Normal Readings", normal_count)
    c3.metric("Noise Predictions", noise_count)
    c4.metric(
        "Latest Absolute Error",
        f"{latest['absolute_error']:.2f}" if total else "—"
    )

    st.markdown("### 📊 Recent Monitoring Results")

    if total:
        fig = px.line(
            history,
            x="sample",
            y=["setpoint", "feedback"],
            markers=True,
            title="Setpoint vs Feedback"
        )
        st.plotly_chart(
            style_chart(fig, 420),
            use_container_width=True,
            key="dashboard_readings"
        )

        st.markdown("### Latest Records")
        st.dataframe(
            history[
                ["timestamp", "model", "setpoint", "feedback",
                 "error", "prediction"]
            ].tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "Go to Predict Diagnosis and submit your first reading "
            "to populate the dashboard."
        )

    st.caption(
        "Graphs update after submitted predictions. "
        "This is not live hardware telemetry."
    )


# =====================================================
# 10. PREDICT DIAGNOSIS
# =====================================================
elif page == "Predict Diagnosis":
    st.subheader("🔮 Predict Feedback Condition")

    if not available_models:
        st.error(
            "No trained models found. Run src/train_models.py first."
        )
        for name, error_message in model_errors.items():
            st.warning(f"{name}: {error_message}")
    else:
        selected_model = st.selectbox(
            "Choose Prediction Model",
            list(available_models.keys())
        )

        st.caption(f"Selected model: {selected_model}")

        with st.form("prediction_form"):
            st.markdown("### Sensor Input Values")

            c1, c2 = st.columns(2)

            with c1:
                setpoint = st.number_input(
                    "🎯 Setpoint", value=50.0, step=1.0
                )
                feedback = st.number_input(
                    "📡 Current Feedback", value=49.0, step=1.0
                )

            with c2:
                feedback_change = st.number_input(
                    "↕️ Feedback Change", value=0.5, step=0.1
                )
                st.info(
                    "These are manually entered values, not live "
                    "sensor measurements."
                )

            submitted = st.form_submit_button(
                "🚀 Predict & Update Graphs",
                use_container_width=True
            )

        if submitted:
            try:
                model, features = available_models[selected_model]

                prediction, error, absolute_error, confidence = (
                    predict_with_model(
                        model,
                        features,
                        feedback,
                        setpoint,
                        feedback_change
                    )
                )

                record = make_record(
                    len(st.session_state.history) + 1,
                    selected_model,
                    prediction,
                    setpoint,
                    feedback,
                    feedback_change,
                    error,
                    absolute_error,
                    confidence
                )

                st.session_state.history.append(record)
                st.session_state.last_prediction = record
                st.rerun()

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.last_prediction

    if result is not None:
        st.divider()
        st.markdown("### 🧠 Latest Prediction")

        st.caption(f"Model used: {result['model']}")

        if result["prediction"] == "normal":
            st.success("Prediction: NORMAL")
        elif result["prediction"] == "feedback_noise":
            st.error("Prediction: FEEDBACK NOISE")
        else:
            st.warning(f"Model output: {result['prediction']}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Error", f"{result['error']:.2f}")
        c2.metric("Absolute Error", f"{result['absolute_error']:.2f}")
        c3.metric(
            "Feedback Change",
            f"{result['feedback_change']:.2f}"
        )

        if result["confidence_percent"] is not None:
            st.metric(
                "Model Confidence",
                f"{result['confidence_percent']:.1f}%"
            )

        st.caption(
            "Model output is a classification, not confirmation "
            "of a physical hardware fault."
        )


# =====================================================
# 11. COMPARE RANDOM FOREST AND KNN
# =====================================================
elif page == "Compare Models":
    st.subheader("⚖️ Random Forest vs KNN")

    if len(available_models) < 2:
        st.warning(
            "Both trained models are needed for comparison. "
            "Run src/train_models.py first."
        )
    else:
        st.write(
            "Enter one set of readings to compare the predictions "
            "from both models."
        )

        with st.form("compare_form"):
            c1, c2 = st.columns(2)

            with c1:
                compare_setpoint = st.number_input(
                    "Setpoint", value=50.0, step=1.0,
                    key="compare_setpoint"
                )
                compare_feedback = st.number_input(
                    "Feedback", value=49.0, step=1.0,
                    key="compare_feedback"
                )

            with c2:
                compare_change = st.number_input(
                    "Feedback Change", value=0.5, step=0.1,
                    key="compare_change"
                )

            compare_clicked = st.form_submit_button(
                "🔍 Compare Both Models",
                use_container_width=True
            )

        if compare_clicked:
            comparison_records = []

            for model_name, (model, features) in available_models.items():
                try:
                    pred, err, abs_err, confidence = predict_with_model(
                        model,
                        features,
                        compare_feedback,
                        compare_setpoint,
                        compare_change
                    )

                    comparison_records.append({
                        "Model": model_name,
                        "Prediction": pred,
                        "Error": err,
                        "Absolute Error": abs_err,
                        "Confidence (%)": (
                            round(confidence, 2)
                            if confidence is not None else None
                        )
                    })
                except Exception as exc:
                    st.error(f"{model_name} failed: {exc}")

            if comparison_records:
                comparison_df = pd.DataFrame(comparison_records)

                st.markdown("### Results")
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )

                if comparison_df["Prediction"].nunique() > 1:
                    st.warning(
                        "The models disagree. Check the input values "
                        "and validate both models on independent test data."
                    )
                else:
                    st.success(
                        "Both models returned the same class for this input."
                    )

                fig = px.bar(
                    comparison_df,
                    x="Model",
                    y="Absolute Error",
                    color="Model",
                    text="Absolute Error",
                    title="Input Absolute Error"
                )
                st.plotly_chart(
                    style_chart(fig, 350),
                    use_container_width=True,
                    key="comparison_error_chart"
                )


# =====================================================
# 12. LIVE GRAPHS
# =====================================================
elif page == "Live Graphs":
    st.subheader("📈 Reading Variations & Analytics")

    history = pd.DataFrame(st.session_state.history)

    if history.empty:
        st.info(
            "No graph data yet. Submit predictions first."
        )
    else:
        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.line(
                history, x="sample",
                y=["setpoint", "feedback"],
                color_discrete_sequence=["#36e4f5", "#a99aff"],
                markers=True,
                title="Setpoint and Feedback Variation"
            )
            st.plotly_chart(
                style_chart(fig1, 390),
                use_container_width=True,
                key="graph_readings"
            )

        with c2:
            fig2 = px.line(
                history, x="sample", y="error",
                color="model", markers=True,
                title="Error Variation by Model"
            )
            fig2.add_hline(y=0, line_dash="dash")
            st.plotly_chart(
                style_chart(fig2, 390),
                use_container_width=True,
                key="graph_error"
            )

        c3, c4 = st.columns(2)

        with c3:
            fig3 = px.line(
                history, x="sample", y="feedback_change",
                color="model", markers=True,
                title="Feedback Change Variation"
            )
            st.plotly_chart(
                style_chart(fig3, 360),
                use_container_width=True,
                key="graph_change"
            )

        with c4:
            counts = (
                history.groupby(["model", "prediction"])
                .size()
                .reset_index(name="Count")
            )
            fig4 = px.bar(
                counts, x="model", y="Count",
                color="prediction", barmode="group",
                text="Count",
                title="Predictions by Model"
            )
            st.plotly_chart(
                style_chart(fig4, 360),
                use_container_width=True,
                key="graph_distribution"
            )

        st.markdown("### 🗂️ Recorded Readings")
        st.dataframe(
            history.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Graph Data (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="loopdoctor_readings.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear Prediction History"):
            st.session_state.history = []
            st.session_state.last_prediction = None
            st.rerun()


# =====================================================
# 13. PREDICTION HISTORY
# =====================================================
elif page == "Prediction History":
    st.subheader("🕘 Prediction History")

    history = pd.DataFrame(st.session_state.history)

    if history.empty:
        st.info("No predictions have been recorded yet.")
    else:
        st.metric("Total Saved Predictions", len(history))

        st.dataframe(
            history.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Prediction History",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="loopdoctor_prediction_history.csv",
            mime="text/csv"
        )


# =====================================================
# 14. ABOUT PROJECT
# =====================================================
elif page == "About Project":
    st.subheader("💡 About LoopDoctor")

    st.markdown("""
    ## 🩺 LoopDoctor – Feedback-Noise Detector

    **Objective:** Demonstrate machine-learning classification of
    feedback readings into normal and feedback-noise categories.

    ### Algorithms
    - **Random Forest:** Combines multiple decision trees.
    - **KNN:** Classifies an input using its nearest training examples.

    ### Features
    - Manual feedback and setpoint inputs.
    - Model selection and side-by-side comparison.
    - Error and absolute-error calculation.
    - Interactive Plotly graphs.
    - Prediction history and CSV download.
    - Animated dashboard and buttons.

    ### Technologies
    Python, Streamlit, Pandas, Plotly, Scikit-learn and Joblib.

    ### Current Limitation
    Readings are entered manually. Live hardware monitoring requires
    a verified sensor-data connection.

    Model classifications do not independently establish physical
    equipment safety or confirm a real hardware fault.
    """)


# =====================================================
# 15. FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    🩺 LOOPDOCTOR | RANDOM FOREST + KNN |
    AI FEEDBACK-NOISE MONITOR
</div>
""", unsafe_allow_html=True)
