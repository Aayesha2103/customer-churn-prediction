
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnAI | Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. LOAD SAVED MODEL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "churn_threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = float(joblib.load(THRESHOLD_PATH))


# ============================================================
# 3. SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "prediction"

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    
        """
        <style>

        /* ---------- GLOBAL ---------- */

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.18), transparent 28%),
                radial-gradient(circle at 90% 15%, rgba(14, 165, 233, 0.15), transparent 25%),
                #070b18;
            color: #f8fafc;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        /* ---------- SIDEBAR ---------- */

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, #100d2e 0%, #171447 55%, #0b1224 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="stSidebar"] .block-container {
            padding: 2rem 1.1rem;
        }

        .sidebar-brand {
            padding: 18px 10px 28px 10px;
        }

        .sidebar-icon {
            font-size: 2.2rem;
            margin-bottom: 6px;
        }

        .sidebar-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: white;
            letter-spacing: -0.5px;
        }

        .sidebar-subtitle {
            color: #9aa6c3;
            font-size: 0.82rem;
            margin-top: 4px;
        }

        .sidebar-info {
            margin-top: 30px;
            padding: 18px;
            border-radius: 18px;
            background: rgba(255,255,255,0.055);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .sidebar-info-title {
            color: #c4b5fd;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 1.3px;
            text-transform: uppercase;
        }

        .sidebar-info-text {
            color: #aeb9d2;
            font-size: 0.82rem;
            line-height: 1.55;
            margin-top: 8px;
        }

        /* ---------- HERO ---------- */

        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 42px 46px;
            margin-bottom: 38px;
            background:
                linear-gradient(120deg, #5b21b6 0%, #4338ca 43%, #0891b2 100%);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 25px 70px rgba(30, 27, 75, 0.35);
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            right: -90px;
            top: -130px;
            border-radius: 50%;
            background: rgba(255,255,255,0.09);
        }

        .hero-badge {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.22);
            color: #eef2ff;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            font-size: clamp(2.4rem, 5vw, 4.2rem);
            line-height: 1;
            margin: 24px 0 16px 0;
            color: white;
            letter-spacing: -2px;
            font-weight: 900;
        }

        .hero p {
            position: relative;
            z-index: 1;
            max-width: 720px;
            margin: 0;
            color: #e0e7ff;
            font-size: 1.02rem;
            line-height: 1.7;
        }

        /* ---------- SECTION HEADINGS ---------- */

        .section-heading {
            margin: 38px 0 18px 0;
        }

        .section-heading h2 {
            margin: 0;
            color: #f8fafc;
            font-size: 1.65rem;
            font-weight: 850;
            letter-spacing: -0.6px;
        }

        .section-heading p {
            color: #8794b2;
            margin: 7px 0 0 0;
            font-size: 0.92rem;
        }

        .accent-line {
            width: 64px;
            height: 4px;
            border-radius: 10px;
            margin-top: 12px;
            background: linear-gradient(90deg, #8b5cf6, #22d3ee);
        }

        /* ---------- CARDS ---------- */

        .info-card {
            background: rgba(17, 24, 39, 0.72);
            border: 1px solid rgba(148,163,184,0.13);
            border-radius: 18px;
            padding: 20px;
            height: 100%;
            box-shadow: 0 10px 35px rgba(0,0,0,0.13);
        }

        .info-card-label {
            color: #8491ae;
            font-size: 0.69rem;
            font-weight: 800;
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }

        .info-card-value {
            color: #f8fafc;
            font-size: 1.08rem;
            font-weight: 750;
            margin-top: 8px;
            word-break: break-word;
        }

        /* ---------- FORM AREA ---------- */

        .form-card {
            background: rgba(13, 19, 36, 0.76);
            border: 1px solid rgba(148,163,184,0.13);
            border-radius: 22px;
            padding: 24px 26px 18px 26px;
            margin-bottom: 18px;
        }

        .form-card-title {
            color: #f8fafc;
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 4px;
        }

        .form-card-description {
            color: #7f8ba7;
            font-size: 0.82rem;
            margin-bottom: 18px;
        }

        /* Streamlit labels */
        label, [data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important;
            font-weight: 650 !important;
            font-size: 0.86rem !important;
        }

        /* ---------- INPUTS ---------- */

        /* Explicit widget colors so selected values stay readable on Streamlit Cloud. */
        div[data-baseweb="select"] > div {
            background-color: #f8fafc !important;
            border: 1px solid #dbe2ea !important;
            border-radius: 11px !important;
            color: #172033 !important;
        }

        div[data-baseweb="select"] div,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
        }

        div[data-baseweb="select"] svg {
            fill: #172033 !important;
            color: #172033 !important;
        }

        div[data-testid="stNumberInput"] > div {
            background-color: #f8fafc !important;
            border: 1px solid #dbe2ea !important;
            border-radius: 11px !important;
        }

        div[data-testid="stNumberInput"] input,
        div[data-baseweb="input"] input {
            background-color: transparent !important;
            color: #172033 !important;
            -webkit-text-fill-color: #172033 !important;
            caret-color: #172033 !important;
        }

        div[data-testid="stNumberInput"] button {
            color: #172033 !important;
            background-color: transparent !important;
        }

        div[data-testid="stNumberInput"] button svg {
            fill: #172033 !important;
            color: #172033 !important;
        }

        [role="listbox"],
        [data-baseweb="menu"] {
            background-color: #ffffff !important;
            color: #172033 !important;
        }

        [role="listbox"] [role="option"],
        [data-baseweb="menu"] li {
            background-color: #ffffff !important;
            color: #172033 !important;
        }

        [role="listbox"] [role="option"] *,
        [data-baseweb="menu"] li * {
            color: #172033 !important;
        }

        [role="listbox"] [role="option"]:hover,
        [data-baseweb="menu"] li:hover {
            background-color: #eef2ff !important;
            color: #172033 !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
            opacity: 1 !important;
        }

        /* ---------- BUTTON ---------- */

        .stButton > button {
            width: 100%;
            min-height: 50px;
            border: 0;
            border-radius: 14px;
            color: white;
            font-weight: 800;
            font-size: 0.98rem;
            background: linear-gradient(90deg, #7c3aed, #2563eb, #06b6d4);
            box-shadow: 0 12px 30px rgba(59,130,246,0.20);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 38px rgba(59,130,246,0.30);
            color: white;
        }

        /* ---------- RESULT HERO ---------- */

        .result-card {
            position: relative;
            overflow: hidden;
            padding: 32px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(30,41,59,0.92), rgba(15,23,42,0.95));
            border: 1px solid rgba(139,92,246,0.28);
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
            margin-bottom: 24px;
        }

        .result-label {
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 1.6px;
            text-transform: uppercase;
        }

        .probability {
            font-size: clamp(3rem, 7vw, 5.5rem);
            line-height: 1;
            font-weight: 900;
            letter-spacing: -3px;
            margin: 12px 0;
            color: #f8fafc;
        }

        .risk-high {
            color: #fb7185;
            font-size: 1.25rem;
            font-weight: 850;
        }

        .risk-low {
            color: #34d399;
            font-size: 1.25rem;
            font-weight: 850;
        }

        .threshold-note {
            color: #94a3b8;
            font-size: 0.86rem;
            line-height: 1.6;
            margin-top: 10px;
        }

        .recommendation {
            border-radius: 16px;
            padding: 18px 20px;
            background: rgba(245,158,11,0.08);
            border: 1px solid rgba(245,158,11,0.20);
            color: #fbbf24;
            line-height: 1.6;
            margin-top: 18px;
        }

        .recommendation-low {
            background: rgba(52,211,153,0.07);
            border-color: rgba(52,211,153,0.18);
            color: #6ee7b7;
        }

        /* ---------- FOOTER ---------- */

        .footer {
            margin-top: 55px;
            padding-top: 20px;
            border-top: 1px solid rgba(148,163,184,0.10);
            color: #596783;
            text-align: center;
            font-size: 0.78rem;
        }

        /* ---------- MOBILE ---------- */

        @media (max-width: 800px) {
            .block-container {
                padding-top: 1rem;
            }

            .hero {
                padding: 30px 25px;
                border-radius: 22px;
            }

            .hero h1 {
                font-size: 2.5rem;
            }
        }

        </style>
        """,
    unsafe_allow_html=True
)


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        
            """
            <div class="sidebar-brand">
                <div class="sidebar-icon">📊</div>
                <div class="sidebar-title">ChurnAI</div>
                <div class="sidebar-subtitle">Customer Intelligence</div>
            </div>
            """,
        unsafe_allow_html=True
    )

    if st.button("🏠  Customer Prediction", use_container_width=True):
        st.session_state.page = "prediction"
        st.rerun()

    if st.button("📈  Prediction Results", use_container_width=True):
        if st.session_state.prediction_data is not None:
            st.session_state.page = "results"
            st.rerun()
        else:
            st.warning("Run a prediction first.")

    st.markdown(
        
            """
            <div class="sidebar-info">
                <div class="sidebar-info-title">Smart Prediction</div>
                <div class="sidebar-info-text">
                    Estimate customer churn probability and identify
                    customers who may need retention attention.
                </div>
            </div>
            """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def section_heading(title, description=""):
    description_html = f'<p>{description}</p>' if description else ""
    st.markdown(
        
            f"""
            <div class="section-heading">
                <h2>{title}</h2>
                <div class="accent-line"></div>
                {description_html}
            </div>
            """,
        unsafe_allow_html=True
    )


def info_card(label, value):
    st.markdown(
        
            f"""
            <div class="info-card">
                <div class="info-card-label">{label}</div>
                <div class="info-card-value">{value}</div>
            </div>
            """,
        unsafe_allow_html=True
    )


def make_prediction(data):
    input_df = pd.DataFrame([data])

    probability = float(model.predict_proba(input_df)[0][1])

    prediction = "Yes" if probability >= threshold else "No"

    return probability, prediction


# ============================================================
# 7. CUSTOMER PREDICTION PAGE
# ============================================================

if st.session_state.page == "prediction":

    st.markdown(
        
            """
            <div class="hero">
                <div class="hero-badge">Machine Learning • Customer Analytics</div>
                <h1>Customer Churn<br>Predictor</h1>
                <p>
                    Predict whether a telecom customer is likely to leave
                    and identify customers who may require proactive
                    retention attention.
                </p>
            </div>
            """,
        unsafe_allow_html=True
    )

    section_heading(
        "Customer Profile",
        "Enter the customer's information to generate a personalized churn prediction."
    )

    # ---------- BASIC CUSTOMER INFORMATION ----------

    st.markdown(
        '<div class="form-card"><div class="form-card-title">👤 Customer Information</div>'
        '<div class="form-card-description">Basic characteristics and relationship information.</div></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col2:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col3:
        partner = st.selectbox("Partner", ["No", "Yes"])

    with col4:
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    tenure = st.slider(
        "Customer Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )

    st.caption(f"Customer has been with the company for **{tenure} months**.")

    # ---------- TELECOM SERVICES ----------

    st.markdown(
        '<div class="form-card"><div class="form-card-title">📡 Telecom Services</div>'
        "<div class='form-card-description'>Information about the customer's telecom and online services.</div></div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])

    with col2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col3:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )

    with col2:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    # ---------- CONTRACT & BILLING ----------

    st.markdown(
        '<div class="form-card"><div class="form-card-title">💳 Contract & Billing</div>'
        '<div class="form-card-description">Contract, billing and payment details.</div></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

    with col3:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.01
        )

    with col2:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=0.01
        )

    st.write("")

    # ---------- PREDICT ----------

    if st.button("🔮  Predict Customer Churn", use_container_width=True):

        customer_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        probability, prediction = make_prediction(customer_data)

        st.session_state.prediction_data = {
            "data": customer_data,
            "probability": probability,
            "prediction": prediction
        }

        st.session_state.page = "results"
        st.rerun()

    st.markdown(
        
            """
            <div class="footer">
                ChurnAI • Customer Churn Prediction • Machine Learning & Data Science Project
            </div>
            """,
        unsafe_allow_html=True
    )


# ============================================================
# 8. RESULTS PAGE
# ============================================================

else:

    result = st.session_state.prediction_data

    if result is None:
        st.session_state.page = "prediction"
        st.rerun()

    customer = result["data"]
    probability = result["probability"]
    prediction = result["prediction"]

    risk_high = probability >= threshold
    risk_text = "High Churn Risk" if risk_high else "Low Churn Risk"
    risk_class = "risk-high" if risk_high else "risk-low"

    section_heading(
        "Prediction Result",
        "The model's assessment of this customer's likelihood of churn."
    )

    # ---------- MAIN RESULT ----------

    st.markdown(
        
            f"""
            <div class="result-card">
                <div class="result-label">Churn Probability</div>
                <div class="probability">{probability:.2%}</div>
                <div class="{risk_class}">
                    {"⚠️" if risk_high else "✓"} {risk_text}
                </div>
                <div class="threshold-note">
                    Decision threshold: <strong>{threshold:.2f}</strong>.
                    A customer is classified as a potential churner when
                    the predicted probability reaches or exceeds this threshold.
                </div>
            </div>
            """,
        unsafe_allow_html=True
    )

    # ---------- PROBABILITY BAR ----------

    st.markdown("**Churn probability**")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"{probability:.2%} predicted probability of churn • "
        f"{threshold:.0%} classification threshold"
    )

    st.write("")

    # ---------- KEY METRICS ----------

    col1, col2, col3 = st.columns(3)

    with col1:
        info_card("Model Decision", "Potential Churner" if risk_high else "Likely to Stay")

    with col2:
        info_card("Decision Threshold", f"{threshold:.2f}")

    with col3:
        info_card("Customer Tenure", f"{customer['tenure']} months")

    # ---------- CHARTS ----------

    section_heading(
        "Risk Analysis",
        "Visual breakdown of the prediction and the decision boundary."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        fig, ax = plt.subplots(figsize=(6, 4))

        values = [probability, 1 - probability]
        labels = ["Churn Risk", "Stay Probability"]

        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.42}
        )

        ax.set_title("Customer Risk Distribution", pad=18, fontsize=13)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with chart_col2:

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.barh(
            ["Churn Probability"],
            [probability]
        )

        ax.axvline(
            threshold,
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold:.2f})"
        )

        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Probability vs Decision Threshold", pad=18, fontsize=13)
        ax.legend()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ---------- INTERPRETATION ----------

    if risk_high:
        st.markdown(
            
                """
                <div class="recommendation">
                    <strong>⚠️ Retention attention recommended</strong><br>
                    This customer has been identified as a potential churner.
                    The business could prioritize this customer for a targeted
                    retention strategy.
                </div>
                """,
            unsafe_allow_html=True
        )

        st.info(
            "Possible actions: personalized offers, proactive customer support, "
            "contract incentives, or service improvements."
        )

    else:
        st.markdown(
            
                """
                <div class="recommendation recommendation-low">
                    <strong>✓ Lower churn risk</strong><br>
                    The predicted probability is below the selected decision
                    threshold, so this customer is currently classified as
                    unlikely to churn.
                </div>
                """,
            unsafe_allow_html=True
        )

    # ---------- CUSTOMER SUMMARY ----------

    section_heading(
        "Customer Summary",
        "The main characteristics used for this prediction."
    )

    summary = [
        ("Contract", customer["Contract"]),
        ("Internet Service", customer["InternetService"]),
        ("Monthly Charges", f"₹{customer['MonthlyCharges']:.2f}"),
        ("Payment Method", customer["PaymentMethod"]),
        ("Online Security", customer["OnlineSecurity"]),
        ("Tech Support", customer["TechSupport"]),
        ("Device Protection", customer["DeviceProtection"]),
        ("Streaming TV", customer["StreamingTV"]),
        ("Streaming Movies", customer["StreamingMovies"]),
        ("Partner", customer["Partner"]),
        ("Dependents", customer["Dependents"]),
        ("Paperless Billing", customer["PaperlessBilling"])
    ]

    for start in range(0, len(summary), 4):
        cols = st.columns(4)

        for col, (label, value) in zip(cols, summary[start:start + 4]):
            with col:
                info_card(label, value)

        st.write("")

    # ---------- MODEL EXPLANATION ----------

    section_heading(
        "What the Model Does",
        "A simple interpretation of the prediction."
    )

    st.markdown(
        
            f"""
            <div class="form-card">
                <div class="form-card-title">🤖 How to read this result</div>
                <div class="form-card-description" style="font-size:0.92rem; line-height:1.8;">
                    The model estimates how likely this customer is to leave the
                    telecom service based on the information entered above.
                    <br><br>
                    For this customer, the estimated probability of churn is
                    <strong>{probability:.2%}</strong>.
                    <br><br>
                    Because the probability is
                    {"above" if risk_high else "below"} the selected threshold of
                    <strong>{threshold:.2f}</strong>, the model's decision is:
                    <strong>
                        {"potential churner" if risk_high else "likely to stay"}
                    </strong>.
                    <br><br>
                    This prediction can help a business identify customers who
                    may need attention before they decide to leave.
                </div>
            </div>
            """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button("←  Back to Customer Prediction", use_container_width=True):
        st.session_state.page = "prediction"
        st.rerun()

    st.markdown(
        
            """
            <div class="footer">
                ChurnAI • Customer Churn Prediction • Machine Learning & Data Science Project
            </div>
            """,
        unsafe_allow_html=True
    )
