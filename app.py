# -*- coding: utf-8 -*-

import os
import pickle
from textwrap import dedent

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

# 必须保留：
# pickle 反序列化 TabICLv2 模型时需要对应类存在
from tabicl import TabICLClassifier


# ============================================================
# 0. Global configuration
# ============================================================

EXPECTED_FEATURES = [
    "PA",
    "Age",
    "Fbg",
    "ALB",
    "ChE",
    "Lymph%",
    "PLT",
    "Ca",
]


# TabICLv2:
# training-derived Youden operating threshold
OPERATING_THRESHOLD = 0.525


# ============================================================
# 1. Page configuration
# ============================================================

st.set_page_config(
    page_title="Postoperative mGPS=2 Risk Predictor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CSS
# ============================================================

st.markdown(
    dedent(
        """
        <style>
        html, body, [class*="css"] {
            font-family: 'Times New Roman', sans-serif;
        }

        div.stButton > button:first-child {
            background-color: #2E86C1;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 15px;
        }

        div.stButton > button:first-child:hover {
            background-color: #1B4F72;
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }

        [data-testid="stSidebar"] {
            background-color: #F8F9F9;
            border-right: 1px solid #E5E7E9;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2.8rem;
            color: #C0392B;
            font-weight: 900;
        }

        input[type="number"] {
            font-weight: bold;
            color: #154360;
            background-color: #F4F6F7;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .stDeployButton {
            display: none;
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# 3. Header
# ============================================================

col_logo, col_title = st.columns(
    [1, 8]
)


with col_logo:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3004/3004458.png",
        width=80,
    )


with col_title:

    st.title(
        "Preoperative Risk Assessment for "
        "Postoperative mGPS = 2 in Colorectal Cancer"
    )

    st.markdown(
        "**Research prototype powered by "
        "the TabICLv2 tabular foundation model**"
    )


# ============================================================
# Reviewer-requested prominent research-use warning
# ============================================================

st.error(
    """
⚠️ RESEARCH USE ONLY — NOT FOR CLINICAL DECISION-MAKING

This tool is a research prototype and has not been validated for clinical deployment.

Learning-curve analyses demonstrated suboptimal convergence, with relatively flat
cross-validation trajectories and persistent separation between training and
cross-validation performance. Stable and generalisable learning dynamics have
therefore not yet been established within the available development cohort.

This tool MUST NOT be used for clinical decision-making, patient care, treatment
selection, or any other real-world clinical purpose. Further model development,
independent external validation, and prospective evaluation are required before
clinical implementation can be considered.
"""
)


st.markdown(
    """<div style="background-color:#EBF5FB;padding:15px;border-radius:10px;border-left:5px solid #2980B9;margin-bottom:25px;color:#154360;font-size:15px;line-height:1.6;"><b>📊 System Introduction:</b><br><br>This research prototype integrates eight routinely available preoperative clinical indicators to estimate the probability of postoperative <b>modified Glasgow Prognostic Score (mGPS) = 2</b> in patients undergoing colorectal cancer surgery.<br><br>The platform provides TabICLv2-based risk estimation, SHAP-based local feature attribution, and an applicability-domain assessment indicating whether the patient's feature profile is adequately represented by the model-development cohort.<br><br><b>Research-use restriction:</b> The platform is intended exclusively for research and demonstration. It is not a validated clinical decision-support system and must not be used for clinical decision-making or patient care.</div>""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. Utility functions
# ============================================================

def get_base_dir():

    return os.path.dirname(
        os.path.abspath(__file__)
    )


def mean_vector_to_array(
    reference
):

    """
    Convert the development-cohort mean vector to a NumPy array
    in EXPECTED_FEATURES order.

    Supports:
    - list / ndarray
    - dict
    - pandas Series
    """

    mean_obj = reference[
        "mean_vector"
    ]


    if isinstance(
        mean_obj,
        pd.Series
    ):

        return (

            mean_obj
            .loc[
                EXPECTED_FEATURES
            ]
            .to_numpy(
                dtype=float
            )

        )


    if isinstance(
        mean_obj,
        dict
    ):

        return np.asarray(

            [
                mean_obj[
                    feature
                ]
                for feature
                in EXPECTED_FEATURES
            ],

            dtype=float,

        )


    return np.asarray(
        mean_obj,
        dtype=float,
    ).reshape(-1)


def get_reference_background(
    reference
):

    """
    Construct SHAP background using the mean feature profile
    of the model-development cohort.

    This replaces the previous manually specified hypothetical
    reference patient.
    """

    mean_vector = (
        mean_vector_to_array(
            reference
        )
    )


    if (
        len(mean_vector)
        != len(EXPECTED_FEATURES)
    ):

        raise ValueError(
            "Mean-vector dimension does not match "
            "the expected feature count."
        )


    background_df = pd.DataFrame(

        [
            mean_vector
        ],

        columns=EXPECTED_FEATURES,

    )


    return background_df


def normalize_shap_values(
    shap_values_raw,
    n_features,
):

    """
    Convert KernelExplainer output to a one-dimensional
    SHAP vector for one patient.
    """

    if isinstance(
        shap_values_raw,
        list
    ):

        if len(
            shap_values_raw
        ) == 0:

            raise ValueError(
                "SHAP returned an empty list."
            )


        arr = np.asarray(

            shap_values_raw[0],

            dtype=float,

        )


    else:

        arr = np.asarray(

            shap_values_raw,

            dtype=float,

        )


    arr = np.squeeze(
        arr
    )


    if (
        arr.ndim == 1
        and
        arr.shape[0] == n_features
    ):

        return arr


    flat = arr.reshape(-1)


    if (
        flat.shape[0]
        == n_features
    ):

        return flat


    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{np.asarray(shap_values_raw).shape}. "
        f"Expected {n_features} feature attributions."
    )


def normalize_expected_value(
    expected_value
):

    arr = np.asarray(

        expected_value,

        dtype=float,

    ).reshape(-1)


    if arr.size == 0:

        raise ValueError(
            "SHAP expected value is empty."
        )


    return float(
        arr[0]
    )


# ============================================================
# 5. Load TabICLv2 model
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(

        get_base_dir(),

        "tabicl_model.pkl",

    )


    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )


    with open(
        model_path,
        "rb",
    ) as f:

        loaded_model = pickle.load(
            f
        )


    return loaded_model


try:

    model = load_model()


except Exception as e:

    st.error(
        "🚨 Model loading failed. "
        f"Error details: {e}"
    )

    st.stop()


# ============================================================
# 6. Load applicability-domain reference
# ============================================================

@st.cache_resource
def load_applicability_domain():

    ad_path = os.path.join(

        get_base_dir(),

        "applicability_domain.pkl",

    )


    if not os.path.exists(
        ad_path
    ):

        raise FileNotFoundError(
            "Applicability-domain file not found: "
            f"{ad_path}"
        )


    with open(
        ad_path,
        "rb",
    ) as f:

        reference = pickle.load(
            f
        )


    return reference


try:

    ad_reference = (
        load_applicability_domain()
    )


except Exception as e:

    st.error(
        "🚨 Applicability-domain reference "
        f"loading failed: {e}"
    )

    st.stop()


# ============================================================
# 7. Validate applicability-domain file
# ============================================================

required_ad_keys = [

    "features",
    "n_training",
    "feature_min",
    "feature_max",
    "mean_vector",
    "inverse_covariance",
    "mahalanobis_p95",

]


missing_ad_keys = [

    key
    for key in required_ad_keys
    if key not in ad_reference

]


if missing_ad_keys:

    st.error(
        "Applicability-domain file is incomplete. "
        f"Missing fields: {missing_ad_keys}"
    )

    st.stop()


if (
    list(
        ad_reference[
            "features"
        ]
    )
    != EXPECTED_FEATURES
):

    st.error(
        "Applicability-domain feature order does not "
        "match the TabICLv2 model input order."
    )

    st.stop()


# ------------------------------------------------------------
# Validate numerical dimensions
# ------------------------------------------------------------

try:

    _mean_vector = (
        mean_vector_to_array(
            ad_reference
        )
    )


    _inverse_covariance = np.asarray(

        ad_reference[
            "inverse_covariance"
        ],

        dtype=float,

    )


    if (
        _mean_vector.shape
        != (
            len(EXPECTED_FEATURES),
        )
    ):

        raise ValueError(

            "mean_vector shape is "
            f"{_mean_vector.shape}; "

            "expected "
            f"({len(EXPECTED_FEATURES)},)."

        )


    if (

        _inverse_covariance.shape

        !=

        (
            len(EXPECTED_FEATURES),
            len(EXPECTED_FEATURES),
        )

    ):

        raise ValueError(

            "inverse_covariance shape is "
            f"{_inverse_covariance.shape}; "

            "expected "
            f"({len(EXPECTED_FEATURES)}, "
            f"{len(EXPECTED_FEATURES)})."

        )


except Exception as e:

    st.error(
        "Applicability-domain numerical "
        f"structure is invalid: {e}"
    )

    st.stop()


# ============================================================
# 8. Applicability-domain assessment
# ============================================================

def assess_applicability_domain(
    patient_df,
    reference,
):

    features = list(
        reference[
            "features"
        ]
    )


    patient = (

        patient_df[
            features
        ]

        .iloc[0]
        .astype(float)

    )


    # --------------------------------------------------------
    # 8.1 Individual feature range assessment
    # --------------------------------------------------------

    outside_features = []


    for feature in features:

        value = float(
            patient[
                feature
            ]
        )


        lower = float(

            reference[
                "feature_min"
            ][
                feature
            ]

        )


        upper = float(

            reference[
                "feature_max"
            ][
                feature
            ]

        )


        if (
            value < lower
            or
            value > upper
        ):

            outside_features.append(

                {

                    "feature":
                        feature,

                    "value":
                        value,

                    "min":
                        lower,

                    "max":
                        upper,

                }

            )


    # --------------------------------------------------------
    # 8.2 Multivariable Mahalanobis distance
    # --------------------------------------------------------

    x_vector = (

        patient[
            features
        ]

        .to_numpy(
            dtype=float
        )

    )


    mean_vector = (
        mean_vector_to_array(
            reference
        )
    )


    inverse_covariance = np.asarray(

        reference[
            "inverse_covariance"
        ],

        dtype=float,

    )


    delta = (
        x_vector
        -
        mean_vector
    )


    mahalanobis_sq = float(

        delta.T
        @
        inverse_covariance
        @
        delta

    )


    mahalanobis_distance = float(

        np.sqrt(

            max(
                mahalanobis_sq,
                0.0,
            )

        )

    )


    distance_threshold = float(

        reference[
            "mahalanobis_p95"
        ]

    )


    multivariate_outlier = (

        mahalanobis_distance
        >
        distance_threshold

    )


    limited_support = (

        len(
            outside_features
        )
        > 0

        or

        multivariate_outlier

    )


    return {

        "limited_support":
            limited_support,

        "outside_features":
            outside_features,

        "multivariate_outlier":
            multivariate_outlier,

        "distance":
            mahalanobis_distance,

        "distance_threshold":
            distance_threshold,

    }


# ============================================================
# 9. Default example values
# ============================================================

default_values = {

    "PA":
        55.0,

    "Age":
        76.5,

    "Fbg":
        3.37,

    "ALB":
        26.3,

    "ChE":
        1218.0,

    "Lymph_pct":
        17.5,

    "PLT":
        253.0,

    "Ca":
        2.02,

}


for key, val in (
    default_values.items()
):

    slider_key = (
        f"{key}_slider"
    )

    number_key = (
        f"{key}_num"
    )


    if (
        slider_key
        not in st.session_state
    ):

        st.session_state[
            slider_key
        ] = val


    if (
        number_key
        not in st.session_state
    ):

        st.session_state[
            number_key
        ] = val


# ============================================================
# 10. Synchronise sidebar sliders and number inputs
# ============================================================

def sync_inputs(
    source_key,
    destination_key,
):

    st.session_state[
        destination_key
    ] = st.session_state[
        source_key
    ]


# ============================================================
# 11. Sidebar
# ============================================================

st.sidebar.markdown(
    "### 🖥️ System Status"
)


st.sidebar.success(
    "🟢 Core Engine: TabICLv2 Ready"
)


st.sidebar.success(
    "🟢 Applicability Domain: Ready"
)


st.sidebar.caption(
    "Development-cohort reference: "
    f"N = {ad_reference['n_training']}"
)


st.sidebar.markdown(
    "---"
)


st.sidebar.markdown(
    "### 🎛️ Rapid Parameter Adjustment"
)


# ------------------------------------------------------------
# Demographics & Nutrition
# ------------------------------------------------------------

with st.sidebar.expander(
    "👤 Demographics & Nutrition",
    expanded=True,
):

    st.slider(
        "Age (Years)",
        min_value=18.0,
        max_value=100.0,
        step=0.5,
        key="Age_slider",
        on_change=sync_inputs,
        args=(
            "Age_slider",
            "Age_num",
        ),
    )


    st.slider(
        "Prealbumin (PA) mg/L",
        min_value=10.0,
        max_value=500.0,
        step=1.0,
        key="PA_slider",
        on_change=sync_inputs,
        args=(
            "PA_slider",
            "PA_num",
        ),
    )


    st.slider(
        "Albumin (ALB) g/L",
        min_value=10.0,
        max_value=60.0,
        step=0.1,
        key="ALB_slider",
        on_change=sync_inputs,
        args=(
            "ALB_slider",
            "ALB_num",
        ),
    )


    st.slider(
        "Cholinesterase (ChE) U/L",
        min_value=100.0,
        max_value=18000.0,
        step=1.0,
        key="ChE_slider",
        on_change=sync_inputs,
        args=(
            "ChE_slider",
            "ChE_num",
        ),
    )


# ------------------------------------------------------------
# Immuno-coagulation
# ------------------------------------------------------------

with st.sidebar.expander(
    "🩸 Immuno-coagulation Profile",
    expanded=True,
):

    st.slider(
        "Lymphocyte Percentage (Lymph%)",
        min_value=1.0,
        max_value=60.0,
        step=0.1,
        key="Lymph_pct_slider",
        on_change=sync_inputs,
        args=(
            "Lymph_pct_slider",
            "Lymph_pct_num",
        ),
    )


    st.slider(
        "Platelets (PLT) ×10⁹/L",
        min_value=20.0,
        max_value=800.0,
        step=1.0,
        key="PLT_slider",
        on_change=sync_inputs,
        args=(
            "PLT_slider",
            "PLT_num",
        ),
    )


    st.slider(
        "Fibrinogen (Fbg) g/L",
        min_value=1.0,
        max_value=10.0,
        step=0.01,
        key="Fbg_slider",
        on_change=sync_inputs,
        args=(
            "Fbg_slider",
            "Fbg_num",
        ),
    )


    st.slider(
        "Serum Calcium (Ca) mmol/L",
        min_value=1.50,
        max_value=3.00,
        step=0.01,
        key="Ca_slider",
        on_change=sync_inputs,
        args=(
            "Ca_slider",
            "Ca_num",
        ),
    )


# ============================================================
# 12. Main clinical input matrix
# ============================================================

st.markdown(
    "### 🧪 Research Parameter Input Matrix"
)


st.markdown(
    "*Enter exact values below, or use the sidebar sliders "
    "to adjust values synchronously.*"
)


col1, col2, col3, col4 = (
    st.columns(4)
)


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

with col1:

    st.number_input(
        "Age (Years)",
        min_value=18.0,
        max_value=100.0,
        step=0.5,
        format="%.1f",
        key="Age_num",
        on_change=sync_inputs,
        args=(
            "Age_num",
            "Age_slider",
        ),
    )


    st.number_input(
        "PA (mg/L)",
        min_value=10.0,
        max_value=500.0,
        step=1.0,
        format="%.1f",
        key="PA_num",
        on_change=sync_inputs,
        args=(
            "PA_num",
            "PA_slider",
        ),
    )


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    st.number_input(
        "ALB (g/L)",
        min_value=10.0,
        max_value=60.0,
        step=0.1,
        format="%.1f",
        key="ALB_num",
        on_change=sync_inputs,
        args=(
            "ALB_num",
            "ALB_slider",
        ),
    )


    st.number_input(
        "ChE (U/L)",
        min_value=100.0,
        max_value=18000.0,
        step=1.0,
        format="%.0f",
        key="ChE_num",
        on_change=sync_inputs,
        args=(
            "ChE_num",
            "ChE_slider",
        ),
    )


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    st.number_input(
        "Lymph (%)",
        min_value=1.0,
        max_value=60.0,
        step=0.1,
        format="%.1f",
        key="Lymph_pct_num",
        on_change=sync_inputs,
        args=(
            "Lymph_pct_num",
            "Lymph_pct_slider",
        ),
    )


    st.number_input(
        "PLT (×10⁹/L)",
        min_value=20.0,
        max_value=800.0,
        step=1.0,
        format="%.0f",
        key="PLT_num",
        on_change=sync_inputs,
        args=(
            "PLT_num",
            "PLT_slider",
        ),
    )


# ------------------------------------------------------------
# Column 4
# ------------------------------------------------------------

with col4:

    st.number_input(
        "Fbg (g/L)",
        min_value=1.0,
        max_value=10.0,
        step=0.01,
        format="%.2f",
        key="Fbg_num",
        on_change=sync_inputs,
        args=(
            "Fbg_num",
            "Fbg_slider",
        ),
    )


    st.number_input(
        "Ca (mmol/L)",
        min_value=1.50,
        max_value=3.00,
        step=0.01,
        format="%.2f",
        key="Ca_num",
        on_change=sync_inputs,
        args=(
            "Ca_num",
            "Ca_slider",
        ),
    )


# ============================================================
# 13. Build TabICLv2 input
# ============================================================

input_df = pd.DataFrame(

    {

        "PA": [
            st.session_state[
                "PA_num"
            ]
        ],

        "Age": [
            st.session_state[
                "Age_num"
            ]
        ],

        "Fbg": [
            st.session_state[
                "Fbg_num"
            ]
        ],

        "ALB": [
            st.session_state[
                "ALB_num"
            ]
        ],

        "ChE": [
            st.session_state[
                "ChE_num"
            ]
        ],

        "Lymph%": [
            st.session_state[
                "Lymph_pct_num"
            ]
        ],

        "PLT": [
            st.session_state[
                "PLT_num"
            ]
        ],

        "Ca": [
            st.session_state[
                "Ca_num"
            ]
        ],

    }

)


input_df = input_df[
    EXPECTED_FEATURES
]


# ============================================================
# 14. Prediction wrapper used by SHAP
# ============================================================

def predict_positive_prob(
    X
):

    if isinstance(
        X,
        pd.DataFrame
    ):

        X_for_model = (
            X.copy()
        )

        X_for_model = (
            X_for_model[
                EXPECTED_FEATURES
            ]
        )


    else:

        X_for_model = pd.DataFrame(

            X,

            columns=EXPECTED_FEATURES,

        )


    probabilities = (

        model.predict_proba(
            X_for_model
        )[:, 1]

    )


    return np.asarray(
        probabilities,
        dtype=float,
    )


# ============================================================
# 15. Risk assessment
# ============================================================

if st.button(
    "🚀 Run Risk Assessment",
    type="primary",
):

    with st.spinner(
        "🧬 The model is analysing "
        "the clinical feature profile..."
    ):


        # ====================================================
        # 15.1 Applicability-domain assessment
        # ====================================================

        ad_result = assess_applicability_domain(

            input_df,
            ad_reference,

        )


        # ====================================================
        # 15.2 TabICLv2 probability
        # ====================================================

        risk_prob = float(

            model.predict_proba(
                input_df
            )[0][1]

        )


        # ====================================================
        # 16. Applicability-domain report
        # ====================================================

        st.markdown(
            "---"
        )


        st.markdown(
            "### 🔎 Model Applicability Check"
        )


        # ----------------------------------------------------
        # Limited support
        # ----------------------------------------------------

        if ad_result[
            "limited_support"
        ]:

            st.warning(
                "⚠️ **Limited training-data support.** "
                "This patient's feature profile is uncommon "
                "relative to the model-development cohort. "
                "The predicted probability should therefore "
                "be interpreted with additional caution."
            )


            if ad_result[
                "outside_features"
            ]:

                st.markdown(
                    "**Input values outside the observed "
                    "training range:**"
                )


                for item in ad_result[
                    "outside_features"
                ]:

                    st.write(

                        f"- **{item['feature']}**: "
                        f"{item['value']:.3f} "
                        f"(observed training range: "
                        f"{item['min']:.3f}–"
                        f"{item['max']:.3f})"

                    )


            if ad_result[
                "multivariate_outlier"
            ]:

                st.warning(
                    "The overall combination of the eight "
                    "predictors is also relatively uncommon "
                    "compared with the model-development cohort."
                )


        else:

            st.success(
                "✅ **Feature profile is within the "
                "predefined applicability domain.** "
                "The individual values are within the "
                "observed development-cohort ranges and "
                "the multivariable profile does not exceed "
                "the predefined distance threshold."
            )


        # ====================================================
        # Applicability-domain technical details
        # ====================================================

        with st.expander(
            "Applicability-domain details"
        ):

            st.write(
                "**Development cohort:**",
                f"N = {ad_reference['n_training']}",
            )


            st.write(
                "**Patient Mahalanobis distance:**",
                f"{ad_result['distance']:.3f}",
            )


            st.write(
                "**Training 95th-percentile threshold:**",
                f"{ad_result['distance_threshold']:.3f}",
            )


            if (
                ad_result[
                    "distance"
                ]
                <=
                ad_result[
                    "distance_threshold"
                ]
            ):

                st.write(
                    "**Multivariable status:** "
                    "Within reference threshold"
                )


            else:

                st.write(
                    "**Multivariable status:** "
                    "Above reference threshold"
                )


        st.caption(
            "The applicability-domain assessment reflects "
            "similarity to the model-development population. "
            "It is not a formal predictive uncertainty interval "
            "and does not guarantee prediction accuracy."
        )


        # ====================================================
        # 17. Risk inference
        # ====================================================

        st.markdown(
            "---"
        )


        st.markdown(
            "### 🎯 Postoperative Risk Inference Report"
        )


        res_col1, res_col2 = (
            st.columns(
                [1, 2]
            )
        )


        with res_col1:

            st.metric(

                label=(
                    "Predicted probability of "
                    "postoperative mGPS = 2"
                ),

                value=(
                    f"{risk_prob * 100:.2f}%"
                ),

            )


        with res_col2:

            st.markdown(
                "<br>",
                unsafe_allow_html=True,
            )


            if (
                risk_prob
                >= OPERATING_THRESHOLD
            ):

                st.warning(

                    "⚠️ **Higher-risk stratum (research output only).** "
                    "The predicted probability is above the "
                    "training-derived operating threshold "
                    f"({OPERATING_THRESHOLD:.3f}). "

                    "This output is provided solely for research and "
                    "demonstration. It must not be used for clinical "
                    "decision-making, patient care, treatment selection, "
                    "or other real-world clinical purposes."

                )


            else:

                st.info(

                    "ℹ️ **Lower-risk stratum (research output only).** "
                    "The predicted probability is below the "
                    "training-derived operating threshold "
                    f"({OPERATING_THRESHOLD:.3f}). "

                    "A lower model-estimated risk does not exclude "
                    "postoperative inflammatory–nutritional deterioration. "
                    "This output is provided solely for research and "
                    "demonstration and must not be used for clinical "
                    "decision-making or patient care."

                )


        st.caption(
            "The operating threshold of 0.525 was derived "
            "from out-of-fold predictions in the training cohort "
            "by maximising the Youden index."
        )


        # ====================================================
        # 18. SHAP interpretation
        # ====================================================

        st.markdown(
            "<br>",
            unsafe_allow_html=True,
        )


        st.markdown(
            "### 🧠 Risk Factor Attribution "
            "(Real-time SHAP)"
        )


        st.info(

            "💡 **Interpretation Guide:** "
            "SHAP values describe how each feature shifts the "
            "model output away from the reference prediction. "
            "In this implementation, SHAP is calculated on the "
            "positive-class probability scale using the mean "
            "feature profile of the model-development cohort "
            "as the reference. Positive values increase the "
            "model-estimated probability and negative values "
            "decrease it. Individual SHAP values are "
            "contributions to the prediction, not independent "
            "probabilities."

        )


        try:


            # =================================================
            # 18.1 Development-cohort SHAP reference
            # =================================================

            background_df = (
                get_reference_background(
                    ad_reference
                )
            )


            # =================================================
            # 18.2 Kernel SHAP
            #
            # identity link:
            # SHAP decomposition remains on probability scale
            # =================================================

            explainer = (

                shap.KernelExplainer(

                    predict_positive_prob,

                    background_df,

                    link="identity",

                )

            )


            shap_values_raw = (

                explainer.shap_values(

                    input_df,

                    silent=True,

                )

            )


            # =================================================
            # 18.3 Normalize SHAP output
            # =================================================

            shap_val_single = (
                normalize_shap_values(

                    shap_values_raw,

                    n_features=(
                        len(
                            EXPECTED_FEATURES
                        )
                    ),

                )
            )


            # =================================================
            # 18.4 SHAP expected value
            # =================================================

            base_val = (
                normalize_expected_value(

                    explainer.expected_value

                )
            )


            # =================================================
            # 18.5 SHAP Explanation
            # =================================================

            explanation = shap.Explanation(

                values=(
                    shap_val_single
                ),

                base_values=(
                    base_val
                ),

                data=(

                    input_df
                    .iloc[0]
                    .to_numpy(
                        dtype=float
                    )

                ),

                feature_names=(
                    EXPECTED_FEATURES
                ),

            )


            # =================================================
            # 18.6 Numerical reconstruction check
            # =================================================

            reconstructed_prob = float(

                base_val
                +
                shap_val_single.sum()

            )


            reconstruction_error = abs(

                reconstructed_prob
                -
                risk_prob

            )


            with st.expander(
                "SHAP calculation details"
            ):

                st.write(
                    "**Reference profile:** "
                    "Development-cohort mean vector"
                )


                st.write(
                    "**Reference prediction:**",
                    f"{base_val:.6f}",
                )


                st.write(
                    "**SHAP-reconstructed patient prediction:**",
                    f"{reconstructed_prob:.6f}",
                )


                st.write(
                    "**Direct model prediction:**",
                    f"{risk_prob:.6f}",
                )


                st.write(
                    "**Absolute reconstruction difference:**",
                    f"{reconstruction_error:.6e}",
                )


            # =================================================
            # 18.7 Tabs
            # =================================================

            tab1, tab2, tab3, tab4 = (

                st.tabs(

                    [

                        "🌊 Waterfall Plot",

                        "⚖️ Force Plot",

                        "📈 Decision Plot",

                        "📊 Bar Plot",

                    ]

                )

            )


            # =================================================
            # Waterfall plot
            # =================================================

            with tab1:

                st.markdown(
                    "#### 1. Local Waterfall Plot"
                )


                plt.figure(
                    figsize=(10, 6)
                )


                shap.plots.waterfall(

                    explanation,

                    max_display=10,

                    show=False,

                )


                fig = plt.gcf()


                st.pyplot(

                    fig,

                    bbox_inches="tight",

                )


                plt.close(
                    fig
                )


            # =================================================
            # Force plot
            # =================================================

            with tab2:

                st.markdown(
                    "#### 2. Local Force Plot"
                )


                plt.figure(
                    figsize=(11, 4)
                )


                shap.force_plot(

                    base_val,

                    shap_val_single,

                    input_df.iloc[0],

                    matplotlib=True,

                    show=False,

                )


                fig = plt.gcf()


                st.pyplot(

                    fig,

                    bbox_inches="tight",

                )


                plt.close(
                    fig
                )


            # =================================================
            # Decision plot
            # =================================================

            with tab3:

                st.markdown(
                    "#### 3. Decision Plot"
                )


                plt.figure(
                    figsize=(10, 6)
                )


                shap.decision_plot(

                    base_val,

                    shap_val_single,

                    input_df.iloc[0],

                    feature_names=(
                        EXPECTED_FEATURES
                    ),

                    show=False,

                )


                fig = plt.gcf()


                st.pyplot(

                    fig,

                    bbox_inches="tight",

                )


                plt.close(
                    fig
                )


            # =================================================
            # SHAP bar plot
            # =================================================

            with tab4:

                st.markdown(
                    "#### 4. Local SHAP Impact Bar Plot"
                )


                plt.figure(
                    figsize=(10, 6)
                )


                shap.plots.bar(

                    explanation,

                    max_display=10,

                    show=False,

                )


                fig = plt.gcf()


                st.pyplot(

                    fig,

                    bbox_inches="tight",

                )


                plt.close(
                    fig
                )


        except Exception as e:

            st.error(

                "An error occurred while generating "
                f"the SHAP plots: {e}"

            )


# ============================================================
# 19. Research disclaimer
# ============================================================

st.markdown(
    "---"
)


st.error(
    """
⚠️ FINAL RESEARCH-USE DISCLAIMER

This calculator is a research prototype developed from a single-centre
retrospective cohort. Learning-curve analysis demonstrated suboptimal
convergence, and stable, generalisable learning dynamics have not yet been
established.

The calculator is not a validated medical device or clinical decision-support
system and MUST NOT be used for clinical decision-making, patient care,
treatment selection, or other real-world clinical purposes.

Further model development, independent external validation, and prospective
evaluation are required before clinical implementation can be considered.
No patient-identifiable information should be entered into this platform.
"""
)
