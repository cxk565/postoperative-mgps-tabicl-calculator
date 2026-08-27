# Postoperative mGPS TabICLv2 Calculator

This repository contains the source code for the web-based research prototype
described in:

"Preoperative Risk Stratification for Postoperative Inflammatory–Nutritional
Deterioration After Colorectal Cancer Surgery: An Interpretable Machine
Learning Study"

## Overview

The prototype uses eight routinely available preoperative predictors:

- Prealbumin (PA)
- Age
- Fibrinogen (Fbg)
- Albumin (ALB)
- Cholinesterase (ChE)
- Lymphocyte percentage (Lymph%)
- Platelet count (PLT)
- Serum calcium (Ca)

The application provides:

- TabICLv2-based risk estimation
- Patient-level SHAP interpretation
- Applicability-domain assessment based on observed training ranges and
  Mahalanobis distance

## Important notice

This application is a research prototype and is not a validated clinical
decision-support system. Model outputs should not replace independent
clinical judgement.

## Data availability

Patient-level clinical data are not included in this repository because of
institutional data-protection and patient-privacy requirements.

## Model files

Serialized fitted model objects are not publicly distributed because they may
retain information derived from the model-development cohort.

## Web application

https://mgps-tabicl-model.streamlit.app

## Installation

```bash
pip install -r requirements.txt
