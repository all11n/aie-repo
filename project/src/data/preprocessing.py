# src/data/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

FEATURE_NAMES = [
    'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active'
]

DISEASE_LABELS = [
    'ischemic_heart', 'hypertensive_heart', 'heart_failure', 
    'pericardial_disease', 'heart_tumor'
]

def preprocess_features(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler = None) -> Tuple[np.ndarray, StandardScaler]:
    features = df[FEATURE_NAMES].copy()
    if fit_scaler:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    return features_scaled, scaler

def get_multilabel_targets(df: pd.DataFrame) -> np.ndarray:
    return df[DISEASE_LABELS].values

def convert_to_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new['age_years'] = (df_new['age'] / 365.25).round(1)
    
    df_new['ischemic_heart'] = ((df_new['age_years'] > 50) & (df_new['ap_hi'] > 140) & (df_new['cholesterol'] > 1) & (df_new['cardio'] == 1)).astype(int)
    df_new['hypertensive_heart'] = ((df_new['ap_hi'] > 160) & (df_new['cardio'] == 1)).astype(int)
    df_new['heart_failure'] = ((df_new['age_years'] > 60) & (df_new['ap_hi'] > 140) & (df_new['cardio'] == 1)).astype(int)
    df_new['pericardial_disease'] = ((df_new['age_years'] > 40) & (df_new['cardio'] == 1) & (np.random.random(len(df_new)) < 0.05)).astype(int)
    df_new['heart_tumor'] = ((df_new['age_years'] > 50) & (df_new['cardio'] == 1) & (np.random.random(len(df_new)) < 0.01)).astype(int)
    
    return df_new
TARGET_NAMES = DISEASE_LABELS