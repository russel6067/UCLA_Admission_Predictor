import pandas as pd
import os

def load_data(filepath='data/Admission.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found.")
    
    df = pd.read_csv(filepath)
    
    if 'Admit_Chance' not in df.columns:
        raise ValueError("Missing 'Admit_Chance' column.")
    
    return df.drop(columns=['Serial_No'], errors='ignore')