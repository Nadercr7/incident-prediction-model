"""
Data Loader Module
==================
Handles all data loading, cleaning, and preprocessing operations.

Pipeline:
    1. Load CSV data
    2. Standardize column names
    3. Parse dates and extract temporal features
    4. Create binary indicator flags
    5. Remove duplicates and handle missing values

Author: Nader Mohamed
Date: February 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class IncidentDataLoader:
    """
    Load and preprocess incident data for prediction modeling.

    Expects a CSV with at minimum:
        - g_date: incident date (YYYY/MM/DD)
        - location_name: location identifier

    Optional columns (used for binary flags):
        - injury_type_name
        - vehicle_damage_name
        - vehicle_collision_type_name
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df_raw = None
        self.df_clean = None

    # ------------------------------------------------------------------
    # Core pipeline steps
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        """Load raw CSV data."""
        self.df_raw = pd.read_csv(self.filepath)
        print(f"[+] Loaded {len(self.df_raw):,} rows from {self.filepath}")
        return self.df_raw

    def clean_data(self) -> pd.DataFrame:
        """
        Standardize columns, parse dates, drop rows missing
        critical fields, fill categorical NaNs, remove duplicates.
        """
        if self.df_raw is None:
            raise ValueError("Call load_data() first.")

        print("[*] Cleaning data...")
        df = self.df_raw.copy()

        # Standardize column names
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Parse date
        df['g_date'] = pd.to_datetime(df['g_date'], format='%Y/%m/%d', errors='coerce')

        # Drop rows missing date or location
        initial = len(df)
        df = df.dropna(subset=['g_date', 'location_name'])
        print(f"    Dropped {initial - len(df)} rows with missing date/location")

        # Fill missing categoricals
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')

        # Remove duplicates
        dupes = df.duplicated().sum()
        df = df.drop_duplicates()
        if dupes:
            print(f"    Removed {dupes} duplicate rows")

        self.df_clean = df
        print(f"[+] Clean data: {len(df):,} rows")
        return df

    def extract_temporal_features(self) -> pd.DataFrame:
        """Extract year, month, quarter, day_of_week, week_of_year from g_date."""
        if self.df_clean is None:
            raise ValueError("Call clean_data() first.")

        df = self.df_clean
        df['year'] = df['g_date'].dt.year
        df['month'] = df['g_date'].dt.month
        df['day'] = df['g_date'].dt.day
        df['day_of_week'] = df['g_date'].dt.dayofweek
        df['quarter'] = df['g_date'].dt.quarter
        df['week_of_year'] = df['g_date'].dt.isocalendar().week.astype(int)

        self.df_clean = df
        print("[+] Extracted temporal features")
        return df

    def create_binary_flags(self) -> pd.DataFrame:
        """
        Create binary flags for injury, vehicle damage, and collision
        (gracefully skips if columns are missing).
        """
        if self.df_clean is None:
            raise ValueError("Call clean_data() first.")

        df = self.df_clean

        if 'injury_type_name' in df.columns:
            df['has_injury'] = (df['injury_type_name'] != 'Unknown').astype(int)
        else:
            df['has_injury'] = 0

        if 'vehicle_damage_name' in df.columns:
            df['has_vehicle_damage'] = df['vehicle_damage_name'].apply(
                lambda x: 0 if x in ('Unknown', 'No') else 1
            )
        else:
            df['has_vehicle_damage'] = 0

        if 'vehicle_collision_type_name' in df.columns:
            df['is_collision'] = df['vehicle_collision_type_name'].apply(
                lambda x: 1 if 'Contact with moving vehicle' in str(x) else 0
            )
        else:
            df['is_collision'] = 0

        self.df_clean = df
        print("[+] Created binary flags")
        return df

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """Return (min_date, max_date) of clean data."""
        if self.df_clean is None:
            raise ValueError("Call clean_data() first.")
        return (self.df_clean['g_date'].min(), self.df_clean['g_date'].max())

    def get_location_counts(self) -> pd.Series:
        """Incident counts per location."""
        if self.df_clean is None:
            raise ValueError("Call clean_data() first.")
        return self.df_clean['location_name'].value_counts()

    def get_yearly_counts(self) -> pd.Series:
        """Incident counts per year."""
        if self.df_clean is None:
            raise ValueError("Call clean_data() first.")
        return self.df_clean.groupby('year').size()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_all(self) -> pd.DataFrame:
        """Run complete preprocessing pipeline and return clean DataFrame."""
        self.load_data()
        self.clean_data()
        self.extract_temporal_features()
        self.create_binary_flags()
        return self.df_clean


# Convenience function
def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """One-liner to load + preprocess incident data."""
    return IncidentDataLoader(filepath).process_all()
