"""
Data Loader — matches notebook cells 1-5 exactly.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class IncidentDataLoader:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df_raw = None
        self.df_clean = None

    def load_data(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.filepath)
        print(f"[+] Loaded {len(self.df_raw):,} rows from {self.filepath}")
        return self.df_raw

    def clean_data(self) -> pd.DataFrame:
        if self.df_raw is None:
            raise ValueError("Call load_data() first.")

        print("[*] Cleaning data...")
        df = self.df_raw.copy()

        # Standardize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Rename column if needed (April 2026 data export)
        if 'injury_description_name' in df.columns and 'injury_type_name' not in df.columns:
            df.rename(columns={'injury_description_name': 'injury_type_name'}, inplace=True)
            print("    Renamed 'injury_description_name' → 'injury_type_name'")

        # Parse dates
        df['g_date'] = pd.to_datetime(df['g_date'], format='%Y/%m/%d', errors='coerce')

        missing_dates = df['g_date'].isnull().sum()
        print(f"    Rows with missing dates: {missing_dates} ({missing_dates / len(df) * 100:.1f}%)")

        # Drop rows without dates or location (required for time-series)
        df = df.dropna(subset=['g_date'])
        df = df.dropna(subset=['location_name'])

        # Fill categorical NaNs with business-logic defaults
        df['injury_type_name'] = df['injury_type_name'].fillna('No Injury')
        df['vehicle_damage_name'] = df['vehicle_damage_name'].fillna('No Damage')
        df['vehicle_collision_type_name'] = df['vehicle_collision_type_name'].fillna('No Collision')

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('Unknown')

        # Extract temporal features
        df['year'] = df['g_date'].dt.year
        df['month'] = df['g_date'].dt.month
        df['quarter'] = df['g_date'].dt.quarter

        self.df_clean = df
        print(f"[+] Clean data: {len(df):,} rows")
        return df

    def filter_years(self, min_year: int = 2023, max_year: int = 2025) -> pd.DataFrame:
        if self.df_clean is None:
            raise ValueError("Call clean_data() first.")

        yearly = self.df_clean.groupby('year').size()
        print("\nIncidents by Year:")
        for year, count in yearly.items():
            print(f"  {year}: {count:,} records")

        self.df_clean = self.df_clean[
            (self.df_clean['year'] >= min_year) & (self.df_clean['year'] <= max_year)
        ]
        print(f"\nFiltered to {min_year}-{max_year}: {len(self.df_clean):,} rows")
        return self.df_clean

    def process_all(self, min_year: int = 2023, max_year: int = 2025) -> pd.DataFrame:
        self.load_data()
        self.clean_data()
        self.filter_years(min_year, max_year)
        return self.df_clean
