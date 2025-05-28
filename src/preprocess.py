# Import Dependices
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder





class DataProcessor:
    """
    Handle data preprocessing and cleaning for full reports
    
    
    """

    def __init__(self):
        self.label_encoder = LabelEncoder()

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data
        
        """
        
        
        if pd.isna(text):
            return ""

        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.,;:()!?-]', ' ', text)

        # Remove standalone numbers (but keep years and case numbers in context)
        text = re.sub(r'\b\d{1,2}\b', ' ', text)  # Remove 1-2 digit numbers

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_law_area(self, text: str) -> str:
        
        """
        
        Extract area of law from introduction text
        
        """
        
        if pd.isna(text):
            return "unknown"

        text = str(text).lower()

        # Define law area
        area_labels = {
            'Criminal Law and Procedure': [
                'criminal', 'murder', 'theft', 'assault', 'robbery', 'fraud', 'drug',
                 'criminal procedure', 'investigation','bail', 'arrest', 'charge', 'cognizable', 'non-cognizable',
                'magistrate', 'trial', 'conviction', 'acquitted'
            ],
            'Civil Procedure': [
                'civil', 'suit', 'plaint', 'written statement', 'civil procedure code',
                'cpc', 'decree', 'judgment', 'execution', 'appeal', 'revision',
                'injunction', 'specific performance', 'damages', 'contract', 'tort',
                'negligence', 'breach', 'civil court', 'district court', 'plaintiff',
                'defendant', 'civil appeal', 'civil revision', "application"
            ],
            'Enforcement of Fundamental Rights': [
                'fundamental rights', 'constitution', 'constitutional', 'writ',
                'article 32', 'article 226', 'supreme court', 'high court',
                'constitutional validity', 'fundamental right', 'life and liberty',
                'equality', 'freedom', 'right to', 'constitutional law',
                'judicial review', 'constitutional petition'
            ],
            'Company Law': [
                'company', 'corporate', 'companies act', 'director', 'shareholder',
                'board of directors', 'annual general meeting', 'agm', 'share',
                'dividend', 'corporate governance', 'company law', 'incorporation',
                'winding up', 'liquidation', 'merger', 'acquisition', 'nclt',
                'company law tribunal', 'corporate affairs', 'securities',
                'company secretary', 'compliance', 'corporate social responsibility'
            ]
        }

        # Count matches for each area
        area_num = {}
        for area, words in area_labels.items():
            num = 0
            for word in words:
                if (word in text) or (word in area_labels.keys()):
                    num += 3

            if num > 0:
                area_num[area] = num

        # Return area with highest score, or 'Unknown' if no matches
        if area_num:
                return max(area_num.items(), key=lambda x: x[1])[0]
        else:
            return "Unknown"


    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the entire dataset
        
        """
        
        print("Processing report...")

        # Clean full_report column
        df['cleaned_report'] = df['full_report'].apply(self.clean_text)

        # Extract law areas from introduction
        df['law_area'] = df['introduction'].apply(self.extract_law_area)

        # Remove rows with empty cleaned reports
        df = df[df['cleaned_report'].str.len() > 10].reset_index(drop=True)

        print(f"Dataset processed. Shape: {df.shape}")
        print(f"Law areas distribution:\n{df['law_area'].value_counts()}")

        return df
