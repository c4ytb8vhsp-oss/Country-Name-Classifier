import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import random
from pathlib import Path
from config import *
import pycountry

class CountryDataPreparation:
    def __init__(self, csv_path, corporate_data_path=None):
        """
        Initialize with path to country aliases CSV and optional corporate data
        
        Args:
            csv_path: Path to country aliases CSV
            corporate_data_path: Path to corporate data (entities with country names that are NOT countries)
        """
        self.csv_path = csv_path
        self.corporate_data_path = corporate_data_path
        self.country_names = []
        self.corporate_entities = []
        
    def load_country_aliases(self):
        """Load country aliases from CSV"""
        df = pd.read_csv(self.csv_path)
        
        # Extract all country variations
        country_names = []
        
        # Adjust based on actual CSV structure
        # Filter rows where 'AliasDescription' contains 'English' (case-insensitive)
        filtered_df = df[df['AliasDescription'].str.contains('English', case=False, na=False)]

        # Add values from the 'Alias' column only
        country_names.extend(filtered_df['Alias'].dropna().unique().tolist())
        
        # Add pycountry data for official names
        try:
            for country in pycountry.countries:
                country_names.append(country.name)
                if hasattr(country, 'official_name'):
                    country_names.append(country.official_name)
        except ImportError:
            print("Note: pycountry not installed. Install with: pip install pycountry")
        
        # Clean and deduplicate
        self.country_names = list(set([name.strip() for name in country_names if name and len(name) > 2]))
        
        print(f"Loaded {len(self.country_names)} unique country name variations")
        return self.country_names
    
    def load_corporate_data(self):
        """
        Load corporate data - entities with country names that are NOT countries
        
        Your corporate_data should contain:
        - Financial instruments: "Korea Bond", "US Treasury", etc.
        - Organizations: "Bank of Korea", etc.
        - All labeled as NOT COUNTRY
        """
        if self.corporate_data_path is None:
            print("No corporate data provided - will use only synthetic examples")
            return []
        
        from pathlib import Path
        path = Path(self.corporate_data_path)
        
        if not path.exists():
            print(f"Warning: Corporate data not found at {path}")
            return []
        
        # Load based on extension
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            print(f"Error: Unsupported format {path.suffix}")
            return []
        
        print(f"Loaded {len(df)} records from corporate data")
        
        # Extract entity names
        if 'entity_name' in df.columns:
            entities = df['entity_name'].dropna().unique().tolist()
        elif 'name' in df.columns:
            entities = df['name'].dropna().unique().tolist()
        else:
            entities = df.iloc[:, 0].dropna().unique().tolist()
        
        self.corporate_entities = [str(e).strip() for e in entities if e and str(e).strip()]
        
        print(f"✅ Extracted {len(self.corporate_entities)} real corporate entities")
        print(f"   Sample: {self.corporate_entities[:3]}")
        
        return self.corporate_entities
    
    def generate_country_variations(self):
        """Generate additional variations of country names"""
        variations = []
        prefixes = ["State of", "Republic of", "Kingdom of", "Democratic Republic of",
                   "People's Republic of", "Federation of", "Commonwealth of",
                   "Sultanate of", "Emirate of", "Principality of"]
        
        suffixes = ["Region", "Territory", "Province"]
        
        for country in self.country_names:
            # Original name
            variations.append(country)
            
            # Add prefix variations
            for prefix in prefixes:
                variations.append(f"{prefix} {country}")
            
            # Lowercase variations
            variations.append(country.lower())
            
            # Variations with "the"
            if not country.lower().startswith("the "):
                variations.append(f"the {country}")
                variations.append(f"The {country}")
        
        return list(set(variations))
    
    def generate_negative_examples(self, n_samples=None):
        """Generate non-country examples using country names"""
        if n_samples is None:
            n_samples = len(self.country_names)
        
        negative_examples = []
        
        # Patterns that indicate it's NOT a country
        templates = [
            "Bank of {country}",
            "{country} Bank",
            "University of {country}",
            "{country} University",
            "Embassy of {country}",
            "{country} Embassy",
            "{country} Airlines",
            "{country} Airport",
            "Ministry of {country}",
            "{country} Ministry",
            "Hotel {country}",
            "{country} Hotel",
            "Restaurant {country}",
            "{country} Museum",
            "{country} Corporation",
            "{country} Company",
            "{country} Inc",
            "{country} Ltd",
            "Department of {country}",
            "{country} Foundation",
            "{country} Institute",
            "{country} Library",
            "{country} Hospital",
            "Central Bank of {country}",
            "National Bank of {country}",
            "{country} Chamber of Commerce",
            "{country} Stock Exchange",
        ]
        
        for _ in range(n_samples):
            country = random.choice(self.country_names)
            template = random.choice(templates)
            negative_examples.append(template.format(country=country))
        
        # Add some completely random non-country entities
        random_entities = [
            "Microsoft", "Apple Inc", "Google LLC", "Amazon",
            "Harvard University", "MIT", "Stanford",
            "Chase Bank", "Wells Fargo", "Citibank",
            "Hilton Hotel", "Marriott", "Holiday Inn",
            "Delta Airlines", "United Airlines", "British Airways",
            "Metropolitan Museum", "Louvre Museum",
            "New York Public Library", "Boston Public Library",
            "General Hospital", "Mayo Clinic"
        ]
        
        negative_examples.extend(random_entities * (n_samples // len(random_entities)))
        
        return negative_examples[:n_samples]
    
    def create_dataset(self):
        """Create labeled dataset"""
        # Load country names
        self.load_country_aliases()
        
        # Load corporate data (real negative examples)
        self.load_corporate_data()
        
        # Generate variations
        positive_examples = self.generate_country_variations()
        
        # Generate synthetic negative examples
        synthetic_negatives = self.generate_negative_examples(len(positive_examples) // 2)
        
        # Combine with corporate data (real negative examples)
        if self.corporate_entities:
            print(f"\n✅ Adding {len(self.corporate_entities)} REAL corporate entities as negative examples")
            all_negatives = synthetic_negatives + self.corporate_entities
        else:
            print(f"\n⚠️  No corporate data - using only synthetic negative examples")
            all_negatives = synthetic_negatives
        
        # Create DataFrame
        df_positive = pd.DataFrame({
            'text': positive_examples,
            'label': 1,
            'source': 'country'
        })
        
        df_negative_synthetic = pd.DataFrame({
            'text': synthetic_negatives,
            'label': 0,
            'source': 'synthetic'
        })
        
        # Mark corporate data separately
        if self.corporate_entities:
            df_negative_corporate = pd.DataFrame({
                'text': self.corporate_entities,
                'label': 0,
                'source': 'corporate_data'
            })
            df = pd.concat([df_positive, df_negative_synthetic, df_negative_corporate], ignore_index=True)
        else:
            df = pd.concat([df_positive, df_negative_synthetic], ignore_index=True)
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\nCreated dataset with {len(df)} examples")
        print(f"  Positive (Countries): {len(df_positive)}")
        print(f"  Negative (Synthetic): {len(df_negative_synthetic)}")
        if self.corporate_entities:
            print(f"  Negative (Corporate Data): {len(self.corporate_entities)} ⭐ REAL DATA")
        
        return df
    
    def split_and_save_data(self, df):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        train_val, test = train_test_split(
            df, 
            test_size=MODEL_CONFIG['test_size'],
            random_state=MODEL_CONFIG['random_state'],
            stratify=df['label']
        )
        
        # Second split: separate validation set
        val_size_adjusted = MODEL_CONFIG['val_size'] / (1 - MODEL_CONFIG['test_size'])
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=MODEL_CONFIG['random_state'],
            stratify=train_val['label']
        )
        
        # Save datasets
        train.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
        val.to_csv(PROCESSED_DATA_DIR / 'val.csv', index=False)
        test.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
        
        print(f"\nDataset split:")
        print(f"Train: {len(train)} samples")
        print(f"Validation: {len(val)} samples")
        print(f"Test: {len(test)} samples")
        
        return train, val, test


if __name__ == "__main__":
    # Example usage
    csv_path = RAW_DATA_DIR / "country_aliases.csv"
    corporate_path = RAW_DATA_DIR.parent / "corporate" / "corporate_data.csv"  # Your data
    
    # Initialize with corporate data
    prep = CountryDataPreparation(csv_path, corporate_path)
    df = prep.create_dataset()
    train, val, test = prep.split_and_save_data(df)
    
    print("\nSample positive examples:")
    print(df[df['label'] == 1]['text'].head(10).tolist())
    
    print("\nSample negative examples (synthetic):")
    print(df[df['source'] == 'synthetic']['text'].head(5).tolist())
    
    if 'corporate_data' in df['source'].values:
        print("\nSample negative examples (YOUR corporate data):")
        print(df[df['source'] == 'corporate_data']['text'].head(10).tolist())