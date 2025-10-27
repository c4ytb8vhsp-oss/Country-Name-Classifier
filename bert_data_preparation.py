import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from pathlib import Path
from config_bert import *
import logging
import pycountry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedDataPreparation:
    """Enhanced data preparation with financial instrument awareness"""
    
    def __init__(self, country_csv_path, corporate_data_path=None):
        """
        Initialize with country aliases and corporate data
        
        Args:
            country_csv_path: Path to country aliases CSV
            corporate_data_path: Path to your internal corporate_data CSV/Excel
        """
        self.country_csv_path = country_csv_path
        self.corporate_data_path = corporate_data_path
        self.country_names = []
        self.corporate_entities = []
        
    def load_country_aliases(self):
        """Load country names from multiple sources"""
        df = pd.read_csv(self.country_csv_path)
    
        # Extract all country variations
        country_names = []
        
        # Adjust based on actual CSV structure
        # Filter rows where 'AliasDescription' contains 'English' (case-insensitive)
        filtered_df = df[df['AliasDescription'].str.contains('English', case=False, na=False)]

        # Add values from the 'Alias' column only
        country_names.extend(filtered_df['Alias'].dropna().unique().tolist())
        
        # Add pycountry data
        try:
            for country in pycountry.countries:
                country_names.append(country.name)
                if hasattr(country, 'official_name'):
                    country_names.append(country.official_name)
        except ImportError:
            logger.warning("pycountry not installed, skipping official names")
        
        self.country_names = list(set([name.strip() for name in country_names if name and len(name) > 2]))
        logger.info(f"Loaded {len(self.country_names)} unique country names")
        
        return self.country_names
    
    def load_corporate_data(self):
        """
        Load your internal corporate_data
        
        Expected format: CSV/Excel with columns like:
        - entity_name: Name of the entity
        - entity_type: Type (optional: bond, deposit, equity, etc.)
        - is_country: 0 or 1 (if available)
        """
        if self.corporate_data_path is None:
            logger.warning("No corporate data path provided")
            return []
        
        path = Path(self.corporate_data_path)
        
        if not path.exists():
            logger.warning(f"Corporate data not found at {path}")
            return []
        
        # Load based on file extension
        if path.suffix == '.csv':
            df = pd.read_csv(path)
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            logger.error(f"Unsupported file format: {path.suffix}")
            return []
        
        logger.info(f"Loaded {len(df)} records from corporate data")
        
        # Extract entity names
        if 'entity_name' in df.columns:
            self.corporate_entities = df['entity_name'].dropna().tolist()
        elif 'name' in df.columns:
            self.corporate_entities = df['name'].dropna().tolist()
        else:
            # Use first column
            self.corporate_entities = df.iloc[:, 0].dropna().tolist()
        
        logger.info(f"Extracted {len(self.corporate_entities)} corporate entities")
        
        return df
    
    def generate_country_variations(self):
        """Generate country name variations"""
        variations = []
        
        prefixes = [
            "", "State of", "Republic of", "Kingdom of", 
            "Democratic Republic of", "People's Republic of", 
            "Federation of", "Commonwealth of", "Sultanate of", 
            "Emirate of", "Principality of", "Union of"
        ]
        
        for country in self.country_names:
            for prefix in prefixes:
                if prefix:
                    variations.append(f"{prefix} {country}")
                else:
                    variations.append(country)
            
            # Add "the" variations
            if not country.lower().startswith("the "):
                variations.append(f"the {country}")
                variations.append(f"The {country}")
        
        return list(set(variations))
    
    def generate_financial_instrument_negatives(self, n_samples=None):
        """
        Generate financial instrument names containing country names
        These should be classified as NOT countries
        """
        if n_samples is None:
            n_samples = len(self.country_names) * 2
        
        negative_examples = []
        
        # Use patterns from config
        for _ in range(n_samples):
            country = random.choice(self.country_names)
            pattern = random.choice(COUNTRY_FINANCIAL_PATTERNS)
            negative_examples.append(pattern.format(country=country))
        
        # Add specific financial instruments
        specific_instruments = [
            "US Treasury Bond",
            "US Treasury 10Y",
            "German Bund",
            "UK Gilt",
            "Japan Government Bond",
            "JGB 10Y",
            "Korean Won Deposit",
            "Brazil Sovereign Bond",
            "South Africa Government Bond",
            "Mexico Treasury",
            "India Government Securities",
            "China Government Bond",
            "France OAT",
            "Italy BTP",
            "Spain Bonos",
            "Canadian Government Bond",
            "Australian Government Bond",
            "Swiss Confederation Bond",
            "Swedish Government Bond",
            "Norway Government Bond",
            # Currency instruments
            "USD/EUR Forex",
            "GBP/JPY Exchange Rate",
            "Korean Won Currency",
            "Brazilian Real FX",
            # Deposits
            "US Dollar Deposit",
            "Euro Deposit",
            "Sterling Deposit",
            "Yen Deposit",
            # Corporate bonds with country names
            "Samsung Korea Bond",
            "Toyota Japan Bond",
            "Deutsche Bank Germany",
            "HSBC UK Bond",
        ]
        
        negative_examples.extend(specific_instruments)
        
        return negative_examples[:n_samples]
    
    def generate_organization_negatives(self, n_samples=None):
        """Generate organization names with country references"""
        if n_samples is None:
            n_samples = len(self.country_names)
        
        negative_examples = []
        
        org_templates = [
            "Bank of {country}",
            "{country} Bank",
            "Central Bank of {country}",
            "National Bank of {country}",
            "University of {country}",
            "{country} University",
            "Embassy of {country}",
            "{country} Embassy",
            "{country} Airlines",
            "{country} Airport",
            "Ministry of {country}",
            "{country} Ministry",
            "{country} Department of Finance",
            "{country} Stock Exchange",
            "{country} Chamber of Commerce",
            "Government of {country}",
            "{country} Customs",
            "{country} Immigration",
        ]
        
        for _ in range(n_samples):
            country = random.choice(self.country_names)
            template = random.choice(org_templates)
            negative_examples.append(template.format(country=country))
        
        return negative_examples
    
    def create_enhanced_dataset(self):
        """Create comprehensive labeled dataset"""
        logger.info("Creating enhanced dataset...")
        
        # Load data sources
        self.load_country_aliases()
        corporate_df = self.load_corporate_data()
        
        # Generate positive examples (countries)
        positive_examples = self.generate_country_variations()
        
        # Generate negative examples
        financial_negatives = self.generate_financial_instrument_negatives(
            n_samples=len(positive_examples) // 2
        )
        org_negatives = self.generate_organization_negatives(
            n_samples=len(positive_examples) // 3
        )
        
        # Add corporate data if available
        if self.corporate_entities:
            # Assume corporate entities are NOT countries
            corporate_negatives = self.corporate_entities
        else:
            corporate_negatives = []
        
        # Combine all negatives
        all_negatives = financial_negatives + org_negatives + corporate_negatives
        
        # Create DataFrames
        df_positive = pd.DataFrame({
            'text': positive_examples,
            'label': 1,
            'source': 'country_aliases'
        })
        
        df_negative = pd.DataFrame({
            'text': all_negatives,
            'label': 0,
            'source': 'generated'
        })
        
        # Mark financial instruments separately for tracking
        df_negative.loc[
            df_negative['text'].isin(financial_negatives), 
            'source'
        ] = 'financial_instrument'
        
        df_negative.loc[
            df_negative['text'].isin(corporate_negatives),
            'source'
        ] = 'corporate_data'
        
        # Combine and shuffle
        df = pd.concat([df_positive, df_negative], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created dataset with {len(df)} examples")
        logger.info(f"  - Countries: {len(df_positive)}")
        logger.info(f"  - Financial instruments: {len(financial_negatives)}")
        logger.info(f"  - Organizations: {len(org_negatives)}")
        logger.info(f"  - Corporate data: {len(corporate_negatives)}")
        
        return df
    
    def create_financial_test_set(self):
        """
        Create a dedicated test set for financial instruments
        This ensures we properly test false positive cases
        """
        logger.info("Creating financial instrument test set...")
        
        financial_test_cases = []
        
        # Critical test cases - these MUST be classified as NOT country
        critical_cases = [
            # Government bonds
            ("US Treasury Bond", 0),
            ("US Treasury 10 Year", 0),
            ("United States Government Bond", 0),
            ("Korea Government Bond", 0),
            ("Korean Treasury", 0),
            ("South Korea Sovereign Bond", 0),
            ("Japan Government Bond", 0),
            ("Japanese Government Securities", 0),
            ("German Bund", 0),
            ("Germany Government Bond", 0),
            ("UK Gilt", 0),
            ("United Kingdom Government Bond", 0),
            
            # Deposits
            ("Korea Deposit", 0),
            ("Korean Won Deposit", 0),
            ("US Dollar Deposit", 0),
            ("Euro Deposit", 0),
            ("Japan Yen Deposit", 0),
            
            # Currency/FX
            ("Korean Won", 0),
            ("US Dollar", 0),
            ("Euro Currency", 0),
            ("Japan Yen", 0),
            
            # Sovereign debt
            ("Brazil Sovereign Debt", 0),
            ("Mexico Sovereign Bond", 0),
            ("India Sovereign Securities", 0),
            
            # vs. actual countries (TRUE POSITIVES)
            ("United States", 1),
            ("United States of America", 1),
            ("Korea", 1),
            ("South Korea", 1),
            ("Republic of Korea", 1),
            ("Japan", 1),
            ("Germany", 1),
            ("United Kingdom", 1),
            ("Brazil", 1),
            ("Mexico", 1),
            ("India", 1),
        ]
        
        financial_test_cases.extend(critical_cases)
        
        # Generate more systematic cases
        test_countries = ["US", "UK", "Germany", "France", "Japan", "China", 
                         "Korea", "Brazil", "India", "Canada"]
        
        financial_terms = ["Bond", "Treasury", "Deposit", "Note", "Bill", 
                          "Sovereign Bond", "Government Bond"]
        
        for country in test_countries:
            # Add pure country (should be 1)
            financial_test_cases.append((country, 1))
            
            # Add financial instruments (should be 0)
            for term in financial_terms:
                financial_test_cases.append((f"{country} {term}", 0))
        
        df_financial_test = pd.DataFrame(
            financial_test_cases,
            columns=['text', 'label']
        )
        
        df_financial_test['test_type'] = 'financial_critical'
        
        logger.info(f"Created {len(df_financial_test)} financial test cases")
        
        return df_financial_test
    
    def split_and_save_data(self, df, include_financial_test=True):
        """Split data with special attention to financial instruments"""
        
        # Stratified split
        train_val, test = train_test_split(
            df,
            test_size=ML_CONFIG['test_size'],
            random_state=ML_CONFIG['random_state'],
            stratify=df['label']
        )
        
        val_size_adjusted = ML_CONFIG['val_size'] / (1 - ML_CONFIG['test_size'])
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=ML_CONFIG['random_state'],
            stratify=train_val['label']
        )
        
        # Save standard splits
        train.to_csv(PROCESSED_DATA_DIR / 'train.csv', index=False)
        val.to_csv(PROCESSED_DATA_DIR / 'val.csv', index=False)
        test.to_csv(PROCESSED_DATA_DIR / 'test.csv', index=False)
        
        logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        # Create and save financial test set
        if include_financial_test:
            df_financial_test = self.create_financial_test_set()
            df_financial_test.to_csv(
                PROCESSED_DATA_DIR / 'test_financial.csv', 
                index=False
            )
            logger.info(f"Financial test set: {len(df_financial_test)}")
        
        return train, val, test


if __name__ == "__main__":
    # Example usage
    country_csv = RAW_DATA_DIR / "country_aliases.csv"
    corporate_csv = CORPORATE_DATA_DIR / "corporate_data.csv"  # Your internal data
    
    prep = EnhancedDataPreparation(country_csv, corporate_csv)
    df = prep.create_enhanced_dataset()
    train, val, test = prep.split_and_save_data(df)
    
    print("\nSample negative examples (should NOT be countries):")
    print(df[df['label'] == 0].sample(10)['text'].tolist())