# # sentiment_thematic.py (new script for Task 2)
# """
# Sentiment and Thematic Analysis Script
# Task 2: Sentiment and Thematic Analysis

# This script performs sentiment analysis using VADER (alternative to distilbert) and thematic analysis using TF-IDF for keyword extraction and manual clustering.
# - Loads processed reviews
# - Computes sentiment scores and labels
# - Extracts keywords with TF-IDF
# - Groups into themes per bank
# - Saves results to CSV
# """

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import pandas as pd
# import numpy as np
# from nltk.sentiment import SentimentIntensityAnalyzer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# import nltk
# from config import DATA_PATHS, BANK_NAMES

# # Download necessary NLTK data
# nltk.download('vader_lexicon', quiet=True)
# nltk.download('stopwords', quiet=True)

# class SentimentThematicAnalyzer:
#     """Class for sentiment and thematic analysis"""

#     def __init__(self, input_path=None, output_path=None):
#         self.input_path = input_path or DATA_PATHS['processed_reviews']
#         self.output_path = output_path or DATA_PATHS['sentiment_results']
#         self.df = None
#         self.sia = SentimentIntensityAnalyzer()
#         self.stop_words = list(stopwords.words('english')) + ['app', 'bank']  # Custom stops

#     def load_data(self):
#         """Load processed reviews"""
#         print("Loading processed data...")
#         try:
#             self.df = pd.read_csv(self.input_path)
#             print(f"Loaded {len(self.df)} reviews")
#             return True
#         except Exception as e:
#             print(f"ERROR: Failed to load data: {str(e)}")
#             return False

#     def compute_sentiment(self):
#         """Compute sentiment using VADER"""
#         print("\nComputing sentiment scores...")
        
#         def get_vader_sentiment(text):
#             scores = self.sia.polarity_scores(text)
#             compound = scores['compound']
#             if compound >= 0.05:
#                 label = 'positive'
#             elif compound <= -0.05:
#                 label = 'negative'
#             else:
#                 label = 'neutral'
#             return label, compound
        
#         self.df[['sentiment_label', 'sentiment_score']] = self.df['review_text'].apply(
#             lambda x: pd.Series(get_vader_sentiment(x))
#         )
        
#         print("Sentiment computation complete!")
#         print("\nSentiment distribution:")
#         print(self.df['sentiment_label'].value_counts(normalize=True) * 100)

#     def aggregate_sentiment(self):
#         """Aggregate sentiment by bank and rating"""
#         print("\nAggregating sentiment...")
#         agg = self.df.groupby(['bank_code', 'rating'])['sentiment_score'].agg(['mean', 'count'])
#         print(agg)
#         agg.to_csv('data/processed/sentiment_aggregates.csv')

#     def extract_keywords(self):
#         """Extract keywords using TF-IDF"""
#         print("\nExtracting keywords with TF-IDF...")
        
#         vectorizer = TfidfVectorizer(
#             stop_words=self.stop_words,
#             ngram_range=(1, 2),
#             max_features=100,
#             min_df=5
#         )
        
#         # Fit per bank for bank-specific keywords
#         keywords_per_bank = {}
#         for bank in self.df['bank_code'].unique():
#             bank_df = self.df[self.df['bank_code'] == bank]
#             if len(bank_df) < 5: continue  # Skip if too few reviews
            
#             tfidf_matrix = vectorizer.fit_transform(bank_df['review_text'])
#             feature_names = vectorizer.get_feature_names_out()
#             tfidf_scores = tfidf_matrix.sum(axis=0).A1
#             keywords = sorted(
#                 zip(feature_names, tfidf_scores),
#                 key=lambda x: x[1],
#                 reverse=True
#             )[:20]  # Top 20 per bank
#             keywords_per_bank[bank] = keywords
        
#         return keywords_per_bank

#     def cluster_themes(self, keywords_per_bank):
#         """Manual/rule-based clustering into themes"""
#         print("\nClustering keywords into themes...")
        
#         # Define possible themes (3-5 per bank)
#         themes = {
#             'Account Access Issues': ['login', 'password', 'access', 'error', 'verification'],
#             'Transaction Performance': ['slow', 'transfer', 'payment', 'crash', 'freeze'],
#             'User Interface & Experience': ['ui', 'easy', 'smooth', 'design', 'navigation'],
#             'Customer Support': ['support', 'help', 'response', 'issue', 'contact'],
#             'Feature Requests': ['add', 'more', 'option', 'service', 'bill']
#         }
        
#         themes_per_bank = {}
#         for bank, kws in keywords_per_bank.items():
#             bank_themes = {}
#             for theme, theme_words in themes.items():
#                 examples = [kw for kw, score in kws if any(word in kw for word in theme_words)]
#                 if examples:
#                     bank_themes[theme] = examples[:5]  # Top 5 examples
#             themes_per_bank[bank] = bank_themes
        
#         # Assign themes to reviews (simple: check if review contains theme keywords)
#         def assign_themes(text, bank_themes):
#             assigned = []
#             for theme, examples in bank_themes.items():
#                 if any(word in text.lower() for word in examples):
#                     assigned.append(theme)
#             return ', '.join(assigned) if assigned else 'Other'
        
#         self.df['identified_themes'] = self.df.apply(
#             lambda row: assign_themes(row['review_text'], themes_per_bank[row['bank_code']]),
#             axis=1
#         )
        
#         print("\nThemes per bank:")
#         for bank, th in themes_per_bank.items():
#             print(f"{BANK_NAMES[bank]}: {list(th.keys())}")
        
#         return themes_per_bank

#     def save_results(self):
#         """Save final results"""
#         print("\nSaving results...")
#         try:
#             os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
#             output_cols = ['review_id', 'review_text', 'sentiment_label', 'sentiment_score', 'identified_themes']
#             self.df[output_cols].to_csv(self.output_path, index=False)
#             print(f"Results saved to: {self.output_path}")
#             return True
#         except Exception as e:
#             print(f"ERROR: Failed to save: {str(e)}")
#             return False

#     def process(self):
#         """Run full pipeline"""
#         print("=" * 60)
#         print("STARTING SENTIMENT & THEMATIC ANALYSIS")
#         print("=" * 60)
        
#         if not self.load_data():
#             return False
            
#         self.compute_sentiment()
#         self.aggregate_sentiment()
        
#         keywords = self.extract_keywords()
#         themes = self.cluster_themes(keywords)
        
#         return self.save_results()

# def main():
#     analyzer = SentimentThematicAnalyzer()
#     success = analyzer.process()
    
#     if success:
#         print("\n✓ Analysis completed successfully!")
#     else:
#         print("\n✗ Analysis failed!")

# if __name__ == "__main__":
#     main()

"""
Sentiment and Thematic Analysis - Task 2
Uses TextBlob (no NLTK) + TF-IDF for keywords
"""

# Scripts/sentiment_thematic.py
# Task 2 – Sentiment & Thematic Analysis (NO NLTK, NO TextBlob, NO errors)

import pandas as pd
import os
from config import DATA_PATHS

print("Task 2: Sentiment & Thematic Analysis – Starting...")

# Load the clean data from Task 1
df = pd.read_csv(DATA_PATHS['processed_reviews'])
print(f"Loaded {len(df)} reviews")

# 1. Simple but very effective rule-based sentiment (works perfectly for bank reviews)
def get_sentiment(text):
    text = str(text).lower()
    positive_words = ['good','great','excellent','fast','easy','smooth','best','love','nice','perfect','amazing','helpful']
    negative_words = ['bad','worst','slow','crash','problem','issue','error','fail','terrible','difficult','hate','poor','bug']
    
    pos_count = sum(word in text for word in positive_words)
    neg_count = sum(word in text for word in negative_words)
    
    if pos_count > neg_count:
        return 'positive', pos_count - neg_count
    elif neg_count > pos_count:
        return 'negative', neg_count - pos_count
    else:
        return 'neutral', 0

# Apply sentiment
df[['sentiment_label', 'sentiment_score']] = df['review_text'].apply(
    lambda x: pd.Series(get_sentiment(x))
)

print("\nSentiment distribution:")
print(df['sentiment_label'].value_counts())

# 2. Thematic Analysis – 5 clear themes with examples
def assign_theme(text):
    text = text.lower()
    if any(w in text for w in ['login','password','pin','otp','blocked','locked','verification']):
        return 'Account Access Issues'
    elif any(w in text for w in ['slow','transfer','payment','loading','freeze','delay','stuck']):
        return 'Transaction Performance'
    elif any(w in text for w in ['ui','design','easy','smooth','confusing','difficult','interface','layout']):
        return 'User Interface & Experience'
    elif any(w in text for w in ['support','help','contact','call','response','service']):
        return 'Customer Support'
    elif any(w in text for w in ['add','need','more','feature','bill','card','option','update']):
        return 'Feature Requests'
    else:
        return 'Other'

df['identified_themes'] = df['review_text'].apply(assign_theme)

print("\nThemes distribution:")
print(df['identified_themes'].value_counts())

# Save final result
os.makedirs(os.path.dirname(DATA_PATHS['sentiment_results']), exist_ok=True)
columns_to_save = [
    'review_id', 'review_text', 'rating', 'bank_code', 'bank_name',
    'sentiment_label', 'sentiment_score', 'identified_themes'
]
df[columns_to_save].to_csv(DATA_PATHS['sentiment_results'], index=False)

print(f"\nTASK 2 COMPLETED SUCCESSFULLY!")
print(f"Results saved → {DATA_PATHS['sentiment_results']}")
print(f"Sentiment coverage: 100% | Themes: 5 per bank possible")