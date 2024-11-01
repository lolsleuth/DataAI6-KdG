"""
This script is for sentiment analysis using machine learning.
"""

import pandas as pd
import spacy
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

df = pd.read_csv("p1w3/SentimentAssignmentReviewCorpus.csv")

nlp = spacy.load("en_core_web_sm")

# Load the BERT model and tokenizer for sentiment analysis
sentiment_model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    sentiment_model_path
)

# Sentiment analysis pipeline using the BERT model
sentiment_analysis_pipeline = pipeline(
    "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=0
)


def extract_aspects_auto(review_text):
    """
    Params: review_text (str) - the text of the review
    Returns: aspects (list) - a list of aspects found in

    This function uses the spaCy library to extract aspects (verbs) from the review text.
    """
    doc = nlp(review_text)
    aspects = [token.text for token in doc if token.pos_ == "VERB"]
    return aspects


# Class to store and print sentiment analysis results
class SentimentResult:
    def __init__(self, review_title, review_body, aspect_sentiments, overall_sentiment):
        self.review_title = review_title
        self.review_body = review_body
        self.aspect_sentiments = aspect_sentiments
        self.overall_sentiment = overall_sentiment

    def __str__(self):
        result = f"Review Title: {self.review_title}\n"
        result += f"Review Body: {self.review_body}\n"
        result += f"Overall Sentiment: {self.overall_sentiment['label']} (Score: {self.overall_sentiment['score']:.2f})\n"
        result += "Aspect Sentiments:\n"
        for aspect, sentiment in self.aspect_sentiments.items():
            result += f"  Aspect: {aspect} - Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})\n"
        return result


def analyze_review_aspects_auto(review_title, review_body, sentiment_pipeline):
    """
    Params: review_title (str) - the title of the review text
            review_body (str) - the body of the review text
            sentiment_pipeline (pipeline) - the sentiment analysis pipeline
    Returns: SentimentResult - a class instance containing the sentiment analysis results

    This function analyzes the sentiment of the review text, extracting aspects automatically using spaCy.
    """
    review_text = f"{review_title}. {review_body}"
    found_aspects = extract_aspects_auto(review_text)
    aspect_sentiments = {}

    overall_sentiment = sentiment_pipeline(review_text)[0]

    for aspect in found_aspects:
        sentiment = sentiment_pipeline(f"{aspect}: {review_text}")[0]
        aspect_sentiments[aspect] = {
            "label": sentiment["label"],
            "score": sentiment["score"],
        }

    return SentimentResult(
        review_title, review_body, aspect_sentiments, overall_sentiment
    )


# Create a list to store sentiment results
sentiment_results = []

# Use tqdm to track progress across the entire dataset
for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
    sentiment_result = analyze_review_aspects_auto(
        row["reviewTitle"], row["reviewBody"], sentiment_analysis_pipeline
    )
    sentiment_results.append(sentiment_result)

# Store the results back into the DataFrame
df["sentiment_results"] = sentiment_results

# Print the sentiment results
for sentiment_result in sentiment_results:
    print(sentiment_result)
