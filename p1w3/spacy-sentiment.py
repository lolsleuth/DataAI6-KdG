import pandas as pd
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import tqdm
import json

# Load spaCy model for aspect extraction
nlp = spacy.load("en_core_web_sm")

# Load the sentiment analysis model (BERT)
sentiment_model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

# Create the sentiment analysis pipeline
sentiment_analysis_pipeline = pipeline(
    "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=-1
)

def extract_aspects_auto(review_text):
    """
    Extract aspects (verbs) from the review text.
    """
    doc = nlp(review_text)
    aspects = [token.text for token in doc if token.pos_ == "VERB"]
    return aspects

def perform_sentiment_analysis(review_text, sentiment_pipeline):
    """
    Perform aspect-based sentiment analysis.
    """
    found_aspects = extract_aspects_auto(review_text)
    aspect_sentiments = {}

    # Perform sentiment analysis on each extracted aspect
    for aspect in found_aspects:
        sentiment = sentiment_pipeline(f"{aspect}: {review_text}")[0]
        aspect_sentiments[aspect] = {
            "label": sentiment["label"],
            "score": sentiment["score"],
        }

    return aspect_sentiments

class SentimentResult:
    def __init__(self, review_title, review_body, aspect_sentiments):
        self.review_title = review_title
        self.review_body = review_body
        self.aspect_sentiments = aspect_sentiments

    def format_for_json(self):
        """
        Format the sentiment result for JSON export.
        """
        # Check if aspect_sentiments is empty
        if not self.aspect_sentiments:
            overall_sentiment = {"label": "Neutral", "score": 0.0}  # Default if no aspects found
        else:
            overall_sentiment = max(self.aspect_sentiments.values(), key=lambda x: x["score"])

        return {
            "Review Title": self.review_title,
            "Review Body": self.review_body,
            "LLM Sentiment Analysis": f"Overall Sentiment: {overall_sentiment['label']} (Score: {overall_sentiment['score']:.2f})\nAspect Sentiments:\n" +
                "\n".join([f"  Aspect: {aspect.capitalize()} - Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})"
                           for aspect, sentiment in self.aspect_sentiments.items()])
        }

def analyze_reviews(df, sentiment_pipeline):
    """
    Analyze reviews from a DataFrame, perform aspect-based sentiment analysis on each, and return results.
    """
    sentiment_results = []

    # Iterate over the reviews in the DataFrame
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        review_title = row["reviewTitle"]
        review_body = row["reviewBody"]
        review_text = f"{review_title}. {review_body}"

        # Perform sentiment analysis
        aspect_sentiments = perform_sentiment_analysis(review_text, sentiment_pipeline)

        # Store the sentiment results
        sentiment_result = SentimentResult(review_title, review_body, aspect_sentiments)
        sentiment_results.append(sentiment_result)

    return sentiment_results

# Load reviews from CSV
df = pd.read_csv("SentimentAssignmentReviewCorpus.csv")

# Analyze the reviews
sentiment_results = analyze_reviews(df, sentiment_analysis_pipeline)

# Prepare data for JSON output
output_data = [result.format_for_json() for result in sentiment_results]

# Save results to a JSON file
with open("aspect_based_sentiment_analysis_output.json", "w") as f:
    json.dump(output_data, f, indent=4)

# Print confirmation
print("Sentiment analysis results saved to aspect_based_sentiment_analysis_output.json")
