import pandas as pd
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import requests
import tqdm

# Load spaCy model for aspect extraction
nlp = spacy.load("en_core_web_sm")

# Load the sentiment analysis model (BERT)
sentiment_model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

# Create the sentiment analysis pipeline
sentiment_analysis_pipeline = pipeline(
    "sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=0
)

URL = "http://localhost:11434/api"  # The updated API URL

def extract_aspects_auto(review_text):
    """
    Extract aspects (verbs) from the review text.
    """
    doc = nlp(review_text)
    aspects = [token.text for token in doc if token.pos_ == "VERB"]
    return aspects

def perform_sentiment_analysis(review_text, sentiment_pipeline):
    """
    Perform overall and aspect-based sentiment analysis.
    """
    # Extract aspects (verbs) using spaCy
    found_aspects = extract_aspects_auto(review_text)
    aspect_sentiments = {}

    # Get the overall sentiment of the review
    overall_sentiment = sentiment_pipeline(review_text)[0]

    # Perform sentiment analysis on each extracted aspect
    for aspect in found_aspects:
        sentiment = sentiment_pipeline(f"{aspect}: {review_text}")[0]
        aspect_sentiments[aspect] = {
            "label": sentiment["label"],
            "score": sentiment["score"],
        }

    return overall_sentiment, aspect_sentiments

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

def analyze_reviews(df, sentiment_pipeline):
    """
    Analyze reviews from a DataFrame, perform sentiment analysis on each, and return results.
    """
    sentiment_results = []

    # Iterate over the reviews in the DataFrame
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        review_title = row["reviewTitle"]
        review_body = row["reviewBody"]
        review_text = f"{review_title}. {review_body}"

        # Perform sentiment analysis
        overall_sentiment, aspect_sentiments = perform_sentiment_analysis(review_text, sentiment_pipeline)

        # Store the sentiment results
        sentiment_result = SentimentResult(review_title, review_body, aspect_sentiments, overall_sentiment)
        sentiment_results.append(sentiment_result)

    return sentiment_results

def chat_with_api(review_text):
    """
    Chat with the local LLM via the provided API and get a response.
    """
    data = {"prompt": review_text}

    # Make request to local LLM API (Ollama) to analyze the review
    response = requests.post(URL, json=data)
    if response.status_code == 200:
        return response.json().get("reply", "")
    else:
        return "Error: Unable to get response from the API."

# Load reviews from CSV
df = pd.read_csv("SentimentAssignmentReviewCorpus.csv")

# Analyze the reviews
sentiment_results = analyze_reviews(df, sentiment_analysis_pipeline)

# Print the sentiment results and simulate chatbot interaction
for sentiment_result in sentiment_results[:9]:  # Only process the top 9 rows
    print(sentiment_result)

    # Simulate sending the review to the local LLM for analysis
    llm_response = chat_with_api(f"{sentiment_result.review_title}. {sentiment_result.review_body}")
    print(f"LLM Response: {llm_response}\n")
