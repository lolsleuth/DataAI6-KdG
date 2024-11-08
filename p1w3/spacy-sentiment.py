import pandas as pd
import spacy
import tqdm
import json

# Load spaCy model for aspect extraction
nlp = spacy.load("en_core_web_sm")

# Define positive and negative keywords for rule-based sentiment analysis
positive_keywords = {"good", "great", "excellent", "amazing", "positive", "love", "like", "satisfied", "recommend"}
negative_keywords = {"bad", "poor", "terrible", "awful", "negative", "hate", "dislike", "unsatisfied", "complain"}

# Custom stop list to exclude irrelevant aspects
stop_aspects = {"woman", "man", "day", "thing", "time", "person", "people", "place", "something", "everything"}


def extract_aspects_auto(review_text):
    """
    Extract relevant aspects (nouns) from the review text, filtering out irrelevant words.
    """
    doc = nlp(review_text)
    aspects = []

    for token in doc:
        # Extract nouns that are not in the stop list and are not named entities
        if token.pos_ == "NOUN" and token.text.lower() not in stop_aspects and not token.ent_type_:
            aspects.append(token.text)

    return aspects


def perform_rule_based_sentiment_analysis(review_text, aspects):
    """
    Perform rule-based sentiment analysis based on keywords for each aspect in the review text.
    """
    aspect_sentiments = {}

    # Analyze sentiment for each extracted aspect
    for aspect in aspects:
        # Check if any positive or negative keywords are associated with the aspect
        if any(word in review_text.lower() for word in positive_keywords):
            sentiment = "Positive"
            score = 1.0
        elif any(word in review_text.lower() for word in negative_keywords):
            sentiment = "Negative"
            score = -1.0
        else:
            sentiment = "Neutral"
            score = 0.0

        aspect_sentiments[aspect] = {
            "label": sentiment,
            "score": score,
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
        # Determine overall sentiment based on the scores of aspects
        if not self.aspect_sentiments:
            overall_sentiment = {"label": "Neutral", "score": 0.0}  # Default if no aspects found
        else:
            overall_sentiment = max(self.aspect_sentiments.values(), key=lambda x: x["score"])

        return {
            "Review Title": self.review_title,
            "Review Body": self.review_body,
            "Overall Sentiment": f"Label: {overall_sentiment['label']} (Score: {overall_sentiment['score']})",
            "Aspect Sentiments": [
                {
                    "Aspect": aspect,
                    "Label": sentiment['label'],
                    "Score": sentiment['score']
                }
                for aspect, sentiment in self.aspect_sentiments.items()
            ]
        }


def analyze_reviews(df):
    """
    Analyze reviews from a DataFrame, perform aspect-based sentiment analysis on each, and return results.
    """
    sentiment_results = []

    # Iterate over the reviews in the DataFrame
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        review_title = row["reviewTitle"]
        review_body = row["reviewBody"]
        review_text = f"{review_title}. {review_body}"

        # Extract aspects and perform rule-based sentiment analysis
        aspects = extract_aspects_auto(review_text)
        aspect_sentiments = perform_rule_based_sentiment_analysis(review_text, aspects)

        # Store the sentiment results
        sentiment_result = SentimentResult(review_title, review_body, aspect_sentiments)
        sentiment_results.append(sentiment_result)

    return sentiment_results


# Load reviews from CSV
df = pd.read_csv("SentimentAssignmentReviewCorpus.csv")

# Analyze the reviews
sentiment_results = analyze_reviews(df)

# Prepare data for JSON output
output_data = [result.format_for_json() for result in sentiment_results]

# Save results to a JSON file inside the 'spacy' folder
with open("spacy_sentiment_analysis.json", "w") as f:
    json.dump(output_data, f, indent=4)

# Print confirmation
print("Sentiment analysis results saved")

