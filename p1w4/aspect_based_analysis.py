import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import time
import tqdm
import json

# Load sentiment analysis model (ABSA-capable BERT)
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_aspects(review_text, sentiment_pipeline):
    """
    Perform aspect-based sentiment analysis for predefined aspects in review text.
    """
    aspects = ["quality", "price", "design", "durability"]  # Hypothetical aspects
    aspect_sentiments = {}

    # Perform sentiment analysis on each predefined aspect
    for aspect in aspects:
        sentiment = sentiment_pipeline(f"{aspect}: {review_text[:512]}")[0]
        aspect_sentiments[aspect] = {
            "label": sentiment["label"],
            "score": sentiment["score"]
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
        overall_sentiment = max(self.aspect_sentiments.values(), key=lambda x: x["score"])
        result = {
            "Review Title": self.review_title,
            "Review Body": self.review_body,
            "LLM Sentiment Analysis": f"Overall Sentiment: {overall_sentiment['label']} (Score: {overall_sentiment['score']:.2f})\nAspect Sentiments:\n" +
                "\n".join([f"  Aspect: {aspect.capitalize()} - Sentiment: {sentiment['label']} (Score: {sentiment['score']:.2f})"
                           for aspect, sentiment in self.aspect_sentiments.items()])
        }
        return result

def analyze_reviews(df, sentiment_pipeline):
    """
    Analyze reviews from a DataFrame, perform aspect-based sentiment analysis on each, and return results.
    """
    sentiment_results = []
    start_time = time.time()  # Start timing

    # Iterate over the reviews in the DataFrame
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        review_title = row["reviewTitle"]
        review_body = row["reviewBody"]
        review_text = f"{review_title}. {review_body}"

        # Perform sentiment analysis on predefined aspects
        aspect_sentiments = analyze_aspects(review_text, sentiment_pipeline)

        # Store the sentiment results
        sentiment_result = SentimentResult(review_title, review_body, aspect_sentiments)
        sentiment_results.append(sentiment_result)

    end_time = time.time()  # End timing
    print(f"\nTotal Processing Time: {end_time - start_time:.2f} seconds")  # Display time

    return sentiment_results

# Load reviews from CSV
df = pd.read_csv("SentimentAssignmentReviewCorpus.csv")

# Analyze the reviews
sentiment_results = analyze_reviews(df, sentiment_pipeline)

# Prepare data for JSON output
output_data = [result.format_for_json() for result in sentiment_results]

# Save results to a JSON file
with open("aspect_based_sentiment_analysis_output.json", "w") as f:
    json.dump(output_data, f, indent=4)

# Print confirmation
print("Sentiment analysis results saved to aspect_based_sentiment_analysis_output.json")
