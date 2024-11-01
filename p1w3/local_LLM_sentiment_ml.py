import pandas as pd
import requests
import json
import tqdm

URL = "http://localhost:11434/api/generate"  
headers = {"Content-Type": "application/json"} 
model_name = "llama3.2:latest"

class SentimentResult:
    def __init__(self, review_title, review_body, llm_response):
        self.review_title = review_title
        self.review_body = review_body
        self.llm_response = llm_response

    def to_dict(self):
        return {
            "Review Title": self.review_title,
            "Review Body": self.review_body,
            "LLM Sentiment Analysis": self.llm_response
        }

def chat_with_llm(session, review_title, review_body):
    """
    Send the review title and body to the local LLM for sentiment analysis.
    """
    review_text = f"""Analyze the following review for sentiment. Identify each mentioned aspect and provide:
    - An overall sentiment label and score (0.0 to 1.0).
    - For each aspect, specify the sentiment label (5 stars = Positive, 3 stars = Neutral, 1 star = Negative) and a score.
    
    Output only in this structured format:
    Review Title: {review_title}
    Review Body: {review_body}
    Overall Sentiment: [Sentiment label] (Score: [Score])
    Aspect Sentiments:
      Aspect: [Aspect 1] - Sentiment: [Sentiment label] (Score: [Score])
      Aspect: [Aspect 2] - Sentiment: [Sentiment label] (Score: [Score])
      ...
      
    Only include aspects explicitly mentioned in the review. Do not provide any explanations or extraneous text.
    Title: {review_title}
    Body: {review_body}
    """

    data = {"model": model_name, "prompt": review_text, "stream": False}  # Set stream to False for simplicity

    try:
        response = session.post(URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to get response from the API. Details: {str(e)}"

def analyze_reviews(df, output_file="sentiment_results.json"):
    """
    Analyze the reviews using the local LLM API and save the sentiment results to a JSON file.
    """
    sentiment_results = []

    # Reuse a single session for all requests
    with requests.Session() as session:
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
            review_title = row["reviewTitle"]
            review_body = row["reviewBody"]

            # Send the review text to the LLM for sentiment analysis
            llm_response = chat_with_llm(session, review_title, review_body)

            # Create a result object and add it to the results list
            sentiment_result = SentimentResult(review_title, review_body, llm_response)
            sentiment_results.append(sentiment_result.to_dict())

    # Save all results to a JSON file
    with open(output_file, "w") as f:
        json.dump(sentiment_results, f, indent=4)
    print(f"Sentiment analysis results saved to {output_file}")

# Load reviews from the CSV file
df = pd.read_csv("p1w3/SentimentAssignmentReviewCorpus.csv")

# Analyze the reviews and save the results to JSON
analyze_reviews(df)
