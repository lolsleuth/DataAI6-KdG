import pandas as pd
import requests
import json
import tqdm

URL = "http://localhost:11434/api/generate"  # Updated to match the working example
headers = {"Content-Type": "application/json"}  # Specify JSON headers
model_name = "llama3.1:latest"

class SentimentResult:
    def __init__(self, review_title, review_body, llm_response):
        self.review_title = review_title
        self.review_body = review_body
        self.llm_response = llm_response

    def __str__(self):
        result = f"Review Title: {self.review_title}\n"
        result += f"Review Body: {self.review_body}\n"
        result += f"LLM Sentiment Analysis: {self.llm_response}\n"
        return result

def chat_with_llm(review_title, review_body):
    """
    Send the review title and body to the local LLM for sentiment analysis using streaming.
    """
    review_text = f"""Analyze the following review and identify sentiments for each aspect mentioned. Provide the sentiment label (Positive, Negative, Neutral) and score (0.0 to 1.0) for each aspect only. 
        Don't provide the reason, note or anything else except the sentiment label and score. Don't include aspects that are not mentioned in the review.
        Title: {review_title}
        Body: {review_body}
        """
    # Note: this prompt still provides reasoning sometimes it could be not neccessary

    data = {"model": model_name, "prompt": review_text, "stream": True}

    try:
        # Make a POST request with streaming enabled
        with requests.post(f"{URL}", headers=headers, data=json.dumps(data), stream=True) as response:
            if response.status_code == 200:
                actual_response = ""
                partial_word = ""  # Accumulate incomplete words
                print("LLM Response: ", end="", flush=True)

                # Process the response in chunks (streaming)
                for chunk in response.iter_lines():
                    if chunk:
                        data = json.loads(chunk.decode("utf-8"))
                        response_text = data.get("response", "")

                        # Add the current chunk to the partial word
                        response_text = partial_word + response_text
                        words = response_text.split(" ")

                        # Print all words except the last, which may be incomplete
                        for word in words[:-1]:
                            print(word, end=" ", flush=True)
                            actual_response += word + " "

                        # Keep the last word as it might be incomplete
                        partial_word = words[-1]

                # Print any remaining partial word
                if partial_word:
                    print(partial_word, end=" ", flush=True)
                    actual_response += partial_word

                return actual_response.strip()  # Return the final response
            else:
                return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error: Unable to get response from the API. Details: {str(e)}"

def analyze_reviews(df):
    """
    Analyze the reviews using the local LLM API and return the sentiment results.
    """
    sentiment_results = []

    # Iterate over the reviews in the DataFrame
    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        review_title = row["reviewTitle"]
        review_body = row["reviewBody"]

        # Send the review text to the LLM for sentiment analysis
        llm_response = chat_with_llm(review_title, review_body)

        # Store the LLM sentiment result
        sentiment_result = SentimentResult(review_title, review_body, llm_response)
        sentiment_results.append(sentiment_result)

    return sentiment_results

# Load reviews from the CSV file
df = pd.read_csv("p1w3/SentimentAssignmentReviewCorpus.csv")

# Analyze the reviews using the LLM API
sentiment_results = analyze_reviews(df)

# Print the sentiment results for the top rows
for sentiment_result in sentiment_results[:3]:  
    print(sentiment_result)
