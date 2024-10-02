"""
This script is for Exercise 4 of week 2 of Data and AI 6 of the course.

It allows you to choose from all the local models in Ollama and then engages in a conversation, streaming the response word by word.
"""

import json

import requests

# URL of the API
URL = "http://localhost:11434/api"

# Headers for the request
headers = {
    "Content-Type": "application/json",
}


# Fetch available models
def get_available_models():
    response = requests.get(
        f"{URL}/tags", headers=headers
    )  # Corrected endpoint to /tags
    if response.status_code == 200:
        return response.json().get("models", [])
    else:
        print("Error fetching models:", response.status_code, response.text)
        return []


# Let user choose a model
def choose_model(models):
    print("Available Models:")
    for idx, model in enumerate(models):
        print(f"{idx + 1}. {model['name']}")  # Display only the 'name' field

    while True:
        try:
            choice = int(input("Choose a model by number: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]["name"]  # Return only the model name
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# Main conversation loop
def start_conversation(model_name):
    conversation_history = ""

    while True:
        # Take user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Append the user input to the conversation history
        conversation_history += f"\nYou: {user_input}"

        # Prepare the request data
        data = {
            "model": model_name,
            "prompt": conversation_history,
            "stream": True,  # Enable streaming
        }

        # Make the POST request and stream the response
        with requests.post(
            f"{URL}/generate", headers=headers, data=json.dumps(data), stream=True
        ) as response:
            if response.status_code == 200:
                actual_response = ""
                partial_word = ""  # To accumulate incomplete words
                print(
                    "API: ", end="", flush=True
                )  # Print without newline and flush immediately

                # Process the response chunk by chunk
                for chunk in response.iter_lines():
                    if chunk:
                        # Decode the JSON line and extract the text response
                        data = json.loads(chunk.decode("utf-8"))
                        response_text = data.get("response", "")

                        # Add the current chunk to the partial word
                        response_text = partial_word + response_text
                        words = response_text.split(" ")  # Split on spaces

                        # Print all words except the last, which may be incomplete
                        for word in words[:-1]:
                            print(word, end=" ", flush=True)
                            actual_response += word + " "

                        # Keep the last word as it might be incomplete
                        partial_word = words[-1]

                # Print any remaining partial word if it's valid
                if partial_word:
                    print(partial_word, end=" ", flush=True)
                    actual_response += partial_word

                conversation_history += f"\nAPI: {actual_response.strip()}"
                print()  # Add a newline after the streaming finishes
            else:
                print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    # Fetch and choose model
    models = get_available_models()
    if models:
        selected_model = choose_model(models)
        print(f"You selected: {selected_model}\n")
        start_conversation(selected_model)
    else:
        print("No models available locally.")
