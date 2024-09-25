import requests
import os
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient

def load_api_key():
    # Load the API key from environment
    load_dotenv()
    hugging_face_token = os.getenv("HUGGING_FACE_API_KEY")
    return hugging_face_token

def get_models():
    # Return the models as lists
    models = [
        [
            "llama3-8b-8192",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "mistralai/Mistral-7B-Instruct-v0.1",
        ],
        [
            "llama3-8b",
            "gemma2-9b",
            "mixtral-8",
            "Meta-Llama-3-8B",
            "Phi-3-mini-4k",
            "Mistral-7B-Instruct-v0.1",
        ],
    ]
    return models

def select_model():
    valid_choices = [1, 2, 3, 4, 5, 6]
    models = get_models()

    while True:
        try:
            print("Please select a number between 1 and 6:")
            for i, model in enumerate(models[1], start=1):
                print(f"{i} | {model}")

            user_choice = int(input("Enter your choice:"))
            print("\n")

            if user_choice in valid_choices:
                print(f"You selected {models[1][user_choice-1]}.")
                return user_choice
            else:
                print("Invalid choice. Please select a number between 1 and 6.")

        except ValueError:
            print("Invalid input. Please enter a number.")

def enter_prompt(user_choice, hugging_face_token):
    models = get_models()

    while True:
        print(f"You are talking with {models[1][user_choice - 1]}")
        print("If you want to select another model, enter 'RETURN' in the place of the prompt.")

        user_prompt = input("Please enter a prompt:")
        print("\n")

        if user_prompt.upper() == "RETURN":
            return  # Exit this function, allow for selecting a new model

        if user_choice < 3:
            # Use Groq API
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": user_prompt}],
                model=models[0][user_choice - 1],
            )
            print(chat_completion.choices[0].message.content)
            print("\n")

        else:
            # Use Hugging Face API
            client = InferenceClient(token=hugging_face_token)
            for message in client.chat_completion(
                model=models[0][user_choice - 1],
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=500,
                stream=True,
            ):
                print(message.choices[0].delta.content, end="")
            print("\n")

def main():
    # Load API keys
    hugging_face_token = load_api_key()

    print("\n****Welcome to Group 4 AI chatbot****\n")
    
    while True:
        # Select a model
        user_choice = select_model()
        
        # Enter prompt and interact with the model
        enter_prompt(user_choice, hugging_face_token)

# Run the program
if __name__ == "__main__":
    main()
