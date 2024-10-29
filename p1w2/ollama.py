"""
This script is for Exercise 4 of week 2 of Data and AI 6 of the course.

It allows you to choose from all the local models in Ollama and then engages in a conversation,
streaming the response word by word. You can select between the normal conversation mode
and the RAG  mode, which pulls relevant context from a manual that I created.
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
    response = requests.get(f"{URL}/tags", headers=headers)
    if response.status_code == 200:
        return response.json().get("models", [])
    else:
        print("Error fetching models:", response.status_code, response.text)
        return []


# Let user choose a model
def choose_model(models):
    print("Available Models:")
    for idx, model in enumerate(models):
        print(f"{idx + 1}. {model['name']}")

    while True:
        try:
            choice = int(input("Choose a model by number: ")) - 1
            if 0 <= choice < len(models):
                return models[choice]["name"]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# Load documents for RAG
def load_documents():
    try:
        with open("manual.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Document file not found. RAG mode won't work without it.")
        return []


# Retrieve relevant context from documents
def retrieve_context(user_input, documents):
    for doc in documents:
        if any(keyword in user_input.lower() for keyword in doc["keywords"]):
            return doc["content"]
    return ""


# Main conversation loop with RAG option
def start_conversation(model_name, use_rag=False, documents=None):
    conversation_history = ""

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        conversation_history += f"\nYou: {user_input}"

        # Retrieve context if RAG mode is on
        if use_rag and documents:
            retrieved_text = retrieve_context(user_input, documents)
            prompt = f"Context: {retrieved_text}\n\n{conversation_history}" if retrieved_text else conversation_history
        else:
            prompt = conversation_history

        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
        }

        with requests.post(f"{URL}/generate", headers=headers, data=json.dumps(data), stream=True) as response:
            if response.status_code == 200:
                actual_response = ""
                partial_word = ""
                print("API: ", end="", flush=True)

                for chunk in response.iter_lines():
                    if chunk:
                        data = json.loads(chunk.decode("utf-8"))
                        response_text = data.get("response", "")

                        response_text = partial_word + response_text
                        words = response_text.split(" ")

                        for word in words[:-1]:
                            print(word, end=" ", flush=True)
                            actual_response += word + " "

                        partial_word = words[-1]

                if partial_word:
                    print(partial_word, end=" ", flush=True)
                    actual_response += partial_word

                conversation_history += f"\nAPI: {actual_response.strip()}"
                print()
            else:
                print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    # Choose normal or RAG mode
    mode = input("Choose mode (1: Normal, 2: RAG): ").strip()
    use_rag = mode == "2"

    # Load documents if RAG mode is chosen
    documents = load_documents() if use_rag else []

    # Fetch available models
    models = get_available_models()
    if models:
        selected_model = choose_model(models)
        print(f"You selected: {selected_model}\n")
        start_conversation(selected_model, use_rag=use_rag, documents=documents)
    else:
        print("No models available locally.")


# 5. Run 5 of your prompts from last week through your Ollama chatbot.
# See any differences? Regression? Anything you could fix through prompt engineering?

# I ran the following prompts through the Ollama chatbot:
# 1. "Write a prequel about Star Wars"
#     - The response was a bit different, but besides to Mistrals response it was very similar to the other two.
#     - The story that it gives could be definitely fixed through prompt engineering. As currently it's a
#         very vague prompt.

# 2. "Create a new playlist of new song names from 'Metallica'"
#     - The response was very similar to the other models besides Gemini. It gave the song names for the playlist but also
#         explained how it would sound like.
#     - There's nothing to fix here, the prompt is very clear and the response is what I would expect.

# 3. "Write lyrics for the song ‘Emptiness Machine’"
#     - The response was similar to all the other llms, and instead of giving the actual lyrics to the song
#         it came up with new lyrics.
#     - The prompt could be fixed by specifying that this song actually exists by the band Linkin Park.

# 4. "Explain how LLMs work in League of Legends terms"
#     - The response was very similar to the other models here as well.
#     - It could be definitely improved by adding for example "act like a league of legends player" to the prompt.

# 5. "Write about why Arch Linux is way better than all other operating systems, act like a hard
# core arch user""
#     - Again, the respone was very similar to the other models.
#     - The prompt can be improved by specifying that the response should be in the form of a rant for example.
