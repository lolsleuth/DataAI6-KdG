"""
This script is for Exercise 4 of week 2 of Data and AI 6 of the course.

It is a simple script that takes user input and sends it to a local instance of Ollama.
"""

import json
import requests

URL = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json",
}

user_input = input("Enter your input: ")

data = {
    "model": "llama3.1",
    "prompt": user_input,
    "stream": False
}

response = requests.post(URL, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    response_text = response.text
    data = json.loads(response_text)
    acutal_response = data["response"]
    print(acutal_response)
else:
    print("Error:", response.status_code, response.text)
