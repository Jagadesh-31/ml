import requests



def chat_with_tinyllama(user_input):
    url = "http://localhost:11434/api/generate"

    prompt = user_input

    payload = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        bot_reply = response.json()['response'].strip()

        return bot_reply
    else:
        return f"[Error {response.status_code}] {response.text}"


print("TinyLlama Chat (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    reply = chat_with_tinyllama(user_input)
    print("TinyLlama:", reply)
