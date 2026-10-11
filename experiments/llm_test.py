from ollama import chat

response = chat(
    model="qwen3:4b-instruct",
    messages=[
        {
            "role": "user",
            "content": (
                "Explain in one sentence why a battery may "
                "discharge during an electricity price spike."
            )
        }
    ]
)

print(response.message.content)