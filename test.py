from openai import OpenAI
client = OpenAI(api_key="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")

stream = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' ten times fast.",
        },
    ],
    stream=True,
)

for event in stream:
    print(event)