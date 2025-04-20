from openai import OpenAI

client = OpenAI(api_key="sk-AwXaGaXCcQ8znID0cyhEvxaf9ObgWpwAxyUHB9DBNavMwVaj", base_url="https://api.ninepath.top/v1")

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)