
import openai

openai.api_key = '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'

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt="Hello, how are you?",
  temperature=0.5,
  max_tokens=100
)

print(response.choices[0].text.strip())