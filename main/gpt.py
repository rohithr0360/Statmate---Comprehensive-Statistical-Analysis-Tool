import openai
# cit - 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
openai.api_key = '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'


prompt_text = "Translate the following English text to French: 'Hello, how are you?'"

response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt_text,
    max_tokens=150
)

generated_text = response.choices[0].text.strip()
print(generated_text)
