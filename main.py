import anthropic
import openai
anthropic_client = anthropic.Anthropic()
openai_client = openai.OpenAI()

def call_anthropic(prompt, model="claude-3-5-sonnet-20240620"):
  message = anthropic_client.messages.create(
      model=model,
      max_tokens=1024,
      messages=[
          {"role": "user", "content": prompt},
      ],
  )
  return message.content[0].text

def call_openai(prompt, model="gpt-3.5-turbo"):
  chat_completion = openai_client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model=model,
  )
  return chat_completion.choices[0].message.content

question = "how many As are there in the word Blueberry?"

a1, a2 = call_anthropic(question), call_openai(question)
print("Claude Sonnet 3.5:\n", a1)
print("GPT 3.5:\n", a2)
