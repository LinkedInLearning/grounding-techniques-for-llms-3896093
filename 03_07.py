import openai
import os
import requests
from uuid import uuid4
openai_client = openai.OpenAI()
llm_model = "gpt-4o-mini"

def call_openai(prompt, model="gpt-4o-mini"):
  chat_completion = openai_client.chat.completions.create(
    messages=[
      {
        "role": "user",
        "content": prompt,
      }
    ],
    model=model,
    temperature=0.7
  )
  return chat_completion.choices[0].message.content

def classify_with_llm(text):
  prompt = "You are an utterance classifier. Classify into three actions [Tea, Human, None]. Here are some examples:"\
  "u: I want to learn about tea a: Tea\n"\
  "u: I want to talk to someone a: Human\n"\
  "Now classify the following utterance.\n"\
  f"u: {text} a: "
  result = call_openai(prompt)
  return result

def classify_with_voiceflow(text):
  api_key = os.getenv("VOICEFLOW_API_KEY")
  body = {"action": {"type": "text", "payload": text}}
  user_id = uuid4()
  # Start a conversation
  response = requests.post(
      f"https://general-runtime.voiceflow.com/state/user/{user_id}/interact?verbose=true",
      json=body,
      headers={"Authorization": api_key},
  )


  # Log the response
  r = response.json()
  matched_intent = r['request']["payload"]["intent"]["name"]
  return matched_intent

for utterance in ["No more bots, need to talk to a human","Tell me about tea", "Why is the sky blue?" ]:
  print(f"\nUtterance: {utterance}")
  print("LLM intent classification:",classify_with_llm(utterance))
  print("Voiceflow intent classification:",classify_with_voiceflow(utterance))
