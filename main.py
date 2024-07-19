import openai
openai_client = openai.OpenAI()

def call_openai(prompt, model="gpt-4o-mini"):
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

def prompt_template(r):
    # add prompt here
    prompt = f""""""
    return prompt

# ratings and classifications based on datasets from https://snap.stanford.edu/data/web-Amazon-links.html
def load_reviews():
    with open("ecommerce_reviews.txt") as f:
        reviews = f.readlines()
        return reviews

def load_classification():
    with open("ratings.txt") as f:
        classifications = f.read().splitlines()
        return classifications
    

def calculate_accuracy(list1, list2):
	return sum([1 for i in range(len(list1)) if list1[i] == list2[i]]) / len(list1)

def predict_review_sentiment():
    reviews = load_reviews()
    classification = load_classification()
    results = []
    for r in reviews:
        prompt = prompt_template(r)
        r = call_openai(prompt)
        results.append(r)

    accuracy = calculate_accuracy(classification,results)
    print("Accuracy:", accuracy)

predict_review_sentiment()