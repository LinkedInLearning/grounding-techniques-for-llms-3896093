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
    prompt = f"""classify the following review into positive or negative.
review: It doesn't show anywhere (that I could see) that this is only one canvas. Very misleading description. I wish I had read all the reviews before ordering! class: negative###
review: The package contained just one stretched canvas and not 6 as the description claims. I had to return this order as Amazon cannot replace it for some reasons. class: negative###
review: I really enjoy these scissors for my inspiration books that I am making (like collage, but in books) and using these different textures these give is just wonderful, makes a great statement with the pictures and sayings. Want more, perfect for any need you have even for gifts as well. Pretty cool! class: positive###
review: {r} class:"""
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