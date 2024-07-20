import openai
import pandas as pd
openai_client = openai.OpenAI()
llm_model = "gpt-4o-mini"


def call_openai(prompt, model="gpt-4o-mini"):
	chat_completion = openai_client.chat.completions.create(
		messages=[
			{
				"role": "system",
				"content": "Answer the question concisely and accurately."
			},
			{
				"role": "user",
				"content": prompt,
			}
		],
		model=model,
	)
	return chat_completion.choices[0].message.content


def read_questions(file_path: str = 'questions.txt') -> list:
	with open(file_path) as file:
		lines = file.read().splitlines()
	return lines


questions = read_questions()
user_name = input("What is your name? \n")
answers = []
ratings = []
explanations = []
for q in questions:
	answer = call_openai(q, llm_model)
	rating = input(f"Rate the answer to the question as true or false: {q} \n {answer} \n")
	explanation = input(f"Explain why you rated the answer as {rating}: \n")
	answers.append(answer)
	ratings.append(rating)
	explanations.append(explanation)

df = pd.DataFrame(data=
				  {"questions": questions,
				   "llmAnswers": answers,
				   "userRatings": ratings,
				   "explanations": explanations,
				   "userName": [user_name]*len(answers),
                   "model": [llm_model]*len(answers)})

df.to_csv("annotations.txt", sep=',', index=False)
