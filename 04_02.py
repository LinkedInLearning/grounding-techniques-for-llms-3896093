import openai
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

def answer_generator(question):
	prompt = f"Answer the following question: {question}"
	return call_openai(prompt, llm_model)

def answer_critic(question, answer):
	prompt = f"Rate if the answer to the question is true or false. Provide an explanation.\n" \
			 f"q: What is the clearence of London Bridge? a: 8.9m c: True. Based on the wikipedia article from 2024. ###\n" \
			 f"q: What is the wavelength of visible light? a: 400-700nm c: True. Another acceptable answer is 750-420 terahertz.###\n" \
			 f"q: How many continents are there? a: 10 c: False. By convention there are between 4-7 continents, not 10. ###\n" \
			 f"q: {question} a: {answer} c:"
	return call_openai(prompt, llm_model)

question = "How many cards are there in a standard deck of cards?"
answer = answer_generator(question)
answer = "54"
assessment = answer_critic(question, answer)

question = "What is the capital of France?"
answer = answer_generator(question)
assessment = answer_critic(question, answer)

for i in range(0, 10):
	question = "How many years were between the invention of the radio and chat gpt, provide a number."
	answer = answer_generator(question)
	assessment = answer_critic(question, answer)
	print(answer)
	if "False" in assessment:
		print(answer, assessment)