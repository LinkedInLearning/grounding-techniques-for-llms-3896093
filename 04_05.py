from collections import defaultdict
from datasets import load_dataset
import openai
import pandas as pd
train_ds, validation_ds, test_ds = load_dataset('imdb', split=['train[10:13]+train[-2:]', 'train[15:15]+train[-15:-10]', 'test[:25]+test[-25:]'])
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
		model=model, temperature=0.1, max_tokens=10
	)
	return chat_completion.choices[0].message.content

def prompts_template(train_ds, review):
	prompts = {}
	prompt1 = f"""Is this review positive or negative? Only output the number 1 for positive and 0 for negative.
Review: {review}"""
	prompts["Instruction"] = prompt1

	prompt2 = """Is this review positive or negative? Only output the number 1 for positive and 0 for negative.\n"""
	prompt3 = ""
	for row in iter(train_ds):
		text, label = row["text"], row["label"]
		prompt3 += f"Review: {text} rating: {label} ###\n\n"
	prompt3 += f"Review: {review} rating: "
	prompt2 += prompt3
	prompts["ICL"] = prompt3
	prompts["InstructionICL"] = prompt2
	return prompts


def run_evaluation(dataset_to_test, llm_model):
	results = defaultdict(list)
	for row in iter(dataset_to_test):
		text, label = row["text"], row["label"]
		prompts = prompts_template(train_ds,text)
		for name, prompt in prompts.items():
			answer = call_openai(prompt, llm_model)
			results[name].append(answer)

	df = pd.DataFrame(data=
					  {"reviews": dataset_to_test['text'],
					   "groundTruths": map(str, dataset_to_test['label']),
					   **results,
					   "model": [llm_model]*len(dataset_to_test)},
	)
	df.to_csv("results.csv", sep=',', index=False)
	return df, results.keys()


def evaluate_accuracy(df,model_names):
	accuracy_df = pd.DataFrame()
	accuracy_df["model"] = [llm_model]
	for name in model_names:
		accuracy_df[name] = (df["groundTruths"] == df[name]).mean()
	print(accuracy_df)

dataset_to_test = validation_ds
#dataset_to_test = test_ds
df, model_names = run_evaluation(dataset_to_test, llm_model)
evaluate_accuracy(df,model_names)