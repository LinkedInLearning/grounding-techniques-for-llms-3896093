from openai import OpenAI
import pandas as pd
from datasets import load_dataset

client = OpenAI()
training_dataset, validation_dataset = load_dataset('xlangai/spider', split=['train[0:50]', 'validation[:50]'])
llm_model = "gpt-3.5-turbo"

def format_finetuning_dataset():
	system_prompt = "Spider knows SQL"
	dataset_dict = {"training": training_dataset, "validation": validation_dataset}

	def format_dataset(dataset_dict):
		formatted_data = []
		for dataset_name, dataset in dataset_dict.items():
			for row in dataset:
				question = row['question']
				query = row['query']
				message = {
					"messages": [
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": question},
						{"role": "assistant", "content": query}
					]
				}
				formatted_data.append(message)
			formatted_df = pd.DataFrame(formatted_data)
			formatted_df.to_json(f"{dataset_name}_dataset.jsonl", orient='records', lines=True)

	format_dataset(dataset_dict)

def call_openai(system_prompt,prompt, model=llm_model):
	chat_completion = client.chat.completions.create(
		messages=[
			{
				"role": "system",
				"content": system_prompt
			},
			{
				"role": "user",
				"content": prompt,
			}
		],
		model=model,
	)
	return chat_completion.choices[0].message.content


def evaluate_dataset(model, is_finetuned=False):
	results = []
	for row in validation_dataset:
		if is_finetuned:
			system_prompt = "Spider knows SQL"
			prompt = row['question']
		else:
			system_prompt = "Follow the user commands."

			prompt = f"Translate the text command to SQL. Only output the SQL.\n" \
					 f"Text: Return all data from the customers table  SQL: Select * from Table ###\n"\
					 f"Translate the text command to SQL. Only output the SQL.\n"\
					 f"Text: {row['question']} SQL: "
		result = call_openai(system_prompt,prompt, model)
		result = result.replace(";", "")
		results.append(result)

	validation_df = pd.DataFrame(validation_dataset)
	validation_df['predicted_query'] = results

	validation_df.to_csv(f"{model}_validation_accuracy_results.csv", index=False)
	accuracy = (validation_df['predicted_query'] == validation_df['query']).mean()
	print(f"Accuracy: {accuracy}")

def upload_dataset_to_openai():
	validation_file = client.files.create(
		file=open("validation_dataset.jsonl", "rb"),
		purpose="fine-tune"
	)
	training_file = client.files.create(
		file=open("training_dataset.jsonl", "rb"),
		purpose="fine-tune",
	)
	return training_file, validation_file


def train_openai_model(training_file, validation_file):
	results = client.fine_tuning.jobs.create(
		training_file=training_file.id,
		validation_file=validation_file.id,
		model="gpt-3.5-turbo"
	)

evaluate_dataset(model = llm_model)
format_finetuning_dataset()
training_file, validation_file = upload_dataset_to_openai()
train_openai_model(training_file, validation_file)
#evaluate_dataset(model="ft:gpt-3.5-turbo-0125:voiceflow::9ndVtbjB", is_finetuned=True)

