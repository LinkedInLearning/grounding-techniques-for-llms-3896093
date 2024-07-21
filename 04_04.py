from evaluate import load
from transformers import pipeline
bertscore = load("bertscore") # from paper https://arxiv.org/pdf/1904.09675.pdf
bleu = load("bleu") # from paper https://aclanthology.org/P02-1040.pdf
rouge = load("rouge") # from paper https://aclanthology.org/W04-1013.pdf
entailment_pipe = pipeline("text-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

def calculate_results(generated_results,references):
		for generated in generated_results:
			print("\nStatement: ", generated[0], "\nReference: ", references[0])
			results = bleu.compute(predictions=generated, references=references,max_order=2)
			print("Bleu", results)

			results = rouge.compute(predictions=generated, references=references)
			print("Rouge", results)
			
			results = bertscore.compute(predictions=generated, references=references, lang="en")
			print("Bert Score", results)

			result = entailment_pipe({'text':generated, 'text_pair': references})
			print("NLI Entailment", result)

generated_results = [["the cat sat on a little mat"]]
references = ["the cat sat"]
calculate_results(generated_results, references)

generated_results = [["the number of cards in a standard deck of cards is 52"],  ["54 cards"], ["52 cards"]]
references = ["52 cards in a standard deck of cards"]
calculate_results(generated_results, references)