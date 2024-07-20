from evaluate import load
from transformers import pipeline
bertscore = load("bertscore") # from paper https://arxiv.org/pdf/1904.09675.pdf
bleu = load("bleu") # from paper https://aclanthology.org/P02-1040.pdf
rouge = load("rouge") # from paper https://aclanthology.org/W04-1013.pdf
entailment_pipe = pipeline("text-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")


predictions = [["the number of cards in a standard deck of cards is 52"],  ["54 cards"], ["52 cards"]]
references = ["52 cards in a standard deck of cards"]

for prediction in predictions:
	print("\nStatement: ", prediction[0], "\nReference: ", references[0])
	results = bertscore.compute(predictions=prediction, references=references, lang="en")
	print("Bert Score", results)

	results = bleu.compute(predictions=prediction, references=references)
	print("Bleu", results)

	results = rouge.compute(predictions=prediction, references=references)
	print("Rouge", results)
	result = entailment_pipe({'text':prediction, 'text_pair': references})
	print("NLI Entailment", result)


