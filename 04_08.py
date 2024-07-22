import requests
import os
import io
from dspy.datasets import HotPotQA

# Load the HotPotQA dataset
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# Voiceflow API endpoint and headers
url = "https://api.voiceflow.com/v1/knowledge-base/docs/upload?maxChunkSize=1000"
headers = {
    "accept": "application/json",
    "Authorization": os.getenv("VOICEFLOW_API_KEY")
}

# Iterate over each row in the dev dataset
for row in dataset.dev:
    # Prepare the data for uploading
    file_content = f"Q: {row.question}\nA: {row.answer}"
    combined_title = "-".join(list(row.gold_titles))
    files = {
        'file': (combined_title + ".txt", file_content)
    }
    
    response = requests.post(url, headers=headers, files=files)
    print(response.text)