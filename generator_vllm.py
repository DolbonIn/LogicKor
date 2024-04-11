import os
import argparse
import pandas as pd
import requests
import time
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--template', help=' : Template File Location', default='./templates/template-EEVE.json')
parser.add_argument('--batch_size', help=' : Batch Size', default=2, type=int)
parser.add_argument('--num_workers', help=' : Number of DataLoader Workers', default=2, type=int)
parser.add_argument('--api_endpoint', help=' : YOUR_VLLM_ENDPOINT')
parser.add_argument('--model_name', help=' : YOUR_MODEL')
args = parser.parse_args()

df_config = pd.read_json(args.template, typ='series')
SINGLE_TURN_TEMPLATE = df_config.iloc[0]
MULTI_TURN_TEMPLATE = df_config.iloc[1]

API_ENDPOINT = f"{args.api_endpoint}/v1/chat/completions"
df_questions = pd.read_json('questions.jsonl', lines=True)

def format_single_turn_question(question):
    return SINGLE_TURN_TEMPLATE.format(question[0])

class QuestionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]

def collate_fn(batch):
    return pd.DataFrame(batch)

def process_batch(batch):
    single_turn_questions = batch['questions'].apply(lambda x: format_single_turn_question(x))
    print(f"Batch size: {len(batch)}")
    print(f"Single turn questions: {single_turn_questions}")

    def generate(prompt, max_retries=3, delay=5):
        payload = {
            "model": f"{args.model_name}",
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "early_stopping": True,
            "best_of": 4,
            "use_beam_search": True,
            "skip_special_tokens": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant. You will be given a task. You must generate a detailed and long answer in korean."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        retries = 0
        while retries < max_retries:
            response = requests.post(API_ENDPOINT, json=payload, timeout=600)

            if response.status_code == 200:
                try:
                    result = response.json()
                    print(f"Prompt: {prompt}")
                    print(f"API response: {result}")

                    if 'choices' in result:
                        return result['choices'][0]['message']['content'].strip().replace("<|im_end|>", "")
                    else:
                        print(f"Error: Unexpected API response format: {result}")
                        return ""
                except requests.exceptions.JSONDecodeError as e:
                    print(f"Error: Failed to decode API response as JSON: {e}")
                    print(f"Response text: {response.text}")
                    return ""
            elif response.status_code == 524:
                retries += 1
                if retries == max_retries:
                    print(f"Error: Max retries reached for API request. Response status code: {response.status_code}")
                    print(f"Response text: {response.text}")
                    return ""
                else:
                    print(f"Warning: API request timed out. Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(delay)
            else:
                print(f"Error: Unexpected API response status code: {response.status_code}")
                print(f"Response text: {response.text}")
                return ""

    single_turn_outputs = []
    s = 0
    for prompt in single_turn_questions.tolist():
        output = generate(prompt)
        single_turn_outputs.append(output)
        s = s + 1
        print(f"Single turn output {s}: {output}")

    def format_multi_turn_question(row):
        print(f"Row name: {row.name}, Single turn outputs length: {len(single_turn_outputs)}")
        if len(row['questions']) < 2:
            print(f"Warning: Insufficient questions for multi-turn format in row: {row}")
            return ""

        return MULTI_TURN_TEMPLATE.format(
            row['questions'][0], single_turn_outputs[row.name], row['questions'][1]
        )

    multi_turn_batch = batch[batch['questions'].apply(lambda x: len(x) >= 2)]
    print(f"Multi-turn batch size: {len(multi_turn_batch)}")
    multi_turn_questions = multi_turn_batch.apply(format_multi_turn_question, axis=1)
    print(f"Multi-turn questions: {multi_turn_questions}")

    multi_turn_outputs = []
    i = 0
    for prompt in multi_turn_questions.tolist():
        if prompt:  # Skip empty prompts
            output = generate(prompt)
            multi_turn_outputs.append(output)
            i = i + 1
            print(f"Multi-turn output {i}: {output}")

    print(f"Single turn outputs: {single_turn_outputs}")
    print(f"Multi-turn outputs: {multi_turn_outputs}")

    return pd.DataFrame({
        'id': batch['id'],
        'category': batch['category'],
        'questions': batch['questions'],
        'outputs': [(single_turn_outputs[i], multi_turn_outputs[j] if j < len(multi_turn_outputs) else "") for i, j in enumerate(range(len(batch)))],
        'references': batch['references']
    })

def process_data(df_questions, batch_size, num_workers):
    dataset = QuestionDataset(df_questions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=None,
        pin_memory=True
    )

    with ThreadPoolExecutor() as executor:
        try:
            results = list(executor.map(process_batch, dataloader))
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Shutting down the executor...")
            executor.shutdown(wait=True)
            print("Executor shut down. Exiting...")
            return

    df_output = pd.concat(results, ignore_index=True)
    df_output.to_json(
        f'{args.model_name}.jsonl',
        orient='records',
        lines=True,
        force_ascii=False
    )

if __name__ == '__main__':
    process_data(df_questions, args.batch_size, args.num_workers)
