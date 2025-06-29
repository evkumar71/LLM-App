from openai import OpenAI
import pandas as pd
import os
import io
import json
import tiktoken
from collections import defaultdict

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

encoding = tiktoken.get_encoding("cl100k_base")


def convert_json_to_jsonl(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def check_file_format(dataset):
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call") for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")


# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def estimate_cost(dataset):
    # Cost estimations
    # Get the length of the conversation
    conversation_length = []

    for msg in dataset:
        messages = msg["messages"]
        conversation_length.append(num_tokens_from_messages(messages))

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096
    TARGET_EPOCHS = 5
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)

    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in conversation_length
    )
    print(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
    )

    num_tokens = n_epochs * n_billing_tokens_in_dataset

    # gpt-3.5-turbo	$0.0080 / 1K tokens -- need updates, use gpt4o-mini
    cost = (num_tokens / 1000) * 0.0080
    print(cost)


# Retrieve the state of fine-tuning
def get_finetune_results():

    # Status field can contain: running or succeeded or failed, etc.
    state = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Fine-tuning job is running \n {state}")

    # once training is finished, you can retrieve the file in "result_files=[]"
    result_file = "file-QXZ8KvqyGvb8spZES6DUgT"

    file_data = client.files.content(result_file)

    # its binary, so read it and then make it a file like object
    file_data_bytes = file_data.read()
    file_like_object = io.BytesIO(file_data_bytes)

    # # now read as csv to create df
    df = pd.read_csv(file_like_object)
    print(df)


def eval_model(input_model, user_question):
    ## Testing and evaluating -- first use the main model and prompt it:
    ## make sure to change the messages to match the data set we fine-tuned our model with
    response = client.chat.completions.create(
        model=input_model,
        messages=[
            {
                "role": "system",
                "content": "This is a customer support chatbot designed to help with common inquiries.",
                "role": "user",
                "content": user_question,
            }
        ],
    )

    print(
        f" ==== >> {input_model} response: \n {response.choices[0].message.content}"
    )


# ----------- Start here -----------

# convert_json_to_jsonl("./test.json", "./output.jsonl")

data_path = "./output.jsonl"

with open(data_path, "r", encoding="utf-8") as fil:
    dataset = [json.loads(line) for line in fil]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)


check_file_format(dataset)

estimate_cost(dataset)

# Upload file once all validations are successful!
training_file = client.files.create(
    file=open("output.jsonl", "rb"), purpose="fine-tune"
)
print(training_file.id)

# file-KX5samMQEwxLHXDiuyVfzq

"""
# == Next steps: Create a fine-tuned model ===
# Start the fine-tuning job
# After you've started a fine-tuning job, it may take some time to complete. 
# Your job may be queued behind other jobs and training a model 
# can take minutes or hours depending on the model and dataset size.
"""
print("Started fine tuning...")

response = client.fine_tuning.jobs.create(
    training_file=training_file.id,  # get the training file name from above response from when we upload the our jsonl file!
    model="gpt-4.1-mini-2025-04-14",
    hyperparameters={
        "n_epochs": 5  # the number of read throughs -- how many times the file will be read through while fine-tuning
    },
)
print(response.id)

# Response.id
job_id = "ftjob-7dFQrioNHET4Q7PKiC1P055A"

# get_finetune_results()
fine_tuned_model = "ft:gpt-4.1-mini-2025-04-14:personal::BnpFst9a"

query1 = "What types of tea are included in the subscription ?"

# This model response could be hallucinations...
eval_model("gpt-4o-mini", query1)
print("-" * 100)

# "This tuned model repsponse must be good...
eval_model(fine_tuned_model, query1)
print("-" * 100)

query1 = "How do I change my tea preferences for the next shipment ?"

# This model response could be hallucinations...
eval_model("gpt-4o-mini", query1)
print("-" * 100)

# "This tuned model repsponse must be good...
eval_model(fine_tuned_model, query1)
