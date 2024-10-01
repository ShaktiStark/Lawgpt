import torch
import pandas as pd
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load your legal dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('LAWS.csv')

# Encode labels using LabelEncoder
le = LabelEncoder()
df['Output'] = le.fit_transform(df['Output'])

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer and model initialization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(le.classes_))

# Prepare data for training
def prepare_data(df, tokenizer, max_length=128):
    tokenized = tokenizer(df['Text'].tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    labels = torch.tensor(df['Output'].tolist())
    return TensorDataset(tokenized.input_ids, tokenized.attention_mask, labels)

train_dataset = prepare_data(train_df, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Train the classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(3):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Inference loop
while True:
    user_input = input("Please describe your legal situation (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break

    # Tokenize user input
    user_tokens = tokenizer.encode_plus(
        user_input,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Make predictions for user input
    model.eval()
    with torch.no_grad():
        input_ids = user_tokens['input_ids'].to(device)
        attention_mask = user_tokens['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        recommended_legal_case = le.inverse_transform(predictions.cpu().numpy())[0]

    print(f"Recommended Legal Case: {recommended_legal_case}")

print("Exiting the program.")