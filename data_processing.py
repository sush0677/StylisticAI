import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

# Load the dataset
df = pd.read_excel('updated_fashion_recommendations_scenarios.xlsx')

# Tokenize the scenarios using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Encode the scenarios
df['Scenario_encoded'] = df['Scenario'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=50))

# Encode the recommendations (labels)
label_encoder = LabelEncoder()
df['Recommendation_encoded'] = label_encoder.fit_transform(df['Recommendation'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(df['Scenario_encoded'], df['Recommendation_encoded'], test_size=0.2, random_state=42)

# Pad sequences so that all have the same length
X_train_padded = pad_sequence([torch.tensor(x) for x in X_train], batch_first=True, padding_value=tokenizer.pad_token_id)
X_val_padded = pad_sequence([torch.tensor(x) for x in X_val], batch_first=True, padding_value=tokenizer.pad_token_id)

# Convert to torch tensors
y_train = torch.tensor(y_train.tolist(), dtype=torch.long)
y_val = torch.tensor(y_val.tolist(), dtype=torch.long)

# Save the preprocessed data
torch.save((X_train_padded, X_val_padded, y_train, y_val, label_encoder), 'preprocessed_data.pth')

print("Data preprocessing completed and saved.")
