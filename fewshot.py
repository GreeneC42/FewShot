# this is the distilbert model
# import torch
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score
# from torch.utils.data import DataLoader, Dataset
# from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW
# from sklearn.model_selection import train_test_split
# from torch import nn
# import time
# import torch.nn.functional as F
# import numpy as np
# import itertools

# # Hyperparameters
# MAX_LEN = 512
# TRAIN_BATCH_SIZE = 8
# VALID_BATCH_SIZE = 4
# EPOCHS = 22
# LEARNING_RATE = 1e-03
# num_classes = 4
# random_state = 42
# random_prompt_length = 10
#
# # random seed
# seed = 999999999
# torch.manual_seed(seed)
# np.random.seed(seed)
#
# # Move the classifiers to device (GPU/CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Start measuring time
# start_time = time.time()
#
#
# class TextClassificationDataset(Dataset):
#     def __init__(self, encodings_rationale, encodings_text, labels):
#         self.encodings_rationale = encodings_rationale
#         self.encodings_text = encodings_text
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         input_ids_rationale = self.encodings_rationale["input_ids"][idx]
#         attention_mask_rationale = self.encodings_rationale["attention_mask"][idx]
#         input_ids_text = self.encodings_text["input_ids"][idx]
#         attention_mask_text = self.encodings_text["attention_mask"][idx]
#         label = self.labels[idx]
#
#         return {
#             "input_ids_rationale": input_ids_rationale,
#             "attention_mask_rationale": attention_mask_rationale,
#             "input_ids_text": input_ids_text,
#             "attention_mask_text": attention_mask_text,
#             "labels": label
#         }
#
#
# # Read CSV
# data = pd.read_csv("newest_fewshot_dataset.csv")
#
# # Clean labels
# data['Type'] = data['Type'].str.strip()
#
# # Define labels
# label_map = {'Fraud': 0, 'Phishing': 1, 'False Positives': 2, 'Commercial Spam': 3}
#
# # Check the unique labels to get rif of the error
# unique_labels = data['Type'].unique()
# for label in unique_labels:
#     if label not in label_map:
#         print(f"Label '{label}' not found in label_map.")
#
# # Keep clean labels
# all_labels = [label_map[label] for label in data['Type']]
#
# # Separate rationale and text
# all_rationale = data['Rational_with_label'].tolist()
# all_text = data['Text'].tolist()
#
# # Split the data
# train_text, test_text, train_rationale, test_rationale, train_labels, test_labels = train_test_split(
#     all_text, all_rationale, all_labels, train_size=0.2, test_size=0.2, random_state=random_state)
# print(f"Test Text: {test_text}")
# print(f"Test Rationale: {test_rationale}")
#
# # DistilBERT tokenizer
# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
#
# # Tokenize rationale and text
# train_encodings_rationale = tokenizer(train_rationale, truncation=True, padding=True, max_length=MAX_LEN,
#                                       return_tensors="pt")
# train_encodings_text = tokenizer(train_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
# test_encodings_text = tokenizer(test_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
#
# # DistilBERT model
# distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
#
# class DistilBERTClassifier(nn.Module):
#     def __init__(self, distilbert_model, num_classes):
#         super(DistilBERTClassifier, self).__init__()
#         self.distilbert_model = distilbert_model
#         self.pre_classifier = torch.nn.Linear(768, 256)
#         self.relu_func = nn.ReLU()
#         self.bn = nn.BatchNorm1d(256)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.classifier = torch.nn.Linear(256, num_classes)
#         self.random_prompt = nn.Parameter(torch.randn((random_prompt_length, 768), requires_grad=True))
#         self.attention_weights = nn.Parameter(torch.randn(1, 1, 1))
#
#         # Freeze DistilBERT parameters
#         for param in distilbert_model.parameters():
#             param.requires_grad = False
#
#     def forward(self, input_ids_rationale, attention_mask_rationale, input_ids_text, attention_mask_text):
#         output_rationale = self.distilbert_model(input_ids=input_ids_rationale, attention_mask=attention_mask_rationale)
#         output_text = self.distilbert_model(input_ids=input_ids_text, attention_mask=attention_mask_text)
#
#         # Truncate the last 10 characters of text and replace them with the prompt, leave the rationale
#         output_rationale = output_rationale['last_hidden_state']
#         truncated_output_text = output_text['last_hidden_state'][:, :-random_prompt_length, :]
#
#         random_prompt = self.random_prompt.unsqueeze(0).expand(output_rationale.size(0), -1, -1)
#
#         # add the prompt to the end of the text
#         concat_prompt_text = torch.cat((truncated_output_text, random_prompt), dim=1)
#
#         # Attention to adjust contributions of prompt
#         # attention_weights = F.softmax(self.attention_weights, dim=-1)
#         # weighted_random_prompt = attention_weights * random_prompt
#
#         concatenated_output = torch.cat((output_rationale, concat_prompt_text, random_prompt), dim=1)
#
#         max_output = concatenated_output.mean(dim=1)
#
#         lin_lay = self.pre_classifier(max_output)
#         lin_lay = F.relu(lin_lay)
#
#         # Apply batch normalization only if the batch size is greater than 1
#         if max_output.size(0) > 1:
#             lin_lay = self.bn(lin_lay)
#
#         dropout_output = self.dropout(lin_lay)
#         output = self.classifier(dropout_output)
#
#         return output
#
#
# # Create datasets for rationale and text
# train_dataset = TextClassificationDataset(train_encodings_rationale, train_encodings_text, train_labels)
# test_dataset = TextClassificationDataset(test_encodings_text, test_encodings_text, test_labels)
#
# # Data loaders
# train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
#
# # classifiers
# classifier = DistilBERTClassifier(distilbert_model, num_classes=num_classes)
# classifier.to(device)
#
#
# # Parameters gradient true
# model_params = [param for param in classifier.parameters() if param.requires_grad]
#
#
# for name, param in classifier.named_parameters():
#     print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
#
# optimizer = AdamW(model_params, lr=LEARNING_RATE, weight_decay=5e-4)
#
#
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#
#
# def train(model, data_loader, optimizer, device):
#     n_correct = 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#     tr_loss = 0
#     loss_function = torch.nn.CrossEntropyLoss()
#     model.train()
#
#     for batch in data_loader:
#         input_ids_rationale = batch['input_ids_rationale'].to(device)
#         attention_mask_rationale = batch['attention_mask_rationale'].to(device)
#         input_ids_text = batch['input_ids_text'].to(device)
#         attention_mask_text = batch['attention_mask_text'].to(device)
#         labels = batch['labels'].to(device)
#
#         optimizer.zero_grad()
#         outputs = model(input_ids_rationale=input_ids_rationale, attention_mask_rationale=attention_mask_rationale,
#                         input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
#         loss = loss_function(outputs, labels)
#         tr_loss += loss.item()
#
#         _, preds = torch.max(outputs, dim=1)
#         n_correct += torch.sum(preds == labels).item()
#         nb_tr_steps += 1
#         nb_tr_examples += labels.size(0)
#
#         loss.backward()
#         optimizer.step()
#         # print("Random Prompt Gradients:", classifier.random_prompt.grad)  # Print gradients
#         # print("Classifier Gradients: ", classifier.pre_classifier.weight.grad)
#
#     epoch_loss = tr_loss / nb_tr_steps
#     epoch_accu = (n_correct * 100) / nb_tr_examples
#
#     return epoch_loss, epoch_accu
#
#
#
# ##
# test_accuracies = []
# for epoch in range(EPOCHS):
#     loss, acc = train(classifier, train_loader, optimizer, device)
#     scheduler.step()
#
#     print(f"Epoch {epoch + 1}/{EPOCHS}")
#     print(f"Average Loss: {loss:.4f}")
#     print(f"Training Accuracy: {acc:.4f}")
#
#     # from here shift tab to bring it back, in epoch loop for best epoch
#     # Testing
#     # classifier = DistilBERTClassifier(distilbert_model, num_classes=num_classes)
#     # classifier.to(device)
#     classifier.eval()
#     predictions = []
#     actual_labels = []
#
#     with torch.no_grad():
#         for batch in test_loader:
#             # Test with only text
#             input_ids_text = batch['input_ids_text'].to(device)
#             attention_mask_text = batch['attention_mask_text'].to(device)
#
#             # Pass text input to both branches
#             outputs = classifier(input_ids_rationale=input_ids_text, attention_mask_rationale=attention_mask_text,
#                                  input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
#
#             _, preds = torch.max(outputs, dim=1)
#             predictions.extend(preds.cpu().tolist())
#             actual_labels.extend(batch['labels'])
#
#     # accuracy and f1
#     f1_test = f1_score(actual_labels, predictions, average='weighted')
#
#     accuracy = accuracy_score(actual_labels, predictions)
#     print(f"Test accuracy: {accuracy:.4f}")
#
#     print(f"Test F1 Score: {f1_test:.4f}")
#     test_accuracies.append((accuracy, epoch))
#     # and then here ends the shift tab
#
# # End time
# end_time = time.time()
# total_time = end_time - start_time
#
#
# sorted_accuracies = sorted(test_accuracies, key=lambda x: x[0], reverse=True)
#
# # Print the sorted list
# print("\nSorted Test Accuracies:")
# for acc, epoch in sorted_accuracies:
#     print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")
#
# print(f"Total execution time: {total_time:.2f} seconds")
#



#
#
# # This is the RoBERTa Model
# import torch
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score
# from torch.utils.data import DataLoader, Dataset
# from transformers import RobertaTokenizerFast, RobertaModel, AdamW
# from sklearn.model_selection import train_test_split
# from torch import nn
# import time
# import torch.nn.functional as F
# import numpy as np
# import itertools
#
# # Hyperparameters
# MAX_LEN = 512
# TRAIN_BATCH_SIZE = 8
# VALID_BATCH_SIZE = 4
# EPOCHS = 22
# LEARNING_RATE = 1e-03
# num_classes = 4
# random_state = 42
# random_prompt_length = 10
#
# # random seed
# seed = 999999999
# torch.manual_seed(seed)
# np.random.seed(seed)
#
# # Move the classifiers to device (GPU/CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Start measuring time
# start_time = time.time()
#
#
# class TextClassificationDataset(Dataset):
#     def __init__(self, encodings_rationale, encodings_text, labels):
#         self.encodings_rationale = encodings_rationale
#         self.encodings_text = encodings_text
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         input_ids_rationale = self.encodings_rationale["input_ids"][idx]
#         attention_mask_rationale = self.encodings_rationale["attention_mask"][idx]
#         input_ids_text = self.encodings_text["input_ids"][idx]
#         attention_mask_text = self.encodings_text["attention_mask"][idx]
#         label = self.labels[idx]
#
#         return {
#             "input_ids_rationale": input_ids_rationale,
#             "attention_mask_rationale": attention_mask_rationale,
#             "input_ids_text": input_ids_text,
#             "attention_mask_text": attention_mask_text,
#             "labels": label
#         }
#
#
# # Read CSV
# data = pd.read_csv("newest_fewshot_dataset.csv")
#
# # Clean labels
# data['Type'] = data['Type'].str.strip()
#
# # Define labels
# label_map = {'Fraud': 0, 'Phishing': 1, 'False Positives': 2, 'Commercial Spam': 3}
#
# # Check the unique labels to get rif of the error
# unique_labels = data['Type'].unique()
# for label in unique_labels:
#     if label not in label_map:
#         print(f"Label '{label}' not found in label_map.")
#
# # Keep clean labels
# all_labels = [label_map[label] for label in data['Type']]
#
# # Separate rationale and text
# all_rationale = data['Rational_with_label'].tolist()
# all_text = data['Text'].tolist()
#
# # Split the data
# train_text, test_text, train_rationale, test_rationale, train_labels, test_labels = train_test_split(
#     all_text, all_rationale, all_labels, train_size=0.2, test_size=0.2, random_state=random_state)
# print(f"Test Text: {test_text}")
# print(f"Test Rationale: {test_rationale}")
#
# # RoBERTa tokenizer
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
#
# # Tokenize rationale and text
# train_encodings_rationale = tokenizer(train_rationale, truncation=True, padding=True, max_length=MAX_LEN,
#                                       return_tensors="pt")
# train_encodings_text = tokenizer(train_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
# test_encodings_text = tokenizer(test_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
#
# # RoBERTa model
# roberta_model = RobertaModel.from_pretrained("roberta-base")
#
# class RoBERTaClassifier(nn.Module):
#     def __init__(self, roberta_model, num_classes):
#         super(RoBERTaClassifier, self).__init__()
#         self.roberta_model = roberta_model
#         self.pre_classifier = torch.nn.Linear(768, 256)
#         self.relu_func = nn.ReLU()
#         self.bn = nn.BatchNorm1d(256)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.classifier = torch.nn.Linear(256, num_classes)
#         self.random_prompt = nn.Parameter(torch.randn((random_prompt_length, 768), requires_grad=True))
#         self.attention_weights = nn.Parameter(torch.randn(1, 1, 1))
#
#         # Freeze RoBERTa parameters
#         for param in roberta_model.parameters():
#             param.requires_grad = False
#
#     def forward(self, input_ids_rationale, attention_mask_rationale, input_ids_text, attention_mask_text):
#         output_rationale = self.roberta_model(input_ids=input_ids_rationale, attention_mask=attention_mask_rationale)
#         output_text = self.roberta_model(input_ids=input_ids_text, attention_mask=attention_mask_text)
#
#         # Truncate the last 10 characters of text and replace them with the prompt, leave the rationale
#         output_rationale = output_rationale['last_hidden_state']
#         truncated_output_text = output_text['last_hidden_state'][:, :-random_prompt_length, :]
#
#         random_prompt = self.random_prompt.unsqueeze(0).expand(output_rationale.size(0), -1, -1)
#
#         # add the prompt to the end of the text
#         concat_prompt_text = torch.cat((truncated_output_text, random_prompt), dim=1)
#
#         # Attention to adjust contributions of prompt
#         # attention_weights = F.softmax(self.attention_weights, dim=-1)
#         # weighted_random_prompt = attention_weights * random_prompt
#
#         concatenated_output = torch.cat((output_rationale, concat_prompt_text, random_prompt), dim=1)
#
#         max_output = concatenated_output.mean(dim=1)
#
#         lin_lay = self.pre_classifier(max_output)
#         lin_lay = F.relu(lin_lay)
#
#         # Apply batch normalization only if the batch size is greater than 1
#         if max_output.size(0) > 1:
#             lin_lay = self.bn(lin_lay)
#
#         dropout_output = self.dropout(lin_lay)
#         output = self.classifier(dropout_output)
#
#         return output
#
#
# # Create datasets for rationale and text
# train_dataset = TextClassificationDataset(train_encodings_rationale, train_encodings_text, train_labels)
# test_dataset = TextClassificationDataset(test_encodings_text, test_encodings_text, test_labels)
#
# # Data loaders
# train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
#
# # classifiers
# classifier = RoBERTaClassifier(roberta_model, num_classes=num_classes)
# classifier.to(device)
#
#
# # Parameters gradient true
# model_params = [param for param in classifier.parameters() if param.requires_grad]
#
#
# for name, param in classifier.named_parameters():
#     print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
#
# optimizer = AdamW(model_params, lr=LEARNING_RATE, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#
#
# def train(model, data_loader, optimizer, device):
#     n_correct = 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#     tr_loss = 0
#     loss_function = torch.nn.CrossEntropyLoss()
#     model.train()
#
#     for batch in data_loader:
#         input_ids_rationale = batch['input_ids_rationale'].to(device)
#         attention_mask_rationale = batch['attention_mask_rationale'].to(device)
#         input_ids_text = batch['input_ids_text'].to(device)
#         attention_mask_text = batch['attention_mask_text'].to(device)
#         labels = batch['labels'].to(device)
#
#         optimizer.zero_grad()
#         outputs = model(input_ids_rationale=input_ids_rationale, attention_mask_rationale=attention_mask_rationale,
#                         input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
#         loss = loss_function(outputs, labels)
#         tr_loss += loss.item()
#
#         _, preds = torch.max(outputs, dim=1)
#         n_correct += torch.sum(preds == labels).item()
#         nb_tr_steps += 1
#         nb_tr_examples += labels.size(0)
#
#         loss.backward()
#         optimizer.step()
#         # print("Random Prompt Gradients:", classifier.random_prompt.grad)  # Print gradients
#         # print("Classifier Gradients: ", classifier.pre_classifier.weight.grad)
#
#     epoch_loss = tr_loss / nb_tr_steps
#     epoch_accu = (n_correct * 100) / nb_tr_examples
#
#     return epoch_loss, epoch_accu
#
#
#
# ##
# test_accuracies = []
# for epoch in range(EPOCHS):
#     loss, acc = train(classifier, train_loader, optimizer, device)
#     scheduler.step()
#
#     print(f"Epoch {epoch + 1}/{EPOCHS}")
#     print(f"Average Loss: {loss:.4f}")
#     print(f"Training Accuracy: {acc:.4f}")
#
#     # from here shift tab to bring it back, in epoch loop for best epoch
#     # Testing
#     # classifier = RoBERTaClassifier(roberta_model, num_classes=num_classes)
#     # classifier.to(device)
#     classifier.eval()
#     predictions = []
#     actual_labels = []
#
#     with torch.no_grad():
#         for batch in test_loader:
#             # Test with only text
#             input_ids_text = batch['input_ids_text'].to(device)
#             attention_mask_text = batch['attention_mask_text'].to(device)
#
#             # Pass text input to both branches
#             outputs = classifier(input_ids_rationale=input_ids_text, attention_mask_rationale=attention_mask_text,
#                                  input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
#
#             _, preds = torch.max(outputs, dim=1)
#             predictions.extend(preds.cpu().tolist())
#             actual_labels.extend(batch['labels'])
#
#     # accuracy and f1
#     f1_test = f1_score(actual_labels, predictions, average='weighted')
#
#     accuracy = accuracy_score(actual_labels, predictions)
#     print(f"Test accuracy: {accuracy:.4f}")
#
#     print(f"Test F1 Score: {f1_test:.4f}")
#     test_accuracies.append((accuracy, epoch))
#     # and then here ends the shift tab
#
# # End time
# end_time = time.time()
# total_time = end_time - start_time
#
#
# sorted_accuracies = sorted(test_accuracies, key=lambda x: x[0], reverse=True)
#
# # Print the sorted list
# print("\nSorted Test Accuracies:")
# for acc, epoch in sorted_accuracies:
#     print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")
#
# print(f"Total execution time: {total_time:.2f} seconds")
#





# # This is the BERT model
# Since this is probably going to get messed up, the 90% accuracy is the random_prompt_text.py file on fry
# The other file is random_prompt_1_optimizer on pycharm
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, AdamW
from sklearn.model_selection import train_test_split
from torch import nn
import time
import torch.nn.functional as F
import numpy as np
import itertools

# Hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 22
LEARNING_RATE = 1e-03
num_classes = 4
random_state = 42
random_prompt_length = 10
num_shot = 5

# random seed
seed = 999999999
torch.manual_seed(seed)
np.random.seed(seed)

# Move the classifiers to device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start measuring time
start_time = time.time()


class TextClassificationDataset(Dataset):
    def __init__(self, encodings_rationale, encodings_text, labels):
        self.encodings_rationale = encodings_rationale
        self.encodings_text = encodings_text
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids_rationale = self.encodings_rationale["input_ids"][idx]
        attention_mask_rationale = self.encodings_rationale["attention_mask"][idx]
        input_ids_text = self.encodings_text["input_ids"][idx]
        attention_mask_text = self.encodings_text["attention_mask"][idx]
        label = self.labels[idx]

        return {
            "input_ids_rationale": input_ids_rationale,
            "attention_mask_rationale": attention_mask_rationale,
            "input_ids_text": input_ids_text,
            "attention_mask_text": attention_mask_text,
            "labels": label
        }


# Read CSV
data = pd.read_csv("newest_fewshot_dataset.csv")

# Clean labels
data['Type'] = data['Type'].str.strip()

# Define labels
label_map = {'Fraud': 0, 'Phishing': 1, 'False Positives': 2, 'Commercial Spam': 3}

# Check the unique labels to get rif of the error
unique_labels = data['Type'].unique()
for label in unique_labels:
    if label not in label_map:
        print(f"Label '{label}' not found in label_map.")

# Keep clean labels
all_labels = [label_map[label] for label in data['Type']]

# Separate rationale and text
all_rationale = data['Rational_with_label'].tolist()
all_text = data['Text'].tolist()

# Split the data
train_text, test_text, train_rationale, test_rationale, train_labels, test_labels = train_test_split(
    all_text, all_rationale, all_labels, train_size=num_shot, test_size=0.2, random_state=random_state)
print(f"Test Text: {test_text}")
print(f"Test Rationale: {test_rationale}")

# BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize rationale and text
train_encodings_rationale = tokenizer(train_rationale, truncation=True, padding=True, max_length=MAX_LEN,
                                      return_tensors="pt")
train_encodings_text = tokenizer(train_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
test_encodings_text = tokenizer(test_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

# BERT model
bert_model = BertModel.from_pretrained("bert-base-uncased")

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert_model = bert_model
        self.pre_classifier = torch.nn.Linear(768, 256)
        self.relu_func = nn.ReLU()
        self.bn = nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(256, num_classes)
        self.random_prompt = nn.Parameter(torch.randn((random_prompt_length, 768), requires_grad=True))
        self.attention_weights = nn.Parameter(torch.randn(1, 1, 1))

        # Freeze BERT parameters
        for param in bert_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids_rationale, attention_mask_rationale, input_ids_text, attention_mask_text):
        output_rationale = self.bert_model(input_ids=input_ids_rationale, attention_mask=attention_mask_rationale)
        output_text = self.bert_model(input_ids=input_ids_text, attention_mask=attention_mask_text)

        # Truncate the last 10 characters of text and replace them with the prompt, leave the rationale
        output_rationale = output_rationale['last_hidden_state']
        truncated_output_text = output_text['last_hidden_state'][:, :-random_prompt_length, :]

        random_prompt = self.random_prompt.unsqueeze(0).expand(output_rationale.size(0), -1, -1)

        # add the prompt to the end of the text
        concat_prompt_text = torch.cat((truncated_output_text, random_prompt), dim=1)


        # # Attention to adjust contributions of prompt
        # attention_weights = F.softmax(self.attention_weights, dim=-1)
        # weighted_random_prompt = attention_weights * random_prompt

        concatenated_output = torch.cat((output_rationale, concat_prompt_text, random_prompt), dim=1)

        max_output = concatenated_output.mean(dim=1)

        lin_lay = self.pre_classifier(max_output)
        lin_lay = F.relu(lin_lay)

        # Apply batch normalization only if the batch size is greater than 1
        if max_output.size(0) > 1:
            lin_lay = self.bn(lin_lay)

        dropout_output = self.dropout(lin_lay)
        output = self.classifier(dropout_output)

        return output


# Create datasets for rationale and text
train_dataset = TextClassificationDataset(train_encodings_rationale, train_encodings_text, train_labels)
test_dataset = TextClassificationDataset(test_encodings_text, test_encodings_text, test_labels)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# classifiers
classifier = BERTClassifier(bert_model, num_classes=num_classes)
classifier.to(device)


# Parameters gradient true
model_params = [param for param in classifier.parameters() if param.requires_grad]


for name, param in classifier.named_parameters():
    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

optimizer = AdamW(model_params, lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train(model, data_loader, optimizer, device):
    n_correct = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_loss = 0
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()

    for batch in data_loader:
        input_ids_rationale = batch['input_ids_rationale'].to(device)
        attention_mask_rationale = batch['attention_mask_rationale'].to(device)
        input_ids_text = batch['input_ids_text'].to(device)
        attention_mask_text = batch['attention_mask_text'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids_rationale=input_ids_rationale, attention_mask_rationale=attention_mask_rationale,
                        input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
        loss = loss_function(outputs, labels)
        tr_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        n_correct += torch.sum(preds == labels).item()
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        loss.backward()
        optimizer.step()
        # print("Random Prompt Gradients:", classifier.random_prompt.grad)  # Print gradients
        # print("Classifier Gradients: ", classifier.pre_classifier.weight.grad)

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    return epoch_loss, epoch_accu



##
test_accuracies = []
test_f1s = []
for epoch in range(EPOCHS):
    loss, acc = train(classifier, train_loader, optimizer, device)
    scheduler.step()

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Average Loss: {loss:.4f}")
    print(f"Training Accuracy: {acc:.4f}")

    # from here shift tab to bring it back, in epoch loop for best epoch
    # Testing
    # classifier = BERTClassifier(bert_model, num_classes=num_classes)
    # classifier.to(device)
    classifier.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Test with only text
            input_ids_text = batch['input_ids_text'].to(device)
            attention_mask_text = batch['attention_mask_text'].to(device)

            # Pass text input to both branches
            outputs = classifier(input_ids_rationale=input_ids_text, attention_mask_rationale=attention_mask_text,
                                 input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(batch['labels'])

    # accuracy and f1
    f1_test = f1_score(actual_labels, predictions, average='weighted')

    accuracy = accuracy_score(actual_labels, predictions)
    print(f"Test accuracy: {accuracy:.4f}")

    print(f"Test F1 Score: {f1_test:.4f}")
    test_accuracies.append((accuracy, epoch))
    test_f1s.append((f1_test, epoch))
    # and then here ends the shift tab

# End time
end_time = time.time()
total_time = end_time - start_time


sorted_accuracies = sorted(test_accuracies, key=lambda x: x[0], reverse=True)
sorted_f1s = sorted(test_f1s, key=lambda x: x[0], reverse=True)

# Print the sorted list
print("\nSorted Test Accuracies:")
for acc, epoch in sorted_accuracies:
    print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")


print("\nSorted Test f1s: ")
for f1, epoch in sorted_f1s:
    print(f"Epoch{epoch}: Test f1 = {f1:.4f}")

print(f"Total execution time: {total_time:.2f} seconds")




###############################################################
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, AdamW
from sklearn.model_selection import train_test_split
from torch import nn
import time
import torch.nn.functional as F
import numpy as np

# Hyperparameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 22
LEARNING_RATE = 1e-03
num_classes = 4
random_state = 42
num_shot = 5

# random seed
seed = 999999999
torch.manual_seed(seed)
np.random.seed(seed)

# Move the classifiers to device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start measuring time
start_time = time.time()


class TextClassificationDataset(Dataset):
    def __init__(self, encodings_text, labels):
        self.encodings_text = encodings_text
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids_text = self.encodings_text["input_ids"][idx]
        attention_mask_text = self.encodings_text["attention_mask"][idx]
        label = self.labels[idx]

        return {
            "input_ids_text": input_ids_text,
            "attention_mask_text": attention_mask_text,
            "labels": label
        }


# Read CSV
data = pd.read_csv("newest_fewshot_dataset.csv")

# Clean labels
data['Type'] = data['Type'].str.strip()

# Define labels
label_map = {'Fraud': 0, 'Phishing': 1, 'False Positives': 2, 'Commercial Spam': 3}

# Check the unique labels to get rif of the error
unique_labels = data['Type'].unique()
for label in unique_labels:
    if label not in label_map:
        print(f"Label '{label}' not found in label_map.")

# Keep clean labels
all_labels = [label_map[label] for label in data['Type']]

# Separate text
all_text = data['Text'].tolist()

# Split the data
train_text, test_text, train_labels, test_labels = train_test_split(
    all_text, all_labels, train_size=num_shot, test_size=0.2, random_state=random_state)
print(f"Test Text: {test_text}")

# BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize text
train_encodings_text = tokenizer(train_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
test_encodings_text = tokenizer(test_text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")

# BERT model
bert_model = BertModel.from_pretrained("bert-base-uncased")

class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert_model = bert_model
        self.pre_classifier = torch.nn.Linear(768, 256)
        self.relu_func = nn.ReLU()
        self.bn = nn.BatchNorm1d(256)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(256, num_classes)

        # Freeze BERT parameters
        for param in bert_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids_text, attention_mask_text):
        output_text = self.bert_model(input_ids=input_ids_text, attention_mask=attention_mask_text)

        max_output = output_text['last_hidden_state'].mean(dim=1)

        lin_lay = self.pre_classifier(max_output)
        lin_lay = F.relu(lin_lay)

        # Apply batch normalization only if the batch size is greater than 1
        if max_output.size(0) > 1:
            lin_lay = self.bn(lin_lay)

        dropout_output = self.dropout(lin_lay)
        output = self.classifier(dropout_output)

        return output


# Create datasets for text
train_dataset = TextClassificationDataset(train_encodings_text, train_labels)
test_dataset = TextClassificationDataset(test_encodings_text, test_labels)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

# classifiers
classifier = BERTClassifier(bert_model, num_classes=num_classes)
classifier.to(device)

# Parameters gradient true
model_params = [param for param in classifier.parameters() if param.requires_grad]

for name, param in classifier.named_parameters():
    print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")

optimizer = AdamW(model_params, lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def train(model, data_loader, optimizer, device):
    n_correct = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_loss = 0
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()

    for batch in data_loader:
        input_ids_text = batch['input_ids_text'].to(device)
        attention_mask_text = batch['attention_mask_text'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
        loss = loss_function(outputs, labels)
        tr_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        n_correct += torch.sum(preds == labels).item()
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    return epoch_loss, epoch_accu

# Training loop
test_accuracies = []
test_f1s = []
for epoch in range(EPOCHS):
    loss, acc = train(classifier, train_loader, optimizer, device)
    scheduler.step()

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Average Loss: {loss:.4f}")
    print(f"Training Accuracy: {acc:.4f}")

    # Testing
    classifier.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_text = batch['input_ids_text'].to(device)
            attention_mask_text = batch['attention_mask_text'].to(device)

            outputs = classifier(input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(batch['labels'])

    f1_test = f1_score(actual_labels, predictions, average='weighted')

    accuracy = accuracy_score(actual_labels, predictions)
    print(f"Test accuracy: {accuracy:.4f}")

    print(f"Test F1 Score: {f1_test:.4f}")
    test_accuracies.append((accuracy, epoch))
    test_f1s.append((f1_test, epoch))
    # and then here ends the shift tab

# End time
end_time = time.time()
total_time = end_time - start_time

sorted_accuracies = sorted(test_accuracies, key=lambda x: x[0], reverse=True)
sorted_f1s = sorted(test_f1s, key=lambda x: x[0], reverse=True)

# Print the sorted list
print("\nSorted Test Accuracies:")
for acc, epoch in sorted_accuracies:
    print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")

print("\nSorted Test f1s: ")
for f1, epoch in sorted_f1s:
    print(f"Epoch{epoch}: Test f1 = {f1:.4f}")

print(f"Total execution time: {total_time:.2f} seconds")
