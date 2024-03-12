# The BERT model with the new dataset: smaller dataset with rationale
# Benchmark and actual model
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
EPOCHS = 30
LEARNING_RATE = 1e-03
num_classes = 2
random_state = 42
random_prompt_length = 10
num_shot = 1

# random seed
seed = 999999999
torch.manual_seed(seed)
np.random.seed(seed)

# Move the classifiers to device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Start measuring time
start_time = time.time()


class TextClassificationDataset(Dataset):
    def __init__(self, encodings_text, encodings_rationale, labels):
        self.encodings_text = encodings_text
        self.encodings_rationale = encodings_rationale
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Make sure the index is within the bounds of the dataset
        idx = idx % len(self.labels)

        input_ids_text = self.encodings_text["input_ids"][idx]
        attention_mask_text = self.encodings_text["attention_mask"][idx]
        input_ids_rationale = self.encodings_rationale["input_ids"][idx]
        attention_mask_rationale = self.encodings_rationale["attention_mask"][idx]
        label = self.labels[idx]

        return {
            "input_ids_text": input_ids_text,
            "attention_mask_text": attention_mask_text,
            "input_ids_rationale": input_ids_rationale,
            "attention_mask_rationale": attention_mask_rationale,
            "labels": label
        }


# Load dataset with rationales
smaller_dataset = pd.read_csv("smaller_dataset_with_rationale.csv", encoding='latin1')

# X number of samples from each class
few_shot_samples = pd.concat([smaller_dataset[smaller_dataset['Class'] == 0].sample(n=num_shot, random_state=random_state),
                           smaller_dataset[smaller_dataset['Class'] == 1].sample(n=num_shot, random_state=random_state)])

# ensure that the num_shot is right

# BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize text and rationale
encodings_text = tokenizer(few_shot_samples['Text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN,
                           return_tensors="pt")
encodings_rationale = tokenizer(few_shot_samples['Rationale'].tolist(), truncation=True, padding=True,
                                max_length=MAX_LEN, return_tensors="pt")

# Create train dataset
labels = few_shot_samples['Class'].tolist()
train_dataset = TextClassificationDataset(encodings_rationale, encodings_text, labels)

# Data loader
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

# Test Dataset
# Read the fraud_email dataset
test_data = pd.read_csv("fraud_email_.csv", encoding='latin1')
print(f"Test Data size: ", test_data.shape)

# Keep only the desired columns
test_data = test_data[['Text', 'Class']]
# Remove rows with None values in the 'Text' column
test_data = test_data.dropna(subset=['Text'])
test_data = test_data.dropna(subset=['Class'])

# Sample 10% of the testing dataset
percentage = 0.10
test_data = test_data.sample(frac=percentage, random_state=random_state)
print(f"Sample 10% of test data: ", test_data.shape)

# Tokenize test data
test_texts = test_data['Text'].tolist()

# Tokenize each text individually
test_encodings_text = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAX_LEN,
                                return_tensors="pt")

# Create dataset for testing
test_labels = test_data['Class'].tolist()
test_dataset = TextClassificationDataset(test_encodings_text, test_encodings_text, test_labels)

# Data loader for testing
test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

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

    def forward(self, input_ids_text, attention_mask_text, input_ids_rationale, attention_mask_rationale):
        output_text = self.bert_model(input_ids=input_ids_text, attention_mask=attention_mask_text)
        output_rationale = self.bert_model(input_ids=input_ids_rationale, attention_mask=attention_mask_rationale)
        output_text = output_text['last_hidden_state'][:, :-random_prompt_length, :]
        output_rationale = output_rationale['last_hidden_state']

        random_prompt = self.random_prompt.unsqueeze(0).expand(output_rationale.size(0), -1, -1)

        # add the prompt to the end of the text
        concat_prompt_text = torch.cat((output_text, random_prompt), dim=1)

        concatenated_output = torch.cat((output_rationale, concat_prompt_text, random_prompt), dim=1)

        mean_output = concatenated_output.mean(dim=1)

        lin_lay = self.pre_classifier(mean_output)
        lin_lay = F.relu(lin_lay)

        # Apply batch normalization only if the batch size is greater than 1
        if concatenated_output.size(0) > 1:
            lin_lay = self.bn(lin_lay)

        dropout_output = self.dropout(lin_lay)
        output = self.classifier(dropout_output)

        return output


# Create datasets for rationale and text
# this is repeated???
# train_dataset = TextClassificationDataset(encodings_rationale, encodings_text, labels)

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
f1_scores = []
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

    actual_labels = [int(label) for label in actual_labels]

    # accuracy and f1
    f1_test = f1_score(actual_labels, predictions, average='weighted')

    accuracy = accuracy_score(actual_labels, predictions)
    print(f"Test accuracy: {accuracy:.4f}")

    print(f"Test F1 Score: {f1_test:.4f}")
    f1_scores.append((f1_test, epoch))
    test_accuracies.append((accuracy, epoch))
    # and then here ends the shift tab

# End time
end_time = time.time()
total_time = end_time - start_time

sorted_f1s = sorted(f1_scores, key=lambda x: x[0], reverse=True)
sorted_accuracies = sorted(test_accuracies, key=lambda x: x[0], reverse=True)

# Print the sorted list
print("\nSorted Test Accuracies:")
for acc, epoch in sorted_accuracies:
    print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")

print("\nSorted F1s:")
for f1, epoch in sorted_f1s:
    print(f"Epoch {epoch}: F1 Score = {f1:.4f}")
print(f"Total execution time: {total_time:.2f} seconds")

# #######################################################
# # Benchmark
# import torch
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizerFast, BertModel, AdamW
# from torch import nn
# import time
# import torch.nn.functional as F
# import numpy as np
#
# # Hyperparameters
# MAX_LEN = 512
# TRAIN_BATCH_SIZE = 8
# VALID_BATCH_SIZE = 4
# EPOCHS = 30
# LEARNING_RATE = 1e-03
# num_classes = 2
# random_state = 42
# num_shot = 5
#
# # Random seed
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
# class TextClassificationDataset(Dataset):
#     def __init__(self, encodings_text, labels):
#         self.encodings_text = encodings_text
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, idx):
#         # Make sure the index is within the bounds of the dataset
#         idx = idx % len(self.labels)
#
#         input_ids_text = self.encodings_text["input_ids"][idx]
#         attention_mask_text = self.encodings_text["attention_mask"][idx]
#         label = self.labels[idx]
#
#         return {
#             "input_ids_text": input_ids_text,
#             "attention_mask_text": attention_mask_text,
#             "labels": label
#         }
#
# # Load dataset without rationales
# smaller_dataset = pd.read_csv("smaller_dataset_with_rationale.csv", encoding='latin1')
#
# # X number of samples from each class
# few_shot_samples = pd.concat([smaller_dataset[smaller_dataset['Class'] == 0].sample(n=num_shot, random_state=random_state),
#                            smaller_dataset[smaller_dataset['Class'] == 1].sample(n=num_shot, random_state=random_state)])
#
#
# # BERT tokenizer
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#
# # Tokenize only text
# encodings_text = tokenizer(few_shot_samples['Text'].tolist(), truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
#
# # Create dataset
# labels = few_shot_samples['Class'].tolist()
# train_dataset = TextClassificationDataset(encodings_text, labels)
#
# # Data loader
# train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
#
# # Test Dataset
# # Read the fraud_email dataset
# test_data = pd.read_csv("fraud_email_.csv", encoding='latin1')
# print(f"Test Data size: ", test_data.shape)
#
# # Keep only the desired columns
# test_data = test_data[['Text', 'Class']]
# # Remove rows with None values in the 'Text' column
# test_data = test_data.dropna(subset=['Text'])
# test_data = test_data.dropna(subset=['Class'])
#
# # Sample 10% of the testing dataset
# percentage = 0.10
# test_data = test_data.sample(frac=percentage, random_state=random_state)
# print(f"Sample 10% of test data: ", test_data.shape)
#
# # Tokenize test data
# test_texts = test_data['Text'].tolist()
#
# # Tokenize each text individually
# test_encodings_text = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors="pt")
#
# # Create dataset for testing
# test_labels = test_data['Class'].tolist()
# test_dataset = TextClassificationDataset(test_encodings_text, test_labels)
#
# # Data loader for testing
# test_loader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)
#
# # BERT model
# bert_model = BertModel.from_pretrained("bert-base-uncased")
#
# class BERTClassifier(nn.Module):
#     def __init__(self, bert_model, num_classes):
#         super(BERTClassifier, self).__init__()
#         self.bert_model = bert_model
#         self.pre_classifier = torch.nn.Linear(768, 256)
#         self.relu_func = nn.ReLU()
#         self.bn = nn.BatchNorm1d(256)
#         self.dropout = torch.nn.Dropout(0.3)
#         self.classifier = torch.nn.Linear(256, num_classes)
#
#         # Freeze BERT parameters
#         for param in bert_model.parameters():
#             param.requires_grad = False
#
#     def forward(self, input_ids_text, attention_mask_text):
#         output_text = self.bert_model(input_ids=input_ids_text, attention_mask=attention_mask_text)
#         output_text = output_text['last_hidden_state']
#
#         mean_output = output_text.mean(dim=1)
#
#         lin_lay = self.pre_classifier(mean_output)
#         lin_lay = F.relu(lin_lay)
#
#         # Apply batch normalization only if the batch size is greater than 1
#         if mean_output.size(0) > 1:
#             lin_lay = self.bn(lin_lay)
#
#         dropout_output = self.dropout(lin_lay)
#         output = self.classifier(dropout_output)
#
#         return output
#
# # Create datasets for text only
# train_dataset = TextClassificationDataset(encodings_text, labels)
#
# # classifiers
# classifier = BERTClassifier(bert_model, num_classes=num_classes)
# classifier.to(device)
#
# # Parameters gradient true
# model_params = [param for param in classifier.parameters() if param.requires_grad]
#
# for name, param in classifier.named_parameters():
#     print(f"Parameter: {name}, Requires Gradient: {param.requires_grad}")
#
# optimizer = AdamW(model_params, lr=LEARNING_RATE, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
#
# def train(model, data_loader, optimizer, device):
#     n_correct = 0
#     nb_tr_examples, nb_tr_steps = 0, 0
#     tr_loss = 0
#     loss_function = torch.nn.CrossEntropyLoss()
#     model.train()
#
#     for batch in data_loader:
#         input_ids_text = batch['input_ids_text'].to(device)
#         attention_mask_text = batch['attention_mask_text'].to(device)
#         labels = batch['labels'].to(device)
#
#         optimizer.zero_grad()
#         outputs = model(input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
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
#
#     epoch_loss = tr_loss / nb_tr_steps
#     epoch_accu = (n_correct * 100) / nb_tr_examples
#
#     return epoch_loss, epoch_accu
#
# ##
# test_accuracies = []
# f1_scores = []
# for epoch in range(EPOCHS):
#     loss, acc = train(classifier, train_loader, optimizer, device)
#     scheduler.step()
#
#     print(f"Epoch {epoch + 1}/{EPOCHS}")
#     print(f"Average Loss: {loss:.4f}")
#     print(f"Training Accuracy: {acc:.4f}")
#
#     # from here shift tab to bring it back, in epoch loop for the best epoch
#     # Testing
#     classifier.eval()
#     predictions = []
#     actual_labels = []
#
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids_text = batch['input_ids_text'].to(device)
#             attention_mask_text = batch['attention_mask_text'].to(device)
#
#             outputs = classifier(input_ids_text=input_ids_text, attention_mask_text=attention_mask_text)
#
#             _, preds = torch.max(outputs, dim=1)
#             predictions.extend(preds.cpu().tolist())
#             actual_labels.extend(batch['labels'])
#
#     actual_labels = [int(label) for label in actual_labels]
#
#     # accuracy and f1
#     f1_test = f1_score(actual_labels, predictions, average='weighted')
#
#     accuracy = accuracy_score(actual_labels, predictions)
#     print(f"Test accuracy: {accuracy:.4f}")
#
#     print(f"Test F1 Score: {f1_test:.4f}")
#     f1_scores.append((f1_test, epoch))
#     test_accuracies.append((accuracy, epoch))
#     # and then here ends the shift tab
#
# # End time
# end_time = time.time()
# total_time = end_time - start_time
#
# sorted_f1s = sorted(f1_scores, key=lambda x: x[0], reverse=True)
# sorted_accuracies = sorted(test_accuracies, key=lambda x: x[0], reverse=True)
#
# # Print the sorted list
# print("\nSorted Test Accuracies:")
# for acc, epoch in sorted_accuracies:
#     print(f"Epoch {epoch}: Test Accuracy = {acc:.4f}")
#
# print("\nSorted F1s:")
# for f1, epoch in sorted_f1s:
#     print(f"Epoch {epoch}: F1 Score = {f1:.4f}")
# print(f"Total execution time: {total_time:.2f} seconds")
#
