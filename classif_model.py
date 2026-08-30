import matplotlib.pyplot as plt
import pandas as pd
import random as rd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import fraud_inject as fi
import gemini_integration as gi


# 1. DATA LOADING & SPLITTING
df = fi.fraud()

y_bin = df["Fraud"]
df.drop('Fraud', inplace=True, axis=1)
x_features = df

x_train, x_test, y_train, y_test = train_test_split(
    x_features, y_bin, test_size=0.20, random_state=42, stratify=y_bin
)


# 2. DATASET & DATALOADER
class FraudDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = torch.tensor(x_data.values, dtype=torch.float32)
        self.y = torch.tensor(y_data.values, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = FraudDataset(x_train, y_train)
test_dataset = FraudDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 3. NEURAL NETWORK ARCHITECTURE
class FraudDetector(nn.Module):
    def __init__(self, input_size):
        super(FraudDetector, self).__init__()
        
        self.layer1 = nn.Linear(input_size, 64) 
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        
        self.output_layer = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        
        x = self.output_layer(x)
        return x

n_features = x_train.shape[1] 
model = FraudDetector(input_size=n_features)


# 4. LOSS FUNCTION (CLASS IMBALANCE) & OPTIMIZER
total_legit_transactions = (y_train == 0).sum()
total_fraud_transactions = (y_train == 1).sum()
fraud_weight = torch.tensor([total_legit_transactions / total_fraud_transactions], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=fraud_weight) 
optimizer = optim.Adam(model.parameters(), lr=1e-3) 


# 5. TRAINING LOOP
epochs = 100 

# --- OPTIONAL: List to store loss history for plotting ---
train_loss_history = []

print("Starting Training...")
for epoch in range(epochs):
    model.train() 
    epoch_loss = 0
    
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(train_loader)
    
    # --- OPTIONAL: Append loss to history ---
    train_loss_history.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')


# 6. OPTIONAL BLOCK: PLOT LOSS OVER EPOCHS
def graph_plot():
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_loss_history, marker='o', linestyle='-', color='b', markersize=3)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('BCEWithLogitsLoss')
    plt.grid(True)
    plt.show()

# You can comment out this command if you don't want the graph
graph_plot()

# 7. MODEL EVALUATION & TESTING
model.eval() 
correct_predictions = 0
total_samples = 0

all_y_true = []
all_y_pred = []

threshold = 0.35 

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        y_test_pred_logits = model(batch_x)
        
        y_test_pred_prob = torch.sigmoid(y_test_pred_logits)
        y_test_pred_class = (y_test_pred_prob >= threshold).float()
        
        correct_predictions += y_test_pred_class.eq(batch_y).sum().item()
        total_samples += batch_y.size(0)

        all_y_true.extend(batch_y.numpy().flatten())
        all_y_pred.extend(y_test_pred_class.numpy().flatten())

accuracy = correct_predictions / total_samples
print(f'\nTest Set Accuracy (Threshold={threshold}): {accuracy * 100:.2f}%')


# 8. GEMINI DATA TREATMENT
x_test_analyzed = x_test.copy()
x_test_analyzed['predictions'] = all_y_pred

gemini_dataframe = x_test_analyzed[x_test_analyzed["predictions"] == 1.0]
gemini_dict = gemini_dataframe.to_dict(orient="records")


# 9. OPTIONAL BLOCK: EXTRA EVALUATION METRICS
def extra_metrics():
    precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)

    print("-" * 30)
    print("EXTRA METRICS (Fraud Detection)")
    print("-" * 30)
    print(f"Precision: {precision:.4f} (When the model predicts Fraud, how often is it right?)")
    print(f"Recall:    {recall:.4f} (Out of all real Frauds, how many did the model find?)")
    print(f"F1-Score:  {f1:.4f} (Harmonic mean of Precision and Recall)")
    print("\nConfusion Matrix:")
    print(conf_matrix)

# You can comment out this command if you only want Accuracy
extra_metrics()


# 10. INTEGRATING GEMINI API FOR FRAUD EXPLANATION
print(f"\nTotal of detected frauds: {len(gemini_dict)}")
print("Generating AI explanations for three random cases...\n")

random_cases = rd.sample(gemini_dict, 3)

for item in random_cases:
    explanation = gi.generate_fraud_explanation(item)
    print("\n--- GEMINI FRAUD ANALYSIS ---")
    print(explanation)
    print("=" * 40)