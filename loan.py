import torch
from torch.nn import Module, Linear, BCELoss
from datasets import datas,results

class mainBrain(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(2, 1)
    def forward(self, x): return torch.sigmoid(self.layer1(x))

# [Income, LoanTarget]
datasets = torch.tensor(datas, dtype=torch.float32)
labels = torch.tensor(results, dtype=torch.float32)

# user_input
user_income = 34500
user_loan_amount = 56000

model = mainBrain()
calLoss = BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for trainTime in range(100):
    optim.zero_grad()
    outputs = model(datasets)
    loss = calLoss(outputs, labels)
    optim.step()
    if trainTime % 10 == 0: print(f"Epoch: {trainTime}, Loss: {loss.item():.4f}")

# Predict
with torch.no_grad():
    result = ""
    prediction = model(torch.tensor([[user_income, user_loan_amount]], dtype=torch.float32))
    if prediction.item() >= 0.5: result = "貸款通過"
    else: result = "貸款不通過"
print(f"結果是 {result} , 概率是：{prediction.item()*100}%")