import torch
from torch.nn import Module, Linear, BCELoss

# ml == g
# 蛋糕原料 (麵粉&牛奶：g, 雞蛋：隻)
flour = torch.tensor([ 2.4 ], dtype=torch.float32)
egg = torch.tensor([ 2.0 ], dtype=torch.float32)
milk = torch.tensor([ 30.0 ], dtype=torch.float32)

# 設置自動一階導數
flour.requires_grad = True
egg.requires_grad = True
milk.requires_grad = True

dough = flour * 3 + egg + milk
# 計算麵團, 反向計算（能夠反方向再計算）
dough.backward()

class cakeProduceMachine(Module):
    def __init__(self):
        super().__init__()
        # 判斷dough&milk
        self.layer1 = Linear(2, 1)
    def forward(self, x):
        return torch.sigmoid(self.layer1(x))

# [Dough,Milk]
datasets = torch.tensor([[100, 80], [150, 67], [90, 30], [120,45]], dtype=torch.float32)
labels = torch.tensor([[1],[1],[0],[0]], dtype=torch.float32)

user_flour = 370.8
user_milk = 70.8
user_label = 0
if user_flour > 200.2 and user_milk >= 56.0: user_label = 0
elif user_flour <= 40.0 and user_milk <= 12.3: user_label = 0
else: user_label = 1
user_input = torch.tensor([[user_flour, user_milk]], dtype=torch.float32)
user_except_result = torch.tensor([[user_label]], dtype=torch.float32)
datasets = torch.cat((datasets, user_input), dim=0)
labels = torch.cat((labels, user_except_result), dim=0)
model = cakeProduceMachine()
calLoss = BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01) #optim 是 優化器（用於準確計算誤差loss)

for trainingTime in range(100):
    optim.zero_grad() #清空微分
    outputs = model(datasets)
    loss = calLoss(outputs, labels)
    loss.backward()
    optim.step() #下個訓練周期或次數
    #每10次訓練判斷
    if trainingTime % 10 == 0:
        print(f"Epoch: {trainingTime}, Loss: {loss.item():.4f}")
result = ""
if user_except_result.item() > 0.5: result = "成功做到蛋糕"
elif user_except_result.item() == 0.5: result = "蛋糕做出來一般般"
else: result = "蛋糕不好吃"
print(f"結果是： {result}, 概率是: {user_except_result.item()*100 }%")