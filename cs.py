import torch

model = torch.load("model.pth")
with torch.no_grad():
    output = model(torch.tensor([[100, 80]], dtype=torch.float32))
    print(output.item())