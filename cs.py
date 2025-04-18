import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# 定義特徵詞彙表 (客服相關特徵)
FEATURE_VOCAB = {
    "userPrompt": 0,
    "Response": 1,
    "<UNK>": 2,
    "<PAD>": 3
}

# 定義值詞彙表 (需要根據實際 datasets 建立，這裡只是範例)
# 這個詞彙表會在 build_value_vocab 函數中動態建立
VALUE_VOCAB = {}  # 初始化為空，稍後會建立

class ChatDataset(Dataset):
    def __init__(self, data, feature_vocab, value_vocab):
        # data: 包含多個 user_input 字串的列表 (用於訓練)
        # 每個 user_input 代表一個對話，格式如 "userPrompt:Hi;Response:您好，有什麼可以幫忙？"
        self.data = data
        # feature_vocab: 特徵詞彙表 (FEATURE_VOCAB)
        self.feature_vocab = feature_vocab
        # value_vocab: 值詞彙表 (VALUE_VOCAB)
        self.value_vocab = value_vocab

    def __len__(self):
        # 傳回資料集的大小 (對話的數量)
        return len(self.data)

    def __getitem__(self, idx):
        # 取得指定索引的對話
        user_input = self.data[idx]
        feature_value_pairs = user_input.split(';')
        feature_ids = []
        value_ids = []
        target_value_ids = [] # 目標是生成 Response，所以是 value_ids

        user_prompt = None
        response = None

        for i, pair in enumerate(feature_value_pairs):
            if not pair:
                continue

            try:
                feature, value = pair.split(':')
                feature = feature.strip()
                value = value.strip()

                feature_id = self.feature_vocab.get(feature)
                if feature_id is None:
                    feature_id = self.feature_vocab["<UNK>"]
                feature_ids.append(feature_id)

                value = str(value)
                value_id = self.value_vocab.get(value)
                if value_id is None:
                    value_id = self.value_vocab["<UNK>"]
                value_ids.append(value_id)

                if feature == "userPrompt":
                    user_prompt = value
                elif feature == "Response":
                    response = value

            except ValueError as e:
                print(f"Error processing pair: {pair}. Skipping. Error: {e}")
                continue

        # 自定義判斷邏輯 1: 將 Response 的 value_ids 作為目標
        # 目標是根據 userPrompt 生成 Response
        if response:
            target_value_ids = [self.value_vocab.get(token, self.value_vocab["<UNK>"]) for token in response.split()] # 將 Response 分詞並轉換為 value_ids
        else:
            target_value_ids = [self.value_vocab["<PAD>"]]  # 如果沒有 Response，則填充

        feature_ids = torch.tensor(feature_ids)
        value_ids = torch.tensor(value_ids)
        target_value_ids = torch.tensor(target_value_ids)

        return feature_ids, value_ids, target_value_ids


class ChatModel(nn.Module):
    def __init__(self, feature_vocab_size, value_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.feature_embedding = nn.Embedding(feature_vocab_size, embedding_dim, padding_idx=0)
        self.value_embedding = nn.Embedding(value_vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(2 * embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, value_vocab_size)  # 輸出是 value_vocab 的大小，用於生成回應

    def forward(self, feature_ids, value_ids):
        feature_embedded = self.feature_embedding(feature_ids)
        value_embedded = self.value_embedding(value_ids)
        combined_embedded = torch.cat((feature_embedded, value_embedded), dim=2)
        output, _ = self.lstm(combined_embedded)
        # 自定義判斷邏輯 4: 使用 LSTM 的所有時間步的輸出，用於生成序列
        output = self.linear(output)  # 輸出所有時間步
        return output


# 建立 value_vocab (使用 datasets)
def build_value_vocab(user_inputs, min_freq=1):
    value_counts = Counter()
    for user_input in user_inputs:  # 迭代 user_inputs 列表
        feature_value_pairs = user_input.split(';')
        for pair in feature_value_pairs:
            if not pair:
                continue
            try:
                _, value = pair.split(':')
                value = value.strip()
                # 將 value 分詞，以便建立詞彙表
                tokens = value.split()
                value_counts.update(tokens)
            except ValueError:
                continue

    value_vocab = {value: i + 2 for i, (value, count) in enumerate(value_counts.items()  if min_freq <= 1 else filter(lambda item: item[1] >= min_freq, value_counts.items()))}
    value_vocab["<UNK>"] = 0
    value_vocab["<PAD>"] = 1
    return value_vocab


# 範例用法
datasets = [  # 使用 "datasets" 這個變數名稱
    "userPrompt: Hi; Response: 您好，有什麼可以幫忙？;",
    "userPrompt: How to reset password?; Response: 請到設定頁面重設密碼。;",
    "userPrompt: What is your name?; Response: 我是deepFun 客服模型。;",
    "userPrompt: 你是誰; Response: 我是deepFun 客服模型。;",
    "userPrompt: Your name; Response: 我是deepFun 客服模型。;" 
]

userInput = "userPrompt: What is your name?;"  # 單一的 user_input 字串

# 建立 value_vocab (使用 datasets)
VALUE_VOCAB = build_value_vocab(datasets)  # 使用 datasets 來建立 value_vocab，並更新全域變數

# 建立 Dataset (使用 datasets)
train_dataset = ChatDataset(datasets, FEATURE_VOCAB, VALUE_VOCAB)  # 使用 datasets 來建立訓練資料集

# 建立 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)  # 可以調整 batch_size, 加入 drop_last=True

# 建立模型
model = ChatModel(feature_vocab_size=len(FEATURE_VOCAB), value_vocab_size=len(VALUE_VOCAB), embedding_dim=64, hidden_dim=128)

# 訓練模型
# 訓練迴圈 (簡化版)
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # 交叉熵損失函數，適用於分類問題

num_epochs = 100  # 訓練週期數
loss = 0.0 # 初始化 loss
for epoch in range(num_epochs):
    for batch in train_dataloader:
        feature_ids, value_ids, target_value_ids = batch

        # 梯度歸零
        optimizer.zero_grad()

        # 前向傳播
        output = model(feature_ids, value_ids)

        # 將輸出重塑為 (batch_size * sequence_length, value_vocab_size)
        output = output.view(-1, len(VALUE_VOCAB))
        target_value_ids = target_value_ids.view(-1)

        # 計算損失
        loss = criterion(output, target_value_ids)

        # 反向傳播和優化
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss type: {type(loss)}") # 檢查 loss 的類型
    if isinstance(loss, torch.Tensor):
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")


# 使用訓練好的模型來處理 userInput
def prepare_input(user_input, feature_vocab, value_vocab, max_len=10):  # 添加 max_len 参数
    feature_value_pairs = user_input.split(';')
    feature_ids = []
    value_ids = []

    for pair in feature_value_pairs:
        if not pair:
            continue

        try:
            feature, value = pair.split(':')
            feature = feature.strip()
            value = value.strip()

            feature_id = feature_vocab.get(feature)
            if feature_id is None:
                feature_id = feature_vocab["<UNK>"]
            feature_ids.append(feature_id)

            value = str(value)
            #  將輸入分詞
            tokens = value.split()
            for token in tokens:
                value_id = value_vocab.get(token)
                if value_id is None:
                    value_id = value_vocab["<UNK>"]
                value_ids.append(value_id)

        except ValueError as e:
            print(f"Error processing pair: {pair}. Skipping. Error: {e}")
            continue

    # 填充或截斷 feature_ids
    if len(feature_ids) < max_len:
        feature_ids += [FEATURE_VOCAB["<PAD>"]] * (max_len - len(feature_ids))
    else:
        feature_ids = feature_ids[:max_len]

    # 填充或截斷 value_ids
    if len(value_ids) < max_len:
        value_ids += [VALUE_VOCAB["<PAD>"]] * (max_len - len(value_ids))
    else:
        value_ids = value_ids[:max_len]

    feature_ids = torch.tensor(feature_ids)
    value_ids = torch.tensor(value_ids)

    return feature_ids, value_ids


# 1. 將 userInput 轉換成模型可以理解的格式 (feature_ids, value_ids)
user_feature_ids, user_value_ids = prepare_input(userInput, FEATURE_VOCAB, VALUE_VOCAB, max_len=10) # 指定 max_len

# 2. 將輸入傳遞給模型
model.eval()  # 設定為評估模式
"""
with torch.no_grad():  # 停用梯度計算，以加快速度
    #output = model(user_feature_ids, user_value_ids)
    output = model(user_feature_ids.unsqueeze(0), user_value_ids.unsqueeze(0))
"""
output = model(user_feature_ids.unsqueeze(0), user_value_ids.unsqueeze(0))
# 3. 處理模型的輸出 (生成回應)
# 將輸出轉換為機率分佈
probabilities = torch.softmax(output, dim=2)

# 選擇機率最高的詞彙 ID
predicted_ids = torch.argmax(probabilities, dim=2)

# 將詞彙 ID 轉換為詞彙
predicted_tokens = [ [list(VALUE_VOCAB.keys())[list(VALUE_VOCAB.values()).index(idx.item())] for idx in seq] for seq in predicted_ids]

# 將詞彙連接成句子
predicted_response = " ".join(predicted_tokens[0])

print(f"預測回應: {predicted_response}")