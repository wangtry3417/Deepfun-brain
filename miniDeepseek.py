"""
Mini DeepSeek - 用 PyTorch 從零打造一個小型的 DeepSeek
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np

# ==================== 1. 迷你版 DeepSeek 架構 ====================

class MiniDeepSeek(nn.Module):
    """
    迷你版 DeepSeek 模型
    包含：
    - 多頭注意力 (Multi-Head Attention)
    - 前饋網路 (Feed Forward)
    - 層歸一化 (Layer Norm)
    - 殘差連接 (Residual Connection)
    """
    
    def __init__(self, vocab_size=10000, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_seq_len=1024):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_len, d_model)
        
        # Transformer 層
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ])
        
        # 輸出層
        self.ln_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def create_positional_encoding(self, max_len, d_model):
        """建立位置編碼"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        return pe
    
    def forward(self, x):
        """
        x: (batch_size, seq_len)
        returns: (batch_size, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        
        # 詞嵌入 + 位置編碼
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # 通過 transformer 層
        for layer in self.layers:
            x = layer(x)
        
        # 輸出
        x = self.ln_final(x)
        logits = self.output(x)
        
        return logits

# ==================== 2. Transformer Block ====================

class TransformerBlock(nn.Module):
    """單一 Transformer 層"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 多頭注意力
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        
        # 前饋網路
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 注意力子層 + 殘差連接
        attn_output = self.attention(x)
        x = self.ln1(x + self.dropout(attn_output))
        
        # 前饋子層 + 殘差連接
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        
        return x

# ==================== 3. 多頭注意力 ====================

class MultiHeadAttention(nn.Module):
    """多頭注意力機制"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 線性投影層
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 線性投影並分割成多頭
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # 轉置以便計算注意力
        Q = Q.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 計算注意力分數
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 應用注意力
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 輸出投影
        output = self.W_o(context)
        
        return output

# ==================== 4. 前饋網路 ====================

class FeedForward(nn.Module):
    """前饋網路 (FFN)"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 用 GELU 激活
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ==================== 5. 資料集 ====================

class TextDataset(Dataset):
    """簡單的文字資料集"""
    
    def __init__(self, texts, tokenizer, max_len=128):
        self.data = []
        for text in texts:
            tokens = tokenizer.encode(text)
            # 切割成固定長度
            for i in range(0, len(tokens) - max_len, max_len // 2):
                seq = tokens[i:i+max_len]
                if len(seq) < max_len:
                    seq = seq + [0] * (max_len - len(seq))
                self.data.append(torch.tensor(seq))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==================== 6. 簡單的 tokenizer ====================

class SimpleTokenizer:
    """極簡版 tokenizer"""
    
    def __init__(self):
        self.vocab = {}
        self.inv_vocab = {}
        self.vocab_size = 4  # 0: pad, 1: bos, 2: eos, 3: unk
        
    def add_word(self, word):
        if word not in self.vocab:
            self.vocab[word] = self.vocab_size
            self.inv_vocab[self.vocab_size] = word
            self.vocab_size += 1
    
    def encode(self, text):
        """文字轉 token"""
        words = text.split()
        tokens = [1]  # bos token
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(3)  # unk token
        tokens.append(2)  # eos token
        return tokens
    
    def decode(self, tokens):
        """token 轉文字"""
        words = []
        for t in tokens:
            if t in self.inv_vocab:
                words.append(self.inv_vocab[t])
            elif t == 1:
                words.append('<BOS>')
            elif t == 2:
                words.append('<EOS>')
            else:
                words.append('<UNK>')
        return ' '.join(words)

# ==================== 7. 訓練函數 ====================

def train_model(model, dataloader, epochs=10, lr=1e-4):
    """訓練模型"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 pad token
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"🚀 開始訓練 (使用: {device})")
    print("="*50)
    
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # 輸入 = 除了最後一個 token
            # 目標 = 除了第一個 token
            inputs = batch[:, :-1]
            targets = batch[:, 1:].contiguous()
            
            # 前向傳播
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 計算損失
            loss = criterion(
                outputs.view(-1, model.vocab_size),
                targets.view(-1)
            )
            
            # 反向傳播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"📊 Epoch {epoch+1} 平均損失: {avg_loss:.4f}")
        
        # 每個 epoch 後生成一個範例
        if (epoch + 1) % 5 == 0:
            generate_sample(model, tokenizer, device)
    
    return model

# ==================== 8. 生成文字 ====================

def generate_sample(model, tokenizer, device, prompt="<BOS>", max_len=50):
    """生成文字範例"""
    
    model.eval()
    
    # 將提示轉為 token
    if prompt == "<BOS>":
        input_tokens = [1]
    else:
        input_tokens = tokenizer.encode(prompt)
    
    input_tensor = torch.tensor([input_tokens]).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(input_tensor)
            next_token_logits = outputs[0, -1, :]
            
            # 取最高機率的 token
            next_token = torch.argmax(next_token_logits).item()
            
            if next_token == 2:  # EOS token
                break
            
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]]).to(device)
            ], dim=1)
    
    generated = tokenizer.decode(input_tensor[0].cpu().tolist())
    print(f"\n✨ 生成: {generated}\n")

# ==================== 9. 主程式 ====================

def main():
    """訓練一個迷你 DeepSeek"""
    
    print("="*60)
    print("🧠 迷你 DeepSeek 訓練")
    print("="*60)
    
    # 1. 準備資料
    tokenizer = SimpleTokenizer()
    
    # 加入一些詞彙
    corpus = [
        "hello world",
        "deep learning is fun",
        "transformer is powerful",
        "I love coding",
        "python is great",
        "neural networks are amazing",
        "artificial intelligence",
        "machine learning",
    ]
    
    # 建立詞彙表
    for text in corpus:
        for word in text.split():
            tokenizer.add_word(word)
    
    print(f"📚 詞彙表大小: {tokenizer.vocab_size}")
    
    # 2. 建立資料集
    dataset = TextDataset(corpus, tokenizer, max_len=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. 建立模型 (超迷你版)
    model = MiniDeepSeek(
        vocab_size=tokenizer.vocab_size,
        d_model=128,      # 縮小維度
        n_heads=4,        # 減少頭數
        n_layers=3,       # 減少層數
        d_ff=256,         # 縮小 FFN
        max_seq_len=64
    )
    
    print(f"📊 模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 訓練
    model = train_model(model, dataloader, epochs=20, lr=1e-4)
    
    # 5. 測試生成
    print("\n" + "="*60)
    print("🎯 測試生成")
    print("="*60)
    
    test_prompts = [
        "hello",
        "I love",
        "deep",
        "machine",
    ]
    
    for prompt in test_prompts:
        generate_sample(model, tokenizer, torch.device('cpu'), prompt)

# ==================== 10. 真正的 DeepSeek 特色 ====================

class DeepSeekFeatures:
    """真正的 DeepSeek 有的特色"""
    
    def __init__(self):
        self.features = {
            "1M上下文": "✅ 我們的 mini 版只有 1024",
            "MoE架構": "✅ 我們用普通 FFN",
            "MLA注意力": "✅ 我們用普通多頭注意力",
            "FP8量化": "✅ 我們用 FP32",
            "開源免費": "✅ 這個倒是真的！",
        }
    
    def compare(self):
        print("\n📊 跟真正的 DeepSeek 比較：")
        for feature, status in self.features.items():
            print(f"  {feature}: {status}")

# 執行
if __name__ == "__main__":
    main()
    
    # 比較
    DeepSeekFeatures().compare()
    
    print("\n" + "="*60)
    print("🎉 完成！你剛剛訓練了一個迷你 DeepSeek！")
    print("   雖然很小，但概念是一樣的！")
    print("="*60)