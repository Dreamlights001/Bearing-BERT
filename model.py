# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np


# 阶段二：轻量级残差适配器
class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim)
        )

    def forward(self, x):
        return x + self.adapter(x)


# 振动编码器 (使用1D CNN)
class VibrationEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=31), nn.ReLU(), nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15), nn.ReLU(), nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# 文本编码器 (使用预训练的DistilBERT)
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # 冻结BERT参数，我们只用它来获取好的文本表示
        for param in self.bert.parameters():
            param.requires_grad = False
        # 增加一个投影层将BERT的输出[768]映射到我们的嵌入维度
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)

    def forward(self, text_list):
        # 为一批文本进行编码
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(self.bert.device) for key, val in inputs.items()}

        bert_output = self.bert(**inputs).last_hidden_state[:, 0, :]  # 取[CLS] token的输出
        text_embedding = self.projection(bert_output)
        return text_embedding


# 主CLIP模型
class BearingCLIP(nn.Module):
    def __init__(self, embedding_dim=512, use_adapter=False):
        super().__init__()
        self.vibration_encoder = VibrationEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim)

        self.adapter = None
        if use_adapter:
            self.adapter = ResidualAdapter(embedding_dim)

        # CLIP的logit_scale参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, vibration_data, text_list):
        # 获取振动和文本特征
        vibration_features = self.vibration_encoder(vibration_data)

        # 如果使用适配器，则应用它
        if self.adapter:
            vibration_features = self.adapter(vibration_features)

        text_features = self.text_encoder(text_list)

        # 归一化特征
        vibration_features = F.normalize(vibration_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        return vibration_features, text_features, self.logit_scale.exp()

# --- SimpleTextEncoder 定义 ---
class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.tokenizer = None
    def forward(self, text_list):
        if not self.tokenizer: raise ValueError("SimpleTextEncoder's tokenizer must be set before use.")
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.embedding.weight.device)
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        return hidden.squeeze(0)

# --- TextEncoder 定义 ---
class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.bert.parameters(): param.requires_grad = False
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim)
    def forward(self, text_list):
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: val.to(self.bert.device) for key, val in inputs.items()}
        bert_output = self.bert(**inputs).last_hidden_state[:, 0, :]
        return self.projection(bert_output)

# --- 其他类定义 (ResidualAdapter, VibrationEncoder) ---
class ResidualAdapter(nn.Module):
    # ... (代码不变)
    def __init__(self, input_dim, bottleneck_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.ReLU(), nn.Linear(bottleneck_dim, input_dim))
    def forward(self, x): return x + self.adapter(x)

class VibrationEncoder(nn.Module):
    # ... (代码不变)
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.conv_net = nn.Sequential(nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=31), nn.ReLU(), nn.BatchNorm1d(16), nn.MaxPool1d(kernel_size=2, stride=2), nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(kernel_size=2, stride=2), nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7), nn.ReLU(), nn.BatchNorm1d(64), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(64, embedding_dim)
    def forward(self, x):
        x = self.conv_net(x); x = x.view(x.size(0), -1); return self.fc(x)

# --- BearingCLIP 定义 ---
class BearingCLIP(nn.Module):
    def __init__(self, embedding_dim=512, use_adapter=False, text_encoder_type='distilbert', vocab_size=None, simple_tokenizer=None):
        super().__init__()
        self.vibration_encoder = VibrationEncoder(embedding_dim)
        self.text_encoder_type = text_encoder_type
        if text_encoder_type == 'distilbert':
            self.text_encoder = TextEncoder(embedding_dim)
        elif text_encoder_type == 'simple_lstm':
            if vocab_size is None:
                tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                vocab_size = tokenizer.vocab_size
                simple_tokenizer = tokenizer
            self.text_encoder = SimpleTextEncoder(vocab_size, embedding_dim)
            if simple_tokenizer is not None: self.text_encoder.tokenizer = simple_tokenizer
        else: raise ValueError(f"Unknown text_encoder_type: {text_encoder_type}")
        self.adapter = None
        if use_adapter: self.adapter = ResidualAdapter(embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, vibration_data, text_list):
        vibration_features = self.vibration_encoder(vibration_data)
        if self.adapter: vibration_features = self.adapter(vibration_features)
        text_features = self.text_encoder(text_list)
        vibration_features = F.normalize(vibration_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return vibration_features, text_features, self.logit_scale.exp()