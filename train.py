# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse

from data_loader import BearingDataset
from model import BearingCLIP


def contrastive_loss(logits):
    # 对称交叉熵损失
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return (loss_i + loss_t) / 2


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 准备数据
    train_dataset = BearingDataset(data_dir=args.train_data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 2. 定义模型
    # 阶段二(adapter)需要加载预训练模型，所以use_adapter=True
    model = BearingCLIP(embedding_dim=args.embedding_dim, use_adapter=(args.mode == 'adapter')).to(device)

    # 3. 设置训练模式和优化器
    if args.mode == 'zero_shot':
        print("--- Starting Stage 1: Zero-Shot Pre-training ---")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        save_path = os.path.join(args.weights_dir, 'zero_shot_model.pth')

    elif args.mode == 'adapter':
        print("--- Starting Stage 2: Adapter Fine-tuning ---")
        # 加载第一阶段的模型权重
        if not os.path.exists(args.load_weights):
            raise FileNotFoundError(
                f"Pre-trained weights not found at {args.load_weights}. Please run zero_shot training first.")

        # 只加载编码器的权重，不加载适配器
        pretrained_dict = torch.load(args.load_weights, map_location=device)
        model_dict = model.state_dict()
        # 过滤掉适配器的参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'adapter' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # 冻结除适配器外的所有参数
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:
                print(f"Training adapter parameter: {name}")

        # 只为适配器的参数创建优化器
        adapter_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(adapter_params, lr=args.lr)
        save_path = os.path.join(args.weights_dir, 'adapter_tuned_model.pth')

    # 4. 训练循环
    all_prompts = train_dataset.prompts

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for vibration_batch, label_indices in pbar:
            vibration_batch = vibration_batch.to(device)
            # 根据标签索引获取对应的文本提示
            text_batch = [all_prompts[i] for i in label_indices]

            optimizer.zero_grad()

            # 前向传播
            # 注意：这里我们让模型自己处理文本编码
            # 为了效率，可以预先编码所有文本，但动态编码更灵活
            vibration_features, text_features, logit_scale = model(vibration_batch, text_batch)

            # 计算logits
            logits = logit_scale * vibration_features @ text_features.T

            # 计算损失
            loss = contrastive_loss(logits)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

    # 5. 保存模型
    os.makedirs(args.weights_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bearing Diagnosis CLIP-like model.")
    parser.add_argument('--mode', type=str, required=True, choices=['zero_shot', 'adapter'],
                        help="Training mode: 'zero_shot' (pre-training) or 'adapter' (fine-tuning).")
    parser.add_argument('--train_data_dir', type=str, default='./bearingset/train_set/', help="Path to training data.")
    parser.add_argument('--weights_dir', type=str, default='./pretrained_weights/',
                        help="Directory to save model weights.")
    parser.add_argument('--load_weights', type=str, default='./pretrained_weights/zero_shot_model.pth',
                        help="Path to pre-trained weights for adapter tuning.")

    parser.add_argument('--embedding_dim', type=int, default=512, help="Dimension of the embedding space.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")

    args = parser.parse_args()
    train(args)