# Bearing-BERT: 基于CLIP思想的轴承故障诊断框架

本项目采用 CLIP的核心思想，将BERT模型创新性地应用于工业领域的轴承故障诊断。我们不再使用图像，而是将一维的振动时序信号与描述其状态的自然语言文本进行对齐，构建一个强大的“振动-文本”多模态诊断模型。

**CLIP** (Contrastive Language-Image Pre-Training) 在这里本身不是一个具体的模型，而是一种对比性的“训练思想”或“多模态学习框架”。它的核心思想是：

  - 双编码器结构 (Dual Encoders): 准备两个独立的编码器，一个用于处理一种模态的数据（例如图像），另一个用于处理另一种模态的数据（例如文本）。
  - 对比学习 (Contrastive Learning): 将大量的“数据-描述”对（例如“狗的图片 - a photo of a dog”）同时输入两个编码器，然后通过一个对比损失函数，在同一个高维空间中，将匹配的对拉近，将不匹配的对推远。
  - 目标: 最终得到一个对齐的、通用的多模态特征空间。在这个空间里，相似的概念（无论来自哪个模态）距离相近。

**BERT** (Bidirectional Encoder Representations from Transformers) 是一个具体、强大的“预训练语言模型”。它的核心任务是：

  - 理解文本: BERT是一个语言天才，它通过在海量的互联网文本上进行预训练，学会了语法、语义以及丰富的世界知识。
  - 生成文本特征: 它的专长是将输入的句子转换成高质量、蕴含丰富语义的数字向量（即特征向量或词嵌入）。
  - 该框架不仅实现了高精度的故障诊断，还内置了完善的消融实验流程，方便研究人员快速验证不同模型组件的有效性。

**这是一个基于CLIP思想的多模态故障诊断模型，其中文本编码部分我们采用了预训练的BERT模型。**

## ✨ 核心功能
- **多模态架构**: 结合用于振动信号的 1D-CNN 编码器和用于文本描述的 Transformer 语言模型。
- **两阶段训练**:
  - **零样本预训练**: 通过对比学习，建立振动信号与文本描述之间的深层联系。
  - **适配器微调**: 引入轻量级残差适配器（Residual Adapter），在冻结大部分参数的同时进行高效微调，进一步提升性能。
- **全面的性能评估**: 脚本可自动计算并输出准确率、精确率、召回率、F1 分数、AUROC，并生成带有详细指标的混淆矩阵图。
- **内置消融实验框架**:
  - 通过命令行参数轻松对比有/无适配器的性能。
  - 支持对提示词模板进行消融，文件名自动管理。

- **灵活的结果管理**: 评估结果（图表、预测文件）可以保存到任意指定的文件夹，方便实验管理。
- **特征可视化**: 利用 t-SNE 技术对模型提取的特征进行降维可视化，直观评估分类效果。

## 📁 项目结构
```
/  
├── bearingset/                 # 数据集根目录  
│   ├── train_set/              # 训练集  
│   └── test_set/               # 测试集  
|  
├── pretrained_weights/         # 存放训练好的模型权重  
│   ├── zero_shot_model.pth  
│   ├── adapter_tuned_model.pth  
│   ├── zero_shot_model_simple.pth  # 消融实验模型  
│   └── ...  
|  
├── result/                     # 默认的结果输出文件夹  
|  
├── data_loader.py              # 数据加载模块  
├── model.py                    # 模型定义模块  
├── train.py                    # 训练脚本 (自动化消融实验逻辑为train_bin.py)  
├── evaluate.py                 # 评估脚本  
├── requirements.txt            # Python 依赖库  
└── README.md                   # 本文档  
```

## 🛠️ 安装与设置
1. **克隆仓库**
   ```bash
   git clone https://github.com/Dreamlights001/Bearing-BERT
   cd bearing_BERT
   ```

2. **创建并激活虚拟环境 (推荐)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **处理 Hugging Face 模型下载 (重要！)**
   本项目需要从 Hugging Face 下载预训练的 DistilBERT 模型。由于网络原因，可能会下载失败。请选择以下任一方法解决：

   **方法 A (推荐): 设置环境变量使用镜像**
   在运行脚本的终端中，执行以下命令：
   ```bash
   # Windows (CMD)
   set HF_ENDPOINT=https://hf-mirror.com
   # Windows (PowerShell)
   $env:HF_ENDPOINT = "https://hf-mirror.com"
   # macOS / Linux
   export HF_ENDPOINT=https://hf-mirror.com
   ```

   **方法 B: 手动下载**
   参考之前的回答，将模型文件手动下载到本地文件夹（如 `./local_distilbert`），并修改 `model.py` 中加载模型的路径。

5. **准备数据集**
   将您的轴承数据集放置在根目录下的 `bearingset` 文件夹中，并确保其子目录 `train_set` 和 `test_set` 包含 CSV 数据文件。

## 🚀 核心工作流：训练与评估
### 阶段一: 零样本预训练 (Zero-Shot)
此阶段使用默认的完整提示词模板进行训练。
```bash
python train.py --mode zero_shot --epochs 30
```
**产出**: 在 `./pretrained_weights/` 目录下生成 `zero_shot_model.pth`。

### 阶段二: 适配器微调 (Adapter)
此阶段加载第一阶段的模型，并仅微调适配器。
```bash
python train.py --mode adapter --epochs 15 --lr 5e-5
```
**产出**: 在 `./pretrained_weights/` 目录下生成 `adapter_tuned_model.pth`。

### 阶段三: 评估模型
评估脚本可以评估任何模型，并通过 `--save_path` 指定结果输出目录。
```bash
# 评估带适配器的模型，并将结果保存在 ./results/adapter_final 文件夹
python evaluate.py --model_path ./pretrained_weights/adapter_tuned_model.pth --save_path ./results/adapter_final
```
**产出**: 终端打印详细性能指标，并在指定目录生成 `confusion_matrix_with_metrics.png`, `tsne_visualization.png` 和 `predictions.txt`。

## 🔬 高级用法: 消融实验
本框架的核心优势在于其便捷的消融实验流程。

### 实验 1: 残差适配器的有效性
**目的**: 验证适配器微调是否优于仅进行预训练。

**如何运行**: 分别评估在核心工作流中产出的两个模型。
```bash
# 评估无适配器的基础模型
python evaluate.py --model_path ./pretrained_weights/zero_shot_model.pth --save_path ./results/ablation_no_adapter

# 评估有适配器的微调模型
python evaluate.py --model_path ./pretrained_weights/adapter_tuned_model.pth --save_path ./results/ablation_with_adapter
```
**分析**: 比较 `results/ablation_no_adapter` 和 `results/ablation_with_adapter` 文件夹中的性能指标。

### 实验 2: 提示词模板的重要性
**目的**: 对比完整提示词 ("A {} bearing") 与简单标签 ("{}") 的效果。

**如何运行**: `train.py` 会根据 `--prompt` 参数自动处理文件名。

**训练和评估 (简单提示词)**:
```bash
# 训练 (会自动保存为 ..._simple.pth)
python train_bin.py --mode zero_shot --prompt "{}"
python train_bin.py --mode adapter --prompt "{}"

# 评估
python evaluate.py --model_path ./pretrained_weights/adapter_tuned_model_simple.pth --save_path ./results/ablation_simple_prompt
```

**训练和评估 (完整提示词)**:
使用核心工作流中已生成的 `adapter_tuned_model.pth` 进行评估，或重新运行不带 `--prompt` 参数的训练命令。

**分析**: 比较使用简单提示词和完整提示词的评估结果。

### 实验 3: 预训练语言模型的价值
**目的**: 对比强大的 DistilBERT 和从零训练的 Simple-LSTM 文本编码器的效果。

**如何运行**: 使用 `--text_encoder_type` 参数。
```bash
# 训练 (需要为模型命名，例如添加 --save_suffix)
python train.py --mode zero_shot --text_encoder_type simple_lstm

# 评估 (需要修改 evaluate.py 以识别和加载相应模型)
python evaluate.py --model_path <path_to_lstm_model> --text_encoder_type simple_lstm --save_path ./results/ablation_lstm_encoder
```
**注意**：此实验需要对 `train.py` 和 `evaluate.py` 进行微调以管理不同的模型文件，或在训练后手动重命名模型文件。

## ⚙️ 模型框架细节
- **振动编码器**: 轻量级的一维卷积神经网络 (1D-CNN)。
- **文本编码器**: 预训练的 DistilBERT 或可切换的 Embedding+LSTM。
- **关键公式 - 相似度计算**:
<p align="center">
  <img src="https://latex.codecogs.com/png.latex?%5Ctext%7Blogits%7D%20%3D%20%5Cexp%28%5Ctau%29%20%5Ccdot%20%28%5Ctext%7Bnormalize%7D%28V%29%20%5Ccdot%20%5Ctext%7Bnormalize%7D%28T%29%5ET%29" alt="Similarity Calculation">
</p>

- **关键公式 - 对称对比损失**:
<p align="center">
  <img src="https://latex.codecogs.com/png.latex?L_%7B%5Ctext%7Btotal%7D%7D%20%3D%20%5Cfrac%7B2%20%5Ccdot%20%5Ctext%7BCrossEntropy%7D%28L%2C%20%5Ctext%7Blabels%7D%29%20%2B%20%5Ctext%7BCrossEntropy%7D%28L%5ET%2C%20%5Ctext%7Blabels%7D%29%7D%7B2%7D" alt="Symmetric Contrastive Loss">
</p>

## 📄 许可证
本项目采用 MIT License 许可证。
