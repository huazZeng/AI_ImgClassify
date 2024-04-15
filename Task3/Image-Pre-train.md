# image Pre-train
---
## 一、Supervised Learning{#custom-id}

**使用有标签的图片来做分类任务以训练模型，强化模型的特征提取能力**


#### 数据准备

#### 超参设置

#### 模型结构
* 基于CNN的Res-Net50
* 基于transformer的ViT

#### 训练目标函数
训练时使用分类任务，损失函数为交叉熵

#### 评估
在训练结束后，与未训练过的模型同时去适应同一个下游任务，比较二者的训练速度与训练效果。


---


## 二、Self-Supervised Learning
### 基于上下文预测的预训练方法
**类似BERT的Mask机制**

#### 数据准备

#### 超参设置

#### 模型结构

#### 训练目标函数

#### 评估

---
### 基于对比的预训练训练方法


#### 数据准备

#### 超参设置

#### 模型结构

#### 训练目标函数

#### 评估


