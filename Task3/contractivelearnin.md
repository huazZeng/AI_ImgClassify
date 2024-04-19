## contractive learning
### Instance Discrimination
* 将每个图片都当作一个类，进行个体判别的代理任务
* Memory Bank 存储图片的特征
* 动量更新 Memory Bank
* NCE loss

### Invariant Spreading
* end to end
* 进行个体判别的代理任务
* 缺陷：字典不够大

### CPC
* 生成式结构
* 预测序列接下来的输入
* 正样本为正确输入 负样本为随机的输入
  
### CMC
**Contrativeee Multiview Coding**
* 不同视角下的同一物体应该特征相近
* 正样本为同一物体的不同视角；负样本为不同的物体的任意视角
* 引入多模态 多视角
  
---
### MoCo
* 

### SimCLR