## Task2
### 代码基本结构
- img_dataset.py : 本地数据集处理 继承了pytorch 的Dataset类
- LeCNN .py ： 模仿LeNet写的CNN 修改了LeNet当中过时的部分
  - 采用两个卷积层  **卷积核数量、大小、step与Lenet相同**
  - 采用maxpool
  - 使用relu激活函数
  - 在全连接层和卷积层之后加入batchnorm
  - 在最后的全连接层之前加入dropout 增强网络的泛化能力
- trainfunc .py ：训练函数

### 对网络的改进
**参考AlexNet，AlexNet的任务相对较复杂，如果在我们的实验中直接引入AlexNet模型容易导致过拟合**
- dropout
  - 在网络中埋入多个分类器，使得网络不会过拟合
  - 同时增强网络的能力
- batchnor
  - 对每个batch进行normalization处理
  - 首先，可以加速训练并提高模型的收敛速度
  - 批量归一化引入了额外的参数（γ 和 β），可以作为正则化器，可以降低过拟合的风险
- 重叠的卷积窗口
  - 可以使得信息丢失更少
  - 增加感受野
  - 增强局部相关性


### 实验





### 对网络设计的理解