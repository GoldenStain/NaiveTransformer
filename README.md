# NaiveTransformer

## note1

在Transformer的embedding过程中乘以√d_model，**核心目的是控制数值稳定性**：
1. **方差平衡**：通过缩放使embedding向量的方差保持合理范围（接近1），避免后续线性变换（如Q/K/V投影）因维度（d_model）过大导致输出方差剧烈增长，从而缓解梯度爆炸或消失。
2. **参数初始化协同**：与embedding层的初始化（如小方差分布）配合，确保参数更新时梯度幅度稳定，避免训练初期收敛困难。
3. **位置编码匹配**：使token embedding的数值范围与位置编码相当，防止位置信息或语义信息一方主导输入。

此设计是Transformer训练稳定的关键细节之一，移除可能导致模型对超参数更敏感或收敛效率下降。
