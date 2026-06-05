# 导入必要的PyTorch库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入自定义的检索工具类
from layers.Retrieval import RetrievalTool, series_decomp


def compute_time_series_stats(tensor, dim=-1, keepdim=False):
    """
    计算时间序列的统计特征：均值、标准差、最小值、最大值

    Args:
        tensor: 输入张量，形状 [B, P, C] 或 [B, P, ...]
        dim: 计算统计的维度（默认最后一维）
        keepdim: 是否保持维度

    Returns:
        拼接的统计特征，形状 [..., 4]
    """
    mean_val = tensor.mean(dim=dim, keepdim=keepdim)
    std_val = tensor.std(dim=dim, keepdim=keepdim)
    min_val = torch.amin(tensor, dim=dim, keepdim=keepdim)
    max_val = torch.amax(tensor, dim=dim, keepdim=keepdim)

    # 拼接时，确保在最后一维拼接
    result = torch.cat([mean_val, std_val, min_val, max_val], dim=-1)

    # 如果 keepdim=False 且 dim 是元组，result 形状可能是 [B*4] 而不是 [B, 4]
    # 需要显式 reshape 成 [..., 4] 的形状
    if not keepdim and isinstance(dim, (tuple, int)) and result.dim() == 1:
        # 展平为 1D，需要 reshape
        result = result.view(-1, 4)

    return result


# STRAF模型类：基于检索增强的时序预测模型
class Model(nn.Module):

    def __init__(self, configs, individual=False):
        """
        初始化STRAF模型
        individual: Bool, 是否在不同变量间共享模型（预留参数）
        """
        super(Model, self).__init__()

        # 设置计算设备
        self.device = torch.device(f'cuda:{configs.gpu}')

        # 保存任务相关参数
        self.task_name = configs.task_name  # 任务类型：预测、分类、异常检测等
        self.seq_len = configs.seq_len      # 输入序列长度
        self.pred_len = configs.pred_len    # 预测序列长度

        # 根据任务类型调整预测长度
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len  # 对于这些任务，预测长度等于输入长度
        else:
            self.pred_len = configs.pred_len  # 对于预测任务，使用配置的预测长度

        # 保存输入通道数（特征维度）
        self.channels = configs.enc_in


        # 线性层：直接将输入序列映射到预测序列
        # 输入形状: [batch_size, seq_len] -> 输出形状: [batch_size, pred_len]
        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        
        # 使用 moving average based series_decomp（来自 DLinear）
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        # 两条线性头（shared across channels）：时间维从 seq_len -> pred_len
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        # 检索相关参数
        self.n_period = configs.n_period  # 使用的周期数
        self.topm = configs.topm          # 检索的top-k相似样本数
        self.sim_type = getattr(configs, 'sim_type', 'pearson')  # 相似度类型

        # 初始化检索工具
        self.rt = RetrievalTool(
            seq_len=self.seq_len, # 输入序列长度
            pred_len=self.pred_len, # 预测序列长度
            channels=self.channels, # 输入通道数
            n_period=self.n_period, # 使用的周期数
            topm=self.topm, # 检索的top-k相似样本数
            sim_type=self.sim_type, # 相似度类型
        )



        # 检索结果映射层：对检索预测结果进行线性映射
        # 现在有3个检索成分：seasonal, trend, original
        # 为每个成分添加独立的编码器
        self.encoder_seasonal = nn.Linear(self.pred_len, self.pred_len)
        self.encoder_trend = nn.Linear(self.pred_len, self.pred_len)
        self.encoder_original = nn.Linear(self.pred_len, self.pred_len)

        # 最终融合层：将线性预测和三个检索成分编码进行融合
        # 输入: 4*pred_len (线性预测 + 3个检索成分编码)
        # 输出: pred_len (最终预测结果)
        self.linear_pred = nn.Linear(4 * self.pred_len, self.pred_len)
        
        # Gate 融合模块（借鉴 ChronosBolt 的 gate 机制）
        # 高维方案：gate 输入包含原始预测值，而非仅统计量
        #
        # gate_input 组成：
        #   1. x_pred_from_x 展平: pred_len * channels
        #   2. retrieval_encoded 展平: 3 * pred_len * channels
        #   3. x_norm 的统计量: 4 * channels
        #   4. 各预测成分的统计量: 4 * 4 * channels = 16 * channels
        #
        # 总输入维度 = pred_len * C + 3 * pred_len * C + 4 * C + 16 * C
        #           = 4 * pred_len * C + 20 * C
        #           = C * (4 * pred_len + 20)
        # 对于 ETTh1: C=7, pred_len=192 → 7 * (768 + 20) = 7 * 788 = 5516
        gate_input_dim = self.channels * (4 * self.pred_len + 20)
        gate_hidden_dim = max(256, self.channels * 4)

        self.gate_layer = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
        )
        self.gate_linear1 = nn.Linear(gate_input_dim, gate_hidden_dim)
        # 输出到 1，经过 sigmoid 后得到 [0, 1] 的融合权重
        self.gate_linear2 = nn.Linear(gate_hidden_dim, 1)
        
        # --- MOE 多专家交互模块（借鉴 ChronosBolt 的 moe 设计） ---
        # 我们将模型预测视为一个专家，检索得到的每个成分视为额外专家，
        # 在时间步级别上对专家进行交互并基于注意力+gate进行加权融合。
        self.moe_d_model = max(64, self.channels * 8)
        self.encode_mlp = nn.Sequential(
            nn.Linear(self.channels, self.moe_d_model),
            nn.ReLU(),
            nn.Linear(self.moe_d_model, self.moe_d_model),
        )
        # 多头注意力用于专家间交互（在每个时间步对专家做交互）
        self.mha = nn.MultiheadAttention(embed_dim=self.moe_d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.moe_d_model, self.moe_d_model),
            nn.ReLU(),
            nn.Linear(self.moe_d_model, self.moe_d_model),
        )
        self.moe_gate_layer = nn.Sequential(
            nn.Linear(self.moe_d_model, self.moe_d_model),
            nn.ReLU(),
            nn.Linear(self.moe_d_model, 1),
        )
        # 将融合后的 d_model 映射回通道数
        self.moe_decode = nn.Linear(self.moe_d_model, self.channels)

        # ==================== 轻量级通道卷积融合模块 ====================
        # 核心思想：用1D卷积在通道维度上进行信息融合
        # 输入: [B, P, 2C] (当前预测 + 检索成分 拼接)
        # 处理: 逐点卷积 + 深度卷积
        # 输出: [B, P, C] (融合后的特征)
        self.channel_conv_fusion = nn.ModuleList([
            nn.Sequential(
                # 逐点卷积：改变通道数 (2C → C)
                nn.Conv1d(2 * self.channels, self.channels, kernel_size=1),
                nn.BatchNorm1d(self.channels),
                nn.ReLU(),
                # 深度卷积：通道内学习局部模式
                nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels),
                nn.BatchNorm1d(self.channels),
                nn.ReLU(),
            )
            for _ in range(3)  # 三个检索成分分别融合
        ])
        
        # 最终融合层：将融合后的特征与原始预测融合
        self.channel_fusion_linear = nn.Linear(2 * self.channels, self.channels)

#         if self.task_name == 'classification':
#             self.projection = nn.Linear(
#                 configs.enc_in * configs.seq_len, configs.num_class)

    # 数据集准备方法：构建检索库并预计算所有数据的检索结果
    # train_data: [B, S, C], valid_data: [B, S, C], test_data: [B, S, C]
    '''
    预计算过程详解：
    训练样本 i → 检索预测结果 pred_i[G, P, C]，每个周期内预测长度为P/G，每个周期内预测结果为P/G个时间步，每个时间步的预测结果为C个通道
    ↓ 存储到检索库中
    retrieval_dict['train'][:, i, :, :] = pred_i  # [G, P, C]
    这就是一个巨大的查找表！
    存储复杂度：O(G*N*P*C)
    查找复杂度：O(1)
    查找速度：O(1)
    查找空间：O(G*N*P*C)
    查找空间：O(G*N*P*C)
    '''
    def prepare_dataset(self, train_data, valid_data, test_data):
        # 使用检索工具准备训练数据集，建立检索库
        self.rt.prepare_dataset(train_data)

        # 初始化存储预计算检索结果的字典
        self.retrieval_dict = {}

        # 对训练数据进行检索（训练模式，避免数据泄露）
        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)
        # 返回形状: [G, N, P, C]
        # G: 周期数, N: 训练样本数, P: 预测长度, C: 通道数，每个周期内预测长度为P/G，每个周期内预测结果为P/G个时间步，每个时间步的预测结果为C个通道

        # 对验证数据进行检索（非训练模式）
        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        # 对测试数据进行检索（非训练模式）
        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        # 释放检索工具，节省显存
        del self.rt
        torch.cuda.empty_cache()

        # 将检索结果分离并保存到字典中
        # 这些结果将在推理时被重用，避免重复计算
        '''
预处理阶段：
├── 为训练集的每个样本计算检索预测 → 存储在 retrieval_dict['train']
├── 为验证集的每个样本计算检索预测 → 存储在 retrieval_dict['valid']  
└── 为测试集的每个样本计算检索预测 → 存储在 retrieval_dict['test']

推理阶段：
└── 根据样本索引直接查表获取预测结果
        '''
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    # # 编码器方法：将输入序列转换为预测序列（STRAF模型的核心方法）
    def encoder(self, x, index, mode): # x: [B, S, C], index: [B], mode: 'train' | 'valid' | 'test'
        # 将索引张量移动到指定设备，索引张量是查询样本的索引，用于从检索库中获取对应的预测结果
        index = index.to(self.device)

        # 获取输入数据的形状并进行有效性检查
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)

        # 计算输入数据的均值偏移（用于后续恢复原始尺度）
        x_offset = x.mean(dim=1, keepdim=True).detach()  # 在时间维度上求均值，形状: [B, 1, C]

        # 对输入数据进行归一化（减去均值）
        x_norm = x - x_offset  # 形状: [B, S, C]
       
        # 使用线性层进行直接预测
        # 先将维度调整为 [B, C, S]，然后线性变换，再转回 [B, P, C]
        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)  # [B, P, C]
        

        # 处理设备不匹配和边界检查问题
        # 确保索引在CPU上进行张量索引操作
        index_cpu = index.cpu() if index.device != torch.device('cpu') else index

        # 确保索引在有效范围内，防止数据加载器问题导致的越界
        retrieval_size = self.retrieval_dict[mode].size(1)  # 获取检索结果中样本数量
        index_cpu = torch.clamp(index_cpu, 0, retrieval_size - 1)  # 限制索引范围

        # 从预计算的检索结果中获取对应的预测结果
        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu]  # 形状: [G, B, P, C]
        pred_from_retrieval = pred_from_retrieval.to(self.device)
        
        # 分离三个检索成分
        # pred_from_retrieval 形状: [3, B, P, C] (seasonal + trend + original)
        # 成分顺序: 0=seasonal, 1=trend, 2=original
        seasonal_retrieval = pred_from_retrieval[0]  # [B, P, C]
        trend_retrieval = pred_from_retrieval[1]     # [B, P, C]
        original_retrieval = pred_from_retrieval[2]  # [B, P, C]

     

        # 对每个检索成分进行编码（将通道维映射到统一维度）
        # 先调整维度：[B, P, C] -> [B, C, P] -> 编码 -> [B, C, P] -> [B, P, C]
        seasonal_encoded = self.encoder_seasonal(seasonal_retrieval.permute(0, 2, 1)).permute(0, 2, 1)
        trend_encoded = self.encoder_trend(trend_retrieval.permute(0, 2, 1)).permute(0, 2, 1)
        original_encoded = self.encoder_original(original_retrieval.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 拼接三个编码后的检索成分
        retrieval_encoded = torch.cat([seasonal_encoded, trend_encoded, original_encoded], dim=1)  # [B, 3*P, C]
        

        # 将线性预测和检索编码进行拼接
        # 四个成分拼接后: [B, 4*P, C]
        concat_all = torch.cat([x_pred_from_x, retrieval_encoded], dim=1)  # [B, 4*P, C]

        # ==================== 融合策略选择 ====================
        # 可选融合方式：
        #   1. 'concat': 原始拼接方式
        #   2. 'gate': Gate 门控融合（借鉴 ChronosBolt）
        #   3. 'moe': MoE 多专家融合
        #   4. 'channel_conv': 轻量级通道卷积融合（借鉴TimeRAF思想）
        fusion_method = 'concat'  # 当前使用通道卷积融合

        if fusion_method == 'concat':
            # ==================== 方式1: 原始拼接融合 ====================
            # 直接拼接后通过线性层融合
            pred = torch.cat([x_pred_from_x, retrieval_encoded], dim=1)  # [B, 4*P, C]
            # [B, 4*P, C] -> [B, C, 4*P] -> [B, C, P] -> [B, P, C]
            pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)

        elif fusion_method == 'gate':
            # ==================== 方式2: Gate 门控融合（借鉴 ChronosBolt） ====================
            # 高维方案：gate 输入包含原始预测值，而非仅统计量
            #
            # 数据形状说明:
            #   x_norm:          [B, S, C]  输入序列
            #   x_pred_from_x:   [B, P, C]  模型预测
            #   seasonal_encoded: [B, P, C] 季节性检索
            #   trend_encoded:    [B, P, C] 趋势检索
            #   original_encoded: [B, P, C] 原始数据检索
            #   retrieval_encoded: [B, 3*P, C] 三个检索成分拼接
            #
            # gate_input 组成:
            #   1. x_pred_from_x 展平: [B, P*C]
            #   2. retrieval_encoded 展平: [B, 3*P*C]
            #   3. 统计特征: [B, 20*C] (5组统计量 × 4个值 × C通道)
            # 总维度: P*C + 3*P*C + 20*C = 4*P*C + 20*C = C * (4*P + 20)

            # 步骤1: 展平预测值（模型预测 + 检索编码）
            pred_flat = x_pred_from_x.reshape(bsz, -1)  # [B, P*C]
            retrieval_flat = retrieval_encoded.reshape(bsz, -1)  # [B, 3*P*C]

            # 步骤2: 计算各成分的统计特征（每个通道4个统计量）
            # 统计量形状: [B, C, 4] → view 成 [B, 4*C]
            def compute_channel_stats(tensor):
                """计算每个通道的全局统计量"""
                mean_val = tensor.mean(dim=(1,), keepdim=False)  # [B, C]
                std_val = tensor.std(dim=(1,), keepdim=False)   # [B, C]
                min_val = torch.amin(tensor, dim=(1,))          # [B, C]
                max_val = torch.amax(tensor, dim=(1,))          # [B, C]
                return torch.cat([mean_val, std_val, min_val, max_val], dim=-1)  # [B, 4*C]

            stats_input = compute_channel_stats(x_norm)        # [B, 4*C]
            stats_x_pred = compute_channel_stats(x_pred_from_x)  # [B, 4*C]
            stats_seasonal = compute_channel_stats(seasonal_encoded)  # [B, 4*C]
            stats_trend = compute_channel_stats(trend_encoded)  # [B, 4*C]
            stats_original = compute_channel_stats(original_encoded)  # [B, 4*C]

            # 步骤3: 拼接所有特征
            gate_input = torch.cat([
                pred_flat,        # [B, P*C]，模型预测
                retrieval_flat,   # [B, 3*P*C]，检索编码
                stats_input,      # [B, 4*C]，输入统计
                stats_x_pred,     # [B, 4*C]，预测统计
                stats_seasonal,   # [B, 4*C]，季节性统计
                stats_trend,      # [B, 4*C]，趋势统计
                stats_original,   # [B, 4*C]，原始数据统计
            ], dim=-1)  # [B, C*(P + 3P + 20)] = [B, C*(4P + 20)]

            # 步骤4: 通过 gate 网络计算融合权重
            gate = self.gate_layer(gate_input) + self.gate_linear1(gate_input)  # [B, gate_hidden_dim]
            gate = torch.sigmoid(self.gate_linear2(gate))  # [B, 1]，值域 [0, 1]

            # 步骤5: 扩展 gate 到 [B, P, 1] 以便与 [B, P, C] 进行广播
            gate = gate.unsqueeze(1).expand(-1, self.pred_len, 1)  # [B, 1] -> [B, P, 1]

            # 步骤6: 使用 gate 权重融合线性预测和检索预测
            # retrieval_encoded 是三个检索成分的拼接: [B, 3*P, C]
            retrieval_agg = retrieval_encoded.mean(dim=1, keepdim=True)  # [B, 1, C]，聚合三个检索成分

            # 融合策略：gate * retrieval_agg + (1 - gate) * x_pred_from_x
            pred = gate * retrieval_agg + (1.0 - gate) * x_pred_from_x  # [B, P, C]

        elif fusion_method == 'moe':
            # ==================== 方式3: MoE 多专家融合（借鉴 ChronosBolt） ====================
            # 专家定义：模型自身预测 + 三个检索成分
            # 专家0: x_pred_from_x [B, P, C]
            # 专家1: seasonal_encoded [B, P, C]
            # 专家2: trend_encoded [B, P, C]
            # 专家3: original_encoded [B, P, C]
            experts = [
                x_pred_from_x.unsqueeze(1),      # [B, 1, P, C]
                seasonal_encoded.unsqueeze(1),   # [B, 1, P, C]
                trend_encoded.unsqueeze(1),      # [B, 1, P, C]
                original_encoded.unsqueeze(1),   # [B, 1, P, C]
            ]
            all_experts = torch.cat(experts, dim=1)  # [B, E=4, P, C]

            B, E, P, C = all_experts.shape

            # 编码每个专家的通道维到 d_model
            all_flat = all_experts.reshape(-1, C)  # [B*E*P, C]
            enc_flat = self.encode_mlp(all_flat)   # [B*E*P, d_model]
            enc = enc_flat.reshape(B, E, P, self.moe_d_model)  # [B, E, P, d_model]

            # 变换为按时间步的专家序列：[B, P, E, d_model]
            enc = enc.permute(0, 2, 1, 3).contiguous()  # [B, P, E, d_model]
            enc_bp = enc.reshape(B * P, E, self.moe_d_model)  # [B*P, E, d_model]

            # 多头注意力在专家序列上交互
            att_out, _ = self.mha(enc_bp, enc_bp, enc_bp)  # [B*P, E, d_model]

            # 残差 + FFN
            att_out = att_out + self.ffn(att_out)

            # 计算每个专家的 gate 分数并做 softmax（在专家维度）
            gate_scores = self.moe_gate_layer(att_out)  # [B*P, E, 1]
            alpha = F.softmax(gate_scores, dim=1)       # [B*P, E, 1]

            # 加权融合专家表示，得到 [B*P, d_model]
            fused = torch.sum(alpha * att_out, dim=1)   # [B*P, d_model]
            fused = fused.reshape(B, P, self.moe_d_model)  # [B, P, d_model]

            # 解码回通道维并作为最终预测
            pred = self.moe_decode(fused)  # [B, P, C]

        elif fusion_method == 'channel_conv':
            # ==================== 方式4: 轻量级通道卷积融合 ====================
            # 核心思想：在通道维度上用1D卷积进行信息融合，学习通道间依赖
            #
            # 数据形状说明:
            #   x_pred_from_x:   [B, P, C]  模型预测
            #   seasonal_encoded: [B, P, C] 季节性检索
            #   trend_encoded:    [B, P, C] 趋势检索
            #   original_encoded: [B, P, C] 原始数据检索
            #
            # 融合步骤:
            #   1. 拼接当前预测和每个检索成分 → [B, P, 2C]
            #   2. 通道卷积融合 → [B, P, C]
            #   3. 聚合三个融合结果 → [B, P, C]
            #   4. 与原始预测拼接后通过线性层 → [B, P, C]

            def apply_channel_conv(x_pred, retrieval_comp, fusion_module):
                """
                对当前预测和检索成分应用通道卷积融合
                
                Args:
                    x_pred: [B, P, C] 模型预测
                    retrieval_comp: [B, P, C] 检索成分
                    fusion_module: 融合模块
                
                Returns:
                    fused: [B, P, C] 融合后的特征
                """
                # 步骤1: 拼接 [B, P, C] + [B, P, C] → [B, P, 2C]
                concat = torch.cat([x_pred, retrieval_comp], dim=-1)
                
                # 步骤2: 维度变换 [B, P, 2C] → [B, 2C, P]
                concat_t = concat.permute(0, 2, 1)
                
                # 步骤3: 通道卷积融合
                fused_t = fusion_module(concat_t)  # [B, C, P]
                
                # 步骤4: 维度变回 [B, C, P] → [B, P, C]
                fused = fused_t.permute(0, 2, 1)
                
                return fused

            # 步骤1: 对每个检索成分分别进行通道卷积融合
            seasonal_fused = apply_channel_conv(
                x_pred_from_x, seasonal_encoded, self.channel_conv_fusion[0]
            )  # [B, P, C]
            
            trend_fused = apply_channel_conv(
                x_pred_from_x, trend_encoded, self.channel_conv_fusion[1]
            )  # [B, P, C]
            
            original_fused = apply_channel_conv(
                x_pred_from_x, original_encoded, self.channel_conv_fusion[2]
            )  # [B, P, C]

            # 步骤2: 聚合三个融合结果
            # 思路：seasonal + trend = 分解后的预测（来自 DLinear 思想）
            #       original_fused = 直接检索原始数据的预测
            #       两者再做平均或加权融合
            
            # 季节性 + 趋势性 先相加（分解预测）
            decomposition_fused = seasonal_fused + trend_fused  # [B, P, C]
            
            # 再和原始数据检索结果做平均
            #retrieval_fused = (decomposition_fused + original_fused) / 2.0  # [B, P, C]
            retrieval_fused = torch.stack([
                decomposition_fused, original_fused
            ], dim=1).mean(dim=1)  # [B, P, C]
            # 或者使用加权融合（可学习权重）
            # fusion_weight = torch.sigmoid(self.decomp_original_gate)  # 可学习的融合权重
            # retrieval_fused = fusion_weight * decomposition_fused + (1 - fusion_weight) * original_fused

            # 步骤3: 将融合结果与原始预测拼接
            concat_final = torch.cat([
                x_pred_from_x, retrieval_fused
            ], dim=-1)  # [B, P, 2C]

            # 步骤4: 通过线性层得到最终预测
            # nn.Linear 需要 2D 输入 [batch_size, features]
            # [B, P, 2C] → [B*P, 2C] → linear → [B*P, C] → [B, P, C]
            B, P, _ = concat_final.shape
            pred = self.channel_fusion_linear(concat_final.reshape(B * P, -1))  # [B*P, C]
            pred = pred.reshape(B, P, -1)  # [B, P, C]

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # 恢复原始尺度（加上之前减去的均值偏移）
        pred = pred + x_offset

        return pred
        # 编码器方法：将输入序列转换为预测序列（STRAF模型的核心方法）
    # def encoder(self, x, index, mode): # x: [B, S, C], index: [B], mode: 'train' | 'valid' | 'test'
    #     # 将索引张量移动到指定设备，索引张量是查询样本的索引，用于从检索库中获取对应的预测结果
    #     index = index.to(self.device)

    #     # 获取输入数据的形状并进行有效性检查
    #     bsz, seq_len, channels = x.shape
    #     assert(seq_len == self.seq_len, channels == self.channels)

    #     # 计算输入数据的均值偏移（用于后续恢复原始尺度）
    #     x_offset = x.mean(dim=1, keepdim=True).detach()  # 在时间维度上求均值，形状: [B, 1, C]

    #     # 对输入数据进行归一化（减去均值）
    #     x_norm = x - x_offset  # 形状: [B, S, C]

    #     # 使用线性层进行直接预测
    #     # 先将维度调整为 [B, C, S]，然后线性变换，再转回 [B, P, C]
    #     x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)  # [B, P, C]

    #     # # 处理设备不匹配和边界检查问题
    #     # # 确保索引在CPU上进行张量索引操作
    #     # index_cpu = index.cpu() if index.device != torch.device('cpu') else index

    #     # # 确保索引在有效范围内，防止数据加载器问题导致的越界
    #     # retrieval_size = self.retrieval_dict[mode].size(1)  # 获取检索结果中样本数量
    #     # index_cpu = torch.clamp(index_cpu, 0, retrieval_size - 1)  # 限制索引范围

    #     # # 从预计算的检索结果中获取对应的预测结果
    #     # pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu]  # 形状: [G, B, P, C]
    #     # pred_from_retrieval = pred_from_retrieval.to(self.device)

    #     # # 初始化存储处理后检索预测的列表
    #     # retrieval_pred_list = []

    #     # # 对每个周期粒度的检索结果进行后处理
    #     # for i, pr in enumerate(pred_from_retrieval):
    #     #     # 验证形状正确性
    #     #     assert((bsz, self.pred_len, channels) == pr.shape)

    #     #     # 获取当前周期长度
    #     #     g = self.period_num[i]

    #     #     # 压缩重复维度：将预测序列按周期长度分组
    #     #     # 例如：pred_len=24, g=4 -> [B, 24, C] -> [B, 6, 4, C]
    #     #     pr = pr.reshape(bsz, self.pred_len // g, g, channels)

    #     #     # 只保留每个周期组的第一个元素（减少冗余信息）
    #     #     pr = pr[:, :, 0, :]  # 形状: [B, pred_len//g, C]

    #     #     # 使用对应的线性层进行预测长度恢复
    #     #     # [B, pred_len//g, C] -> [B, C, pred_len//g] -> [B, pred_len//g, C] -> [B, C, pred_len] -> [B, pred_len, C]
    #     #     pr = self.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
    #     #     pr = pr.reshape(bsz, self.pred_len, self.channels)  # 最终形状: [B, P, C]

    #     #     retrieval_pred_list.append(pr)

    #     # # 将所有周期粒度的预测结果堆叠并求和
    #     # retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)  # [B, G, P, C]
    #     # retrieval_pred_list = retrieval_pred_list.sum(dim=1)           # [B, P, C]

    #     # # 将线性预测和检索预测进行拼接
    #     # pred = torch.cat([x_pred_from_x, retrieval_pred_list], dim=1)  # [B, 2*P, C]

    #     # # 使用最终融合层进行预测
    #     # # [B, 2*P, C] -> [B, C, 2*P] -> [B, C, P] -> [B, P, C]
    #     # pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)

    #     # 恢复原始尺度（加上之前减去的均值偏移）
    #     pred = x_pred_from_x + x_offset

    #     return pred

    # 预测方法：用于时序预测任务
    def forecast(self, x_enc, index, mode):
        # 调用编码器进行预测
        return self.encoder(x_enc, index, mode)

    # 插补方法：用于缺失数据填补任务
    def imputation(self, x_enc, index, mode):
        # 调用编码器进行插补
        return self.encoder(x_enc, index, mode)

    # 异常检测方法：用于异常检测任务
    def anomaly_detection(self, x_enc, index, mode):
        # 调用编码器进行异常检测
        return self.encoder(x_enc, index, mode)

    # 分类方法：用于时序分类任务
    def classification(self, x_enc, index, mode):
        # 调用编码器获取编码输出
        enc_out = self.encoder(x_enc, index, mode)

        # 将输出展平为向量形式
        # [batch_size, seq_length, channels] -> [batch_size, seq_length * channels]
        output = enc_out.reshape(enc_out.shape[0], -1)

        # 通过分类投影层得到最终分类结果
        # [batch_size, seq_length * channels] -> [batch_size, num_classes]
        output = self.projection(output)
        return output

    # 前向传播方法：根据任务类型调用相应的处理方法
    def forward(self, x_enc, index, mode='train'):
        # 时序预测任务（长期预测或短期预测）
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            # 返回最后pred_len个时间步的预测结果
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        # 数据插补任务
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            # 返回完整的插补结果
            return dec_out  # [B, L, D]

        # 异常检测任务
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            # 返回异常检测结果
            return dec_out  # [B, L, D]

        # 分类任务
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            # 返回分类概率或类别索引
            return dec_out  # [B, N]

        # 未知任务类型
        return None
