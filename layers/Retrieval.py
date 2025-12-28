import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


# ==========================================
# 独立工具函数：趋势-季节分解
# ==========================================
def trend_season_decompose(data_all, kernel_size=None, seq_len=None):
    """
    趋势-季节分解：使用简单移动平均作为趋势，残差作为季节/中高频成分

    Args:
        data_all: (B, L, C) - 输入序列
        kernel_size: int - 平滑核大小，默认自适应
        seq_len: int - 序列长度，用于自适应kernel_size计算

    Returns:
        trend: (B, L, C) - 低频趋势成分
        seasonal: (B, L, C) - 中高频季节/残差成分
    """
    # 自适应平滑核：长度的 1/4，最小 3，且保证为奇数
    if kernel_size is None:
        if seq_len is not None:
            kernel_size = max(3, seq_len // 4)
        else:
            # 如果没有提供seq_len，从data_all推断
            seq_len = data_all.shape[1]
            kernel_size = max(3, seq_len // 4)

    if kernel_size % 2 == 0:
        kernel_size += 1

    # data_all: B, L, C  →  B, C, L 方便用 1D 池化
    x = data_all.permute(0, 2, 1)  # B, C, L
    pad = (kernel_size - 1) // 2

    # 边界使用 replicate 填充，保持长度不变
    x_padded = F.pad(x, (pad, pad), mode='replicate')
    trend = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1)

    # 回到 (B, L, C)
    trend = trend.permute(0, 2, 1)
    seasonal = data_all - trend

    return trend, seasonal



#检索工具
class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=3,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
        device=None,
    ):

        period_num = [16, 8, 4, 2,1]
        period_num = period_num[-1 * n_period:] #只取最后 n_period 个周期

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)
        
        self.temperature = temperature
        self.topm = topm
        
        self.with_dec = with_dec
        self.return_key = return_key
        
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device



    #准备数据集
    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []
        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])
            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])
            
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()

        # 一阶段：先做「趋势 + 季节」分解（移动平均作为趋势，残差作为季节/中高频）
        self.train_trend_all, self.train_season_all = self.trend_season_decompose(
            self.train_data_all
        )
        
        # 二阶段：只对中高频（季节）部分做多周期分解
        self.train_data_all_mg, _ = self.decompose_mg(self.train_season_all)
        
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_trend_all, self.y_season_all = self.trend_season_decompose(
            self.y_data_all
        )
        self.y_data_all_mg, _ = self.decompose_mg(self.y_season_all)

        self.n_train = self.train_data_all.shape[0]
        

    '''一阶段趋势/季节分解：使用简单移动平均作为趋势，残差作为季节/中高频成分
    输入: data_all (B, L, C)
    输出:
        trend   (B, L, C)  低频趋势
        seasonal(B, L, C)  中高频季节/残差
    '''
    def trend_season_decompose(self, data_all, kernel_size=None):
        # 调用独立的趋势-季节分解函数
        return trend_season_decompose(data_all, kernel_size, self.seq_len)
        
    '''data_all,就是一段已经切好、只含历史输入以及未来目标的序列的季节成分，不含别的拼接。'''
    def decompose_mg(self, data_all, remove_offset=True):
        data_all = copy.deepcopy(data_all) # T, S, C

        mg = []

        '''
        再做多层周期分解（多周期下采样）
        每一步把当前周期分量算出来，然后从「季节序列」里减掉它，再拿残差继续下一层。
        第一次循环：提取最强周期（最大 g）→ 存 mg[0]
        第二次循环：在残差里提取次强周期（较小 g）→ 存 mg[1]
        ……
        最终 mg 列表里保存的是从长到短各周期层的分量，形状 (G, L, 1, C)'''

        for g in self.period_num:   # 自动周期表见下一节
            cur = data_all.unfold(dimension=1, size=g, step=1).mean(dim=-1)  #把 g 步一段求平均，得到该周期的一层平滑分量
            # === pad 回到原始长度 ===
            pad_left = (g - 1) // 2
            pad_right = g - 1 - pad_left
            # 用边界值填充（replicate）
            cur = torch.cat([
                cur[:, 0:1, :].repeat(1, pad_left, 1),
                cur,
                cur[:, -1:, :].repeat(1, pad_right, 1)
            ], dim=1)  # [T, S, C]
            #cur = cur.repeat_interleave(repeats=g, dim=1)  #插值回原始长度 L=seq_len
            
            mg.append(cur)  #存下这层周期分量
            data_all = data_all - cur # T, S, C 从原序列里减掉它，再拿残差继续下一层。
            
        mg = torch.stack(mg, dim=0) # G, T, S, C 堆叠

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p.mean(dim=1, keepdim=True)  # 在时间维度上求均值
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
        else:
            offset = None
            
        offset = torch.stack(offset, dim=0)
            
        return mg, offset

    '''对每条查询向量，并行计算它与整个训练库在所有周期粒度上的 Pearson ≈ 余弦相似度，返回一个「相似度矩阵」，供后续 Top-K 加权使用。
    职责：计算相似度矩阵
    输入：查询序列、训练库
    输出：当前查询序列和训练库的相似度矩阵（G,B,T）
    '''
    '''
    **data_all**: `(G, T, S*C)` - 训练库（已展平）
  - G = 周期组数（如 3 组：16, 8, 4）
  - T = 训练集样本数
  - S*C = 展平后的特征维度（序列长度 × 通道数）

  - **key**: `(G, B, S*C)` - 查询序列（已展平）
  - G = 周期组数
  - B = batch 大小
  - S*C = 展平后的特征维度

  - **in_bsz**: `int` - 内部批处理大小（默认 512）
  - 用于分批处理训练库，避免内存溢出
    '''
    def periodic_batch_corr(self, data_all, key, in_bsz = 64):  #批量计算相似度 
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape #data_all 训练库（key 仓库）G=周期组数，T=样本数，S·C=展平后的特征

        #真 Pearson：零中心 + 单位方差
        '''**功能**：计算查询序列的均值
- `dim=2`：在特征维度（S*C）上求均值
- `keepdim=True`：保持维度，便于广播
- **结果形状**：`(G, B, 1)` '''
        bx_mean = key.mean(dim=2, keepdim=True)
        bx_std = key.std(dim=2, keepdim=True)+1e-5
        bx = (key - bx_mean) / bx_std
        #bx = key - torch.mean(key, dim=2, keepdim=True)  #当前批次的查询数据，已经展平，bx.shape = G , B, S*C，查询数据减去均值，得到bx
        
        iters = math.ceil(train_len / in_bsz)  #**功能**：计算需要多少批次处理训练库
        
        sim = []
        for i in range(iters): #**功能**：遍历每个批次- 分批处理训练库，避免一次性加载所有数据
            #**功能**：计算当前批次的索引范围
            start_idx = i * in_bsz  #- `start_idx`：起始索引
            end_idx = min((i + 1) * in_bsz, train_len) #- `end_idx`：结束索引（不超过训练集大小）
            
            #- **结果形状**：`(G, batch_size, S*C)`
            cur_data = data_all[:, start_idx:end_idx].to(key.device) #**功能**：提取当前批次的训练数据 - `[:, start_idx:end_idx]`：切片，提取第 i 批数据
            ax_mean = cur_data.mean(dim=2, keepdim=True)
            ax_std = cur_data.std(dim=2, keepdim=True)+1e-5
            ax = (cur_data - ax_mean) / ax_std
            #ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True) #ax
            
            '''
            1. **F.normalize(bx, dim=2)**：
   - 对 bx 在特征维度上做 L2 归一化
   - **结果形状**：`(G, B, S*C)`
   - **数学表示**：$\frac{\text{bx}}{||\text{bx}||_2}$

2. **F.normalize(ax, dim=2)**：
   - 对 ax 在特征维度上做 L2 归一化
   - **结果形状**：`(G, batch_size, S*C)`

3. **.transpose(-1, -2)**：
   - 转置最后一个和倒数第二个维度
   - **结果形状**：`(G, S*C, batch_size)`
   - **目的**：为矩阵乘法做准备

4. **torch.bmm()**：
   - 批量矩阵乘法
   - **公式**：`(G, B, S*C) × (G, S*C, batch_size) = (G, B, batch_size)`
经过标准化（零均值 + 单位方差）后，L2 归一化的向量点积等于 Pearson 相关系数：
            '''
            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)
            
        sim = torch.cat(sim, dim=2)  #得到相似度矩阵
        
        return sim
    '''
#### 返回值
- **sim**: `(G, B, T)` - 相似度矩阵
  - sim[g, b, t] = 样本 b 在周期 g 上与训练样本 t 的 Pearson/余弦相似度
    '''

    '''
    职责：完整检索流程
    输入：查询序列、索引
    输出：检索预测 (G, B, P, C),P:历史输出序列  C:通道数
    包含：
     - 周期分解
     - 相似度计算（调用 periodic_batch_corr）
     - 自掩码处理
     - Top-K 筛选
     - 加权平均预测
   - 返回最终的检索预测值
    '''  
    def retrieve(self, x, index, train=True):
        '''
        给定一个 batch 的历史输入 x，返回 G 组周期粒度的「检索预测」(G, B, P, C)——即 用训练集里最像的历史片段对应的未来值，加权平均后作为当前 batch 的预测候选。
        P = pred_len（未来段长度）
        N 就是 训练集里样本的总数（n_train）
        相似度矩阵 sim 的 T 维、检索库 y_data_all 的 T 维，都等于这个 N。
        所以：
        N = 训练集样本数 = 检索库大小=t
        x	(B, L, C)	当前 batch 的历史输入
        index	(B,)	样本在数据集中的全局编号（用于自掩码）
        返回值 pred_from_retrieval	(G, B, P, C)	各周期粒度下的检索预测
        '''
        index = index.to(x.device)
        
        bsz, seq_len, channels = x.shape
        assert seq_len == self.seq_len and channels == self.channels

        # 1) 对当前输入做同样的趋势/季节分解
        #    只对季节（中高频）部分做多周期分解，以和训练库保持一致
        x_trend, x_seasonal = self.trend_season_decompose(x)
        x_mg, mg_offset = self.decompose_mg(x_seasonal)  # G, B, S, C

        '''
        3. 相似度矩阵：查询 vs 训练库
        T = 训练集样本数
        输出 sim[g, b, t] = 样本 b 在周期 g 上与训练样本 t 的 Pearson/余弦相似度

        '''
        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2), # G, T, S * C  
            x_mg.flatten(start_dim=2), # G, B, S * C  s=seq_len
        ) # G, B, T
        '''
        4. 自掩码：排除当前 batch 的样本
        为了防止模型「抄袭」自己的过往预测，需要对相似度矩阵进行自掩码处理。
        思路：把当前 batch 的样本（在训练集中的全局编号），填到相似度矩阵的对应位置上，让模型看不到自己。
        '''
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)
            
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            # 使用当前 sim 的 G 维度，而不是固定的 n_period，防止维度不一致
            G = sim.size(0)
            self_mask = self_mask.unsqueeze(dim=0).repeat(G, 1, 1)
            
            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        G = sim.size(0)
        sim = sim.reshape(G * bsz, self.n_train) # G X B, T   sim	(G*B, T)	所有训练样本的相似度（已 reshape 成 2D）
                
        '''
        5. Top-M 保留 & 温度 Softmax
        只保留相似度最高的 topm 个训练样本；
        用 温度 τ 控制分布尖锐程度（τ→0 近似 one-hot，τ→∞ 均匀）。

        '''
        topm_index = torch.topk(sim, self.topm, dim=1).indices #topm_index.shape = G, B, topm  topm_index	(G*B, topm)	每个 batch 的 topm 个相似样本索引
        ranking_sim = torch.ones_like(sim) * float('-inf')  # 先全设 -inf
        
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)  # 行索引
        ranking_sim[rows, topm_index] = sim[rows, topm_index]  # 只填 Top-M 值,其余置 -inf
        
        sim = sim.reshape(G, bsz, self.n_train) # G, B, T   恢复 3D 形状
        ranking_sim = ranking_sim.reshape(G, bsz, self.n_train) # G, B, T   方便按样本按周期做 Softmax  

        data_len, seq_len, channels = self.train_data_all.shape #data_len=T, seq_len=L, channels=C
            
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2) # G, B, T   softmax 归一化，得到概率分布
        ranking_prob = ranking_prob.detach().cpu() # G, B, T  
        

        '''
        6. 加权回归：相似度 → 预测值
        把「未来目标周期分量」按相同权重加权平均，得到 纯残差空间的检索预测。
        后续在外面再加回偏移量，就得到最终预测。
        '''
        y_data_all = self.y_data_all_mg.flatten(start_dim=2) # G, T, P * C
        
        pred_from_retrieval = torch.bmm(ranking_prob, y_data_all).reshape(G, bsz, -1, channels) # G, B, P, C
        pred_from_retrieval = pred_from_retrieval.to(x.device)
        
        return pred_from_retrieval
    '''

    把整个数据集（或任意子集）一次性跑完，返回 G×N×P×C 的「检索预测」大矩阵，供后续模型融合或存盘复用。
    retrieve_all()  ← 批量检索入口
    ↓
   调用 retrieve()  ← 单个 batch 检索
        ↓
       调用 periodic_batch_corr()  ← 相似度计算核心
    遍历整个数据集
   - 对每个 batch 调用 retrieve
   - 拼接所有结果，返回完整检索预测库

`data`: Dataset 对象，包含要检索的数据集（训练/验证/测试）
`train`: 布尔值，是否处于训练模式（影响自掩码行为）
`device`: 设备，用于计算（默认 CPU）
返回值：`(G, N, P, C)` - 完整检索预测库，N 是样本个数，P 是每个样本要预测的未来步长

功能：
1. 创建数据加载器（DataLoader）：
   - 从 `data` 创建一个数据加载器，设置批量大小为 1024，不 shuffle（保持顺序），使用 8 个工作线程，不丢弃最后一个 batch。
2. 初始化检索列表：
   - 创建一个空列表 `retrievals`，用于存储所有 batch 的检索预测。
3. 在 no_grad 上下文中循环处理数据：
   - 对每个 batch 调用 `retrieve()` 进行检索，得到 `pred_from_retrieval`。
   - 将结果从 GPU 移动到 CPU，并添加到 `retrievals` 列表中。
4. 拼接所有结果：
   - 将 `retrievals` 列表中的所有检索预测拼接在一起，形成最终的完整检索预测库。
5. 返回结果：
   - 返回形状为 `(G, N, P, C)` 的完整检索预测库。
  - `G`: 周期组数（如 3 组：16, 8, 4）
  - `N`: 数据集样本总数
  - `P`: 预测长度（pred_len）
  - `C`: 通道数（特征数）
    '''
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):  
        assert(self.train_data_all_mg != None)
        ## 核心流程

        ### 1. 创建 DataLoader
        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )
        
        retrievals = []
        with torch.no_grad():
            ## 2. 循环处理数据
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)
                
        retrievals = torch.cat(retrievals, dim=1) #(G, N, P, C) N 是"样本个数"，P 是"每个样本要预测的未来步长".所有 batch 的检索预测拼接在一起

        return retrievals


# ==========================================
# 缺失模块 1：季节预测分支（核心）
# ==========================================
# 角色：在整体模型中负责对分解出的各周期尺度季节成分进行参数化动力外推，
#       提供结构化预测能力，与检索分支形成互补
class SeasonalPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, channels, n_period):
        super(SeasonalPredictor, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_period = n_period

        # 为每个周期尺度创建独立的线性预测器
        # 每个预测器将当前尺度的历史季节序列映射到未来预测
        self.scale_predictors = nn.ModuleList([
            nn.Linear(seq_len * channels, pred_len * channels)
            for _ in range(n_period)
        ])

    def forward(self, seasonal_scales):
        """
        参数化预测各周期尺度的季节成分

        Args:
            seasonal_scales: (G, B, L, C) - 各周期尺度的季节成分

        Returns:
            pred_seasonal: (G, B, pred_len, C) - 各尺度的季节预测
        """
        G, B, L, C = seasonal_scales.shape
        assert L == self.seq_len

        pred_seasonal_list = []

        for g in range(G):
            # 当前尺度的季节序列 (B, L, C) -> 展平 (B, L*C)
            scale_input = seasonal_scales[g].reshape(B, -1)  # (B, L*C)

            # 线性预测器生成未来预测 (B, pred_len*C)
            scale_output = self.scale_predictors[g](scale_input)  # (B, pred_len*C)

            # 重塑为预测形状 (B, pred_len, C)
            scale_pred = scale_output.reshape(B, self.pred_len, C)
            pred_seasonal_list.append(scale_pred)

        # 堆叠所有尺度的预测 (G, B, pred_len, C)
        pred_seasonal = torch.stack(pred_seasonal_list, dim=0)

        return pred_seasonal


# ==========================================
# 缺失模块 2：多尺度预测结果融合（预测级融合）
# ==========================================
# 角色：在整体模型中负责将各周期尺度的预测结果进行合理融合，
#       基于尺度一致性权重，避免特征混淆
class MultiScaleFusion(nn.Module):
    def __init__(self, seq_len, pred_len, channels, n_period):
        super(MultiScaleFusion, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.n_period = n_period

        # 为每个尺度学习融合权重（基于尺度的相对重要性）
        # 权重表示该尺度在最终预测中的贡献度
        self.scale_weights = nn.Parameter(torch.ones(n_period))

    def forward(self, scale_predictions):
        """
        基于尺度一致性对多尺度预测进行加权融合

        Args:
            scale_predictions: (G, B, pred_len, C) - 各尺度的预测结果

        Returns:
            fused_prediction: (B, pred_len, C) - 融合后的季节预测
        """
        G, B, P, C = scale_predictions.shape
        assert P == self.pred_len

        # 对权重进行softmax归一化，确保权重之和为1
        # 权重含义：scale_weights[g]表示第g个周期尺度的重要性
        weights = F.softmax(self.scale_weights, dim=0)  # (G,)

        # 加权融合：对各尺度预测进行加权平均
        # weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) -> (G, 1, 1, 1)
        # scale_predictions * weights -> (G, B, P, C)
        # sum(dim=0) -> (B, P, C)
        fused_prediction = (scale_predictions * weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

        return fused_prediction


# ==========================================
# 缺失模块 3：参数化预测 vs 历史检索的融合机制
# ==========================================
# 角色：在整体模型中负责平衡结构外推（参数化预测）和经验约束（检索预测），
#       通过检索相似度自适应调节融合权重
class ParamRetrievalFusion(nn.Module):
    def __init__(self, seq_len, pred_len, channels, temperature=0.1):
        super(ParamRetrievalFusion, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.temperature = temperature

        # 学习基础融合权重
        self.base_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, param_pred, retrieval_pred, retrieval_sim=None):
        """
        基于检索相似度统计的自适应融合

        Args:
            param_pred: (B, pred_len, C) - 参数化预测结果
            retrieval_pred: (B, pred_len, C) - 检索预测结果
            retrieval_sim: (G, B, T) - 检索相似度矩阵，可选，用于计算alpha

        Returns:
            fused_pred: (B, pred_len, C) - 融合预测结果
            alpha: (B,) - 融合权重，物理含义：参数化预测的置信度/重要性
        """
        B, P, C = param_pred.shape
        assert P == self.pred_len

        if retrieval_sim is not None:
            # 方法1：基于检索相似度的统计量计算alpha
            # 使用top-k相似度的均值作为检索可靠性的指标
            # retrieval_sim: (G, B, T)
            top_k_values, _ = torch.topk(retrieval_sim, k=min(5, retrieval_sim.size(-1)), dim=-1)
            retrieval_confidence = top_k_values.mean(dim=-1).mean(dim=0)  # (B,) - 各batch的检索置信度

            # 将置信度映射到[0,1]区间：高置信度->低alpha（更依赖检索），低置信度->高alpha（更依赖参数化）
            alpha = torch.sigmoid(-retrieval_confidence / self.temperature + self.base_alpha)
        else:
            # 方法2：使用固定的基础权重
            alpha = torch.sigmoid(self.base_alpha).expand(B)

        # 确保alpha在合理范围内
        alpha = torch.clamp(alpha, 0.1, 0.9)

        # 线性融合：y_hat = alpha * y_param + (1 - alpha) * y_retrieval
        # alpha的物理含义：参数化预测的权重，当检索不可靠时增加参数化预测的比重
        fused_pred = alpha.unsqueeze(-1).unsqueeze(-1) * param_pred + \
                    (1 - alpha).unsqueeze(-1).unsqueeze(-1) * retrieval_pred

        return fused_pred, alpha


# ==========================================
# 缺失模块 4：趋势预测分支（慢变量）
# ==========================================
# 角色：在整体模型中负责对趋势成分（系统慢变量）进行独立预测，
#       与季节分支完全解耦，提供长期趋势外推能力
class TrendPredictor(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(TrendPredictor, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        # 趋势预测器：将历史趋势序列映射到未来趋势预测
        # 趋势作为慢变量，通常具有较强的线性特性
        self.trend_predictor = nn.Linear(seq_len * channels, pred_len * channels)

    def forward(self, trend_history):
        """
        对趋势成分进行线性外推预测

        Args:
            trend_history: (B, L, C) - 历史趋势序列

        Returns:
            trend_pred: (B, pred_len, C) - 趋势预测结果
        """
        B, L, C = trend_history.shape
        assert L == self.seq_len

        # 展平输入 (B, L*C)
        trend_input = trend_history.reshape(B, -1)

        # 线性预测 (B, pred_len*C)
        trend_output = self.trend_predictor(trend_input)

        # 重塑为预测形状 (B, pred_len, C)
        trend_pred = trend_output.reshape(B, self.pred_len, C)

        return trend_pred