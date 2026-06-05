# 导入必要的PyTorch库
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入NumPy用于数组操作
import numpy as np
# 导入copy用于深拷贝
import copy
# 导入math用于数学运算
import math
# 导入tqdm用于进度条显示
from tqdm import tqdm

# 导入PyTorch数据加载工具
from torch.utils.data import Dataset, DataLoader


# 移动平均模块（来自DLinear）
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


# 系列分解模块（来自DLinear）
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# 检索工具类：实现基于多周期分解的时序数据检索
class RetrievalTool():
    def __init__(
        self,
        seq_len,        # 输入序列长度
        pred_len,       # 预测序列长度
        channels,       # 通道数（特征维度）
        n_period=3,     # 使用的周期数（默认改为 1：仅单周期检索）
        temperature=0.1,# softmax温度参数
        topm=20,        # 检索的top-k相似样本数
        with_dec=False, # 是否使用分解（预留参数）
        return_key=False,# 是否返回键（预留参数）
        sim_type='pearson',  # 相似度类型: 'pearson', 'cosine', 'neg_l2', 'dtw'
    ):
        # 定义周期数列表，从大到小排列（16, 8, 4, 2, 1）
        period_num = [16, 8, 4, 2, 1]
        # 选择最后n_period个周期（例如n_period=3时选择[4, 2, 1]）
        period_num = period_num[-1 * n_period:]

        # 保存输入参数
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        # 对周期数进行排序，从大到小（确保分解顺序）
        self.period_num = sorted(period_num, reverse=True)

        # 保存检索参数
        self.temperature = temperature
        self.topm = topm

        # 保存预留参数（当前版本中未使用）
        self.with_dec = with_dec
        self.return_key = return_key
        
        # 保存相似度类型参数
        self.sim_type = sim_type

        # 初始化季节-趋势分解模块
        kernel_size = 25
        self.series_decomp = series_decomp(kernel_size)
        
    # 准备数据集：构建训练数据的检索库
    def prepare_dataset(self, train_data):
        # 初始化存储训练输入和目标数据的列表
        train_data_all = []
        y_data_all = []

        # 遍历训练数据，提取输入序列和对应的目标序列
        for i in range(len(train_data)):
            td = train_data[i]
            # td[1] 是输入序列（历史数据）
            train_data_all.append(td[1])

            # 根据是否使用分解，选择不同的目标序列切片方式
            if self.with_dec:
                # 如果使用分解，包含更多的历史信息（当前版本中未使用此分支）
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                # 否则只取预测长度的目标序列
                y_data_all.append(td[2][-train_data.pred_len:])

        # 将训练输入数据转换为PyTorch张量，形状为 [样本数, 序列长度, 通道数]
        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()
        # 对训练输入数据应用多周期分解，构建多粒度检索库
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)

        # 将目标数据转换为PyTorch张量，形状为 [样本数, 序列长度, 通道数]
        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        # 对目标数据应用多周期分解，构建多粒度检索库
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        # 保存训练样本的数量
        self.n_train = self.train_data_all.shape[0]

    # 季节-趋势分解函数：使用DLinear的移动平均将时序数据分解为季节和趋势两部分
    # 输入data_all: 时序数据，形状为[T, S, C]，其中T是样本数，S是序列长度，C是通道数
    # 输出: [3, T, S, C]，其中第一个维度是[季节性, 趋势性, 原始数据]
    def decompose_mg(self, data_all, remove_offset=True):
        data_all_copy = copy.deepcopy(data_all)  # T, S, C   
        # 使用DLinear的移动平均方法进行季节-趋势分解
        seasonal, trend = self.series_decomp(data_all_copy)

        # 将季节性、趋势性和原始数据堆叠在一起，形成[3, T, S, C]的输出
        mg = torch.stack([seasonal, trend, data_all_copy], dim=0)  # [3, T, S, C]

        # mg = mg.append(data_all)  #mg形状为[3, T, S, C]
        # 如果需要移除偏移（中心化）
        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                # 计算每个成分在时间维度上的均值（偏移）
                cur_offset = data_p.mean(dim=1, keepdim=True)
                # 从成分中减去偏移，进行中心化
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
            offset = torch.stack(offset, dim=0)
        else:
            offset = None

        # 返回季节-趋势分解结果和对应的偏移
        return mg, offset
    
    # 批处理周期相关性计算：计算查询序列与训练库之间的相似度
    def periodic_batch_corr(self, data_all, key, in_bsz=64):
        # 获取输入张量的形状信息
        _, bsz, features = key.shape  # key形状: [G, B, S*C]，G是周期数，B是批次大小
        _, train_len, _ = data_all.shape  # data_all形状: [G, T, S*C]，T是训练样本数

        # 根据相似度类型选择计算方法
        if self.sim_type == 'pearson':
            return self._pearson_similarity(data_all, key, in_bsz)
        elif self.sim_type == 'cosine':
            return self._cosine_similarity(data_all, key, in_bsz)
        elif self.sim_type == 'neg_l2':
            return self._neg_l2_similarity(data_all, key, in_bsz)
        elif self.sim_type == 'dtw':
            return self._dtw_similarity(data_all, key, in_bsz)
        else:
            raise ValueError(f"Unknown similarity type: {self.sim_type}")

    # 皮尔逊相关系数
    def _pearson_similarity(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        # 对查询数据进行标准化（零中心化和单位方差化，实现Pearson相关系数）
        bx_mean = key.mean(dim=2, keepdim=True)
        bx_std = key.std(dim=2, keepdim=True) + 1e-5
        bx = (key - bx_mean) / bx_std

        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)

            # 对当前批次的训练数据进行标准化
            ax_mean = cur_data.mean(dim=2, keepdim=True)
            ax_std = cur_data.std(dim=2, keepdim=True) + 1e-5
            ax = (cur_data - ax_mean) / ax_std

            # 计算Pearson相关系数（标准化后的余弦相似度）
            cur_sim = torch.bmm(F.normalize(bx, dim=2),
                               F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    # 余弦相似度
    def _cosine_similarity(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)

            # 直接使用L2标准化计算余弦相似度（不进行中心化）
            cur_sim = torch.bmm(F.normalize(key, dim=2),
                               F.normalize(cur_data, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    # 负L2距离（距离越小越相似，所以取负值）
    def _neg_l2_similarity(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)

            # 计算L2距离: ||key - cur_data||
            # key: [G, B, S*C], cur_data: [G, T', S*C]
            # 需要扩展维度来计算每对查询-训练样本的距离
            key_expanded = key.unsqueeze(3)  # [G, B, S*C, 1]
            data_expanded = cur_data.unsqueeze(2)  # [G, T', S*C, 1] -> [G, 1, S*C, T'] after transpose
            
            # 计算欧氏距离
            diff = key_expanded - data_expanded.permute(0, 3, 2, 1)  # [G, B, S*C, T']
            l2_dist = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-5)  # [G, B, T']
            
            # 取负值使得距离越小相似度越高
            cur_sim = -l2_dist
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    # DTW（动态时间规整）相似度
    def _dtw_similarity(self, data_all, key, in_bsz=64):
        """
        使用DTW计算相似度（优化版，使用向量化操作）。
        注意：DTW计算复杂度较高，这里提供一个基于torch的批量化实现。
        对于大规模数据，建议使用dtw库或预先计算好的DTW矩阵。
        """
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        # 获取序列长度（展平后的维度是 S*C，我们需要还原为 S 和 C）
        seq_len = features // self.channels if self.channels > 0 else features
        
        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)  # [G, T', S*C]
            
            # 重塑为 [G, T', S, C] 和 [G, B, S, C]
            cur_data_reshaped = cur_data.reshape(cur_data.shape[0], cur_data.shape[1], seq_len, self.channels)
            key_reshaped = key.reshape(key.shape[0], key.shape[1], seq_len, self.channels)
            
            # 使用优化的向量化DTW计算
            cur_sim = self._compute_dtw_vectorized(key_reshaped, cur_data_reshaped)
            
            # 取负值使得DTW距离越小相似度越高
            sim.append(-cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    # 优化的向量化DTW计算
    def _compute_dtw_vectorized(self, key_reshaped, cur_data_reshaped):
        """
        批量计算DTW距离（向量化版本）。
        key_reshaped: [G, B, S, C]
        cur_data_reshaped: [G, T', S, C]
        返回: [G, B, T']
        """
        G, B, S, C = key_reshaped.shape
        T = cur_data_reshaped.shape[1]
        
        # 计算所有序列对的逐点距离矩阵
        # 扩展维度以便广播: key [G, B, S, 1, C], cur_data [G, 1, T, S, C] -> [G, B, T, S, C]
        key_exp = key_reshaped.unsqueeze(2)  # [G, B, 1, S, C]
        cur_exp = cur_data_reshaped.unsqueeze(1)  # [G, 1, T, S, C]
        
        # 计算欧氏距离: [G, B, T, S]
        dist = torch.sqrt(torch.sum((key_exp - cur_exp) ** 2, dim=-1) + 1e-5)
        
        # 初始化累积cost矩阵
        # 使用triu创建上三角矩阵的掩码
        dtw_distances = torch.zeros(G, B, T, S, device=key_reshaped.device)
        
        # 填充初始值
        dtw_distances[:, :, :, 0] = dist[:, :, :, 0]
        dtw_distances[:, :, 0, :] = dist[:, :, 0, :]
        
        # 动态规划（部分向量化）
        for s in range(1, S):
            for t in range(1, T):
                dtw_distances[:, :, t, s] = dist[:, :, t, s] + torch.minimum(
                    torch.minimum(
                        dtw_distances[:, :, t-1, s],      # 插入
                        dtw_distances[:, :, t, s-1]       # 删除
                    ),
                    dtw_distances[:, :, t-1, s-1]        # 匹配
                )
        
        # 返回最终的DTW距离
        return dtw_distances[:, :, :, -1]  # [G, B, T']
        
    # 检索方法：基于多周期分解的时序数据检索
    def retrieve(self, x, index, train=True):
        # 将索引张量移动到与输入数据相同的设备上
        index = index.to(x.device)

        # 获取输入数据的形状信息并进行有效性检查
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len, channels == self.channels)

        # 对输入数据应用季节-趋势分解
        x_mg, mg_offset = self.decompose_mg(x)  # x_mg形状: [3, B, S, C] (季节性 + 趋势性 + 原始数据)
        num_components = x_mg.shape[0]  # 成分数量：3

        # 计算输入的季节-趋势表示与训练库之间的相似度
        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2),  # [3, T, S*C] 展平后的训练库
            x_mg.flatten(start_dim=2),  # [3, B, S*C] 展平后的查询数据
        )  # 相似度矩阵形状: [3, B, T]

        # 如果是训练模式，需要创建自相似性遮罩以避免数据泄露
        if train:
            # 创建滑动窗口索引，用于标识可能存在数据泄露的时间窗口
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

            # 确保索引在有效范围内
            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            # 创建自相似性遮罩，将可能泄露的样本位置设为True
            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            # 现在有3个成分：季节性、趋势性和原始数据
            self_mask = self_mask.unsqueeze(dim=0).repeat(num_components, 1, 1)

            # 对相似度应用遮罩，将泄露样本的相似度设为负无穷
            sim = sim.masked_fill_(self_mask.bool(), float('-inf'))

        # 重塑张量形状以便进行top-k选择
        sim = sim.reshape(num_components * bsz, self.n_train)  # [3*B, T] (季节性*B + 趋势性*B + 原始*B)

        # 为每个查询样本选择top-m最相似的训练样本
        topm_index = torch.topk(sim, self.topm, dim=1).indices


        
        # 创建概率矩阵：只有top-K位置有概率，其他为0
        ranking_sim = torch.ones_like(sim) * float('-inf')  # [G*B, T]，G是周期数,B是批次大小,T是训练样本数


        # 创建行索引，用于scatter操作，scatter操作是将一个张量的值根据索引分散到另一个张量中
        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)  # [G*B, 1]，G是周期数,B是批次大小
        
        # 只保留top-m样本的相似度分数，其他设为负无穷
        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        # 恢复原始形状：使用实际的分解成分数（seasonal/trend），而不是 self.n_period
        num_components = x_mg.shape[0]  # 通常为3（季节性 + 趋势性 + 原始数据）
        sim = sim.reshape(num_components, bsz, self.n_train)        # [G, B, T]
        ranking_sim = ranking_sim.reshape(num_components, bsz, self.n_train)  # [G, B, T]

        # 对排名相似度应用softmax，得到概率分布（在 CPU 上）
        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2).detach().cpu()  # [G, B, T]

        # 展平目标数据，用于按成分做加权求和；self.y_data_all_mg 形状为 [G, T, P, C]
        # 我们把每个成分展平为 [T, P*C] 以便与 ranking_prob 的 [B, T] 相乘
        y_data_all = self.y_data_all_mg  # [G, T, P, C]（在 CPU 上）

        # 按成分计算加权和，避免一次性错误的 bmm 维度匹配
        preds_per_comp = []
        for comp in range(num_components):
            # ranking_prob[comp]: [B, T], y_data_all[comp]: [T, P, C]
            comp_y = y_data_all[comp].reshape(self.n_train, -1)  # [T, P*C]
            pred_comp = torch.matmul(ranking_prob[comp], comp_y)  # [B, P*C]
            preds_per_comp.append(pred_comp.unsqueeze(0))  # [1, B, P*C]

        # 拼接成 [G, B, P*C]，再重塑为 [G, B, P, C]
        pred_from_retrieval = torch.cat(preds_per_comp, dim=0).reshape(num_components, bsz, self.pred_len, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)
                


        return pred_from_retrieval  #形状为
    
    # 批量检索方法：对整个数据集进行检索处理
    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        # 确保检索库已经准备好（多周期分解结果存在）
        assert(self.train_data_all_mg != None)

        # 创建数据加载器，用于批量处理数据
        rt_loader = DataLoader(
            data,
            batch_size=1024,     # 批次大小
            shuffle=False,       # 不打乱顺序
            num_workers=8,       # 使用8个子进程加载数据
            drop_last=False      # 不丢弃最后不完整的批次
        )

        # 初始化存储所有批次检索结果的列表
        retrievals = []

        # 在不计算梯度的模式下进行推理
        with torch.no_grad():
            # 使用tqdm显示进度条，遍历所有批次
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                # 对当前批次的输入数据进行检索
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                # 将结果移回CPU并保存
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)

        # 将所有批次的检索结果拼接起来
        # dim=1表示在样本维度上拼接，最终形状为[G, N, P, C]，其中N是总样本数
        retrievals = torch.cat(retrievals, dim=1)

        return retrievals
