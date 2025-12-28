import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import (
    RetrievalTool,
    SeasonalPredictor,
    MultiScaleFusion,
    ParamRetrievalFusion,
    TrendPredictor,
    trend_season_decompose
)
from layers.Retrieval import trend_season_decompose
class Model(nn.Module):


    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Model, self).__init__()
        self.device = torch.device(f'cuda:{configs.gpu}')
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        # Series decomposition block from Autoformer
#         self.decompsition = series_decomp(configs.moving_avg)
#         self.individual = individual
        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)
        
        self.n_period = configs.n_period
        self.topm = configs.topm
        
        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            device=self.device,
        )
        
        # 使用RetrievalTool中实际配置的周期
        self.period_num = self.rt.period_num  # 根据n_period参数确定的周期列表
        self.actual_n_period = len(self.period_num)  # 实际的周期数
        
        module_list = [
            nn.Linear(self.pred_len // g, self.pred_len)
            for g in self.period_num
        ]
        self.retrieval_pred = nn.ModuleList(module_list)
        self.linear_pred = nn.Linear(2 * self.pred_len, self.pred_len)

        # ==========================================
        # 核心预测模块：遵循动力系统建模哲学
        # ==========================================

        # 趋势预测分支：系统慢变量的线性外推
        self.trend_predictor = TrendPredictor(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels
        )

        # 季节预测分支：准周期吸引子的多尺度建模
        self.seasonal_predictor = SeasonalPredictor(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.actual_n_period  # 使用实际的周期数
        )

        # 多尺度季节预测融合：基于尺度一致性的预测级融合
        self.multiscale_fusion = MultiScaleFusion(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.actual_n_period  # 使用实际的周期数
        )

        # 参数化vs检索融合：平衡结构外推与历史约束
        self.param_retrieval_fusion = ParamRetrievalFusion(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels
        )

#         if self.task_name == 'classification':
#             self.projection = nn.Linear(
#                 configs.enc_in * configs.seq_len, configs.num_class)

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)
        
        self.retrieval_dict = {}
        
        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)
            
        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()
        
        # Note: self.rt is kept alive for encoder() method which uses self.rt.decompose_mg and self.rt.retrieve
        # If memory is an issue, consider refactoring encoder to use pre-computed results


    '''
    动力系统建模：趋势(慢变量) + 季节(准周期吸引子) + 检索(历史演化约束)
    输入x → 分解 → 参数化外推 + 历史约束 → 融合 → 最终预测
    '''
    def encoder(self, x, index, mode):
        '''
        动力系统视角的预测流程：
        1. 趋势分支：系统慢变量的线性外推
        2. 季节分支：准周期吸引子的多尺度建模与融合
        3. 检索分支：历史真实演化的相似性约束
        4. 融合：参数化预测与检索预测的互补平衡

        输入:
            x: (B, L, C) 历史输入序列
            index: (B,) 样本在数据集中的全局编号
            mode: str 训练/验证/测试模式
        输出:
            pred: (B, P, C) 未来P步预测
        '''
        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len and channels == self.channels)

        # 预处理：去除全局偏移，确保预测在残差空间进行
        x_offset = x.mean(dim=1, keepdim=True).detach()  # (B, 1, C)
        x_norm = x - x_offset  # (B, L, C)

        # ==========================================
        # 第一阶段：趋势-季节分解 (动力系统状态分离)
        # ==========================================
        x_trend, x_seasonal = trend_season_decompose(
            x_norm,
            kernel_size=max(24, self.seq_len//4),
            seq_len=self.seq_len
        )  # x_trend: (B, L, C), x_seasonal: (B, L, C)

        # ==========================================
        # 第二阶段：趋势分支 - 系统慢变量建模
        # ==========================================
        trend_pred = self.trend_predictor(x_trend)  # (B, P, C) - 趋势的线性外推

        # ==========================================
        # 第三阶段：季节分支 - 准周期吸引子建模
        # ==========================================

        # 3.1 多尺度季节分解：将季节成分分解为不同周期尺度的表示
        x_seasonal_mg, _ = self.rt.decompose_mg(x_seasonal)  # (G, B, L, C)

        # 3.2 各尺度独立参数化预测：每个周期尺度的动力外推
        seasonal_scales_pred = self.seasonal_predictor(x_seasonal_mg)  # (G, B, P, C)

        # 3.3 多尺度预测融合：基于尺度一致性权重进行预测级融合
        seasonal_pred = self.multiscale_fusion(seasonal_scales_pred)  # (B, P, C)

        # ==========================================
        # 第四阶段：检索分支 - 历史演化轨迹约束
        # ==========================================

        # 4.1 获取预计算的检索预测结果
        index_cpu = index.cpu()
        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu]  # (G, B, P, C)
        pred_from_retrieval = pred_from_retrieval.to(self.device)

        # 4.2 检索预测的周期压缩-恢复处理
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
            g = self.period_num[i]
            # 压缩：保留周期代表帧
            pr_compressed = pr.reshape(bsz, self.pred_len // g, g, channels)[:, :, 0, :]  # (B, P//g, C)
            # 恢复：线性插值回全长度
            pr_recovered = self.retrieval_pred[i](pr_compressed.permute(0, 2, 1)).permute(0, 2, 1)  # (B, P, C)
            pr_recovered = pr_recovered.reshape(bsz, self.pred_len, self.channels)
            retrieval_pred_list.append(pr_recovered)

        # 4.3 多周期检索预测融合：基于相似度隐含权重的硬融合
        retrieval_pred_list = torch.stack(retrieval_pred_list, dim=1)  # (B, G, P, C)
        retrieval_pred_fused = retrieval_pred_list.sum(dim=1)  # (B, P, C)

        # ==========================================
        # 第五阶段：参数化预测 vs 历史检索的融合
        # ==========================================

        # 5.1 计算检索相似度统计量（用于自适应融合权重）
        # 注：这里需要从检索过程中获取相似度信息，但在当前架构中，
        # retrieval_dict只存储预测结果，需要考虑如何获取相似度
        # 临时方案：使用None，让ParamRetrievalFusion使用默认权重
        retrieval_sim = None  # TODO: 需要修改以获取相似度矩阵

        # 5.2 参数化预测与检索预测的互补融合
        seasonal_final, fusion_weights = self.param_retrieval_fusion(
            param_pred=seasonal_pred,      # (B, P, C) 参数化季节预测
            retrieval_pred=retrieval_pred_fused,  # (B, P, C) 检索季节预测
            retrieval_sim=retrieval_sim
        )

        # ==========================================
        # 第六阶段：最终预测合成
        # ==========================================

        # 动力系统视角的最终预测：慢变量(趋势) + 准周期成分(季节)
        final_pred = trend_pred + seasonal_final  # (B, P, C)

        # 加回全局偏移：从残差空间映射回原始空间
        pred = torch.cat([final_pred, retrieval_pred_fused], dim=1) 
        pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)
        pred = pred + x_offset  # (B, P, C)

        # 返回最终预测结果
        return pred

    def forecast(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        # Encoder
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        # Encoder
        enc_out = self.encoder(x_enc, index, mode)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            if isinstance(dec_out, tuple):
                pred, retrieval_only = dec_out
                return pred[:, -self.pred_len:, :], retrieval_only[:, -self.pred_len:, :]
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out  # [B, N]
        return None
