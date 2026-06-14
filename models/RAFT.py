import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Retrieval import RetrievalTool, series_decomp


def compute_time_series_stats(tensor, dim=-1, keepdim=False):
    mean_val = tensor.mean(dim=dim, keepdim=keepdim)
    std_val = tensor.std(dim=dim, keepdim=keepdim)
    min_val = torch.amin(tensor, dim=dim, keepdim=keepdim)
    max_val = torch.amax(tensor, dim=dim, keepdim=keepdim)

    result = torch.cat([mean_val, std_val, min_val, max_val], dim=-1)

    if not keepdim and isinstance(dim, (tuple, int)) and result.dim() == 1:
        result = result.view(-1, 4)

    return result


class Model(nn.Module):

    def __init__(self, configs, individual=False):
        super(Model, self).__init__()

        self.device = torch.device(f'cuda:{configs.gpu}')

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in

        self.linear_x = nn.Linear(self.seq_len, self.pred_len)

        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

        self.n_period = configs.n_period
        self.topm = configs.topm
        self.sim_type = getattr(configs, 'sim_type', 'pearson')

        self.rt = RetrievalTool(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            channels=self.channels,
            n_period=self.n_period,
            topm=self.topm,
            sim_type=self.sim_type,
        )

        self.encoder_seasonal = nn.Linear(self.pred_len, self.pred_len)
        self.encoder_trend = nn.Linear(self.pred_len, self.pred_len)
        self.encoder_original = nn.Linear(self.pred_len, self.pred_len)

        self.linear_pred = nn.Linear(4 * self.pred_len, self.pred_len)

        gate_input_dim = self.channels * (4 * self.pred_len + 20)
        gate_hidden_dim = max(256, self.channels * 4)

        self.gate_layer = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
        )
        self.gate_linear1 = nn.Linear(gate_input_dim, gate_hidden_dim)
        self.gate_linear2 = nn.Linear(gate_hidden_dim, 1)

        self.moe_d_model = max(64, self.channels * 8)
        self.encode_mlp = nn.Sequential(
            nn.Linear(self.channels, self.moe_d_model),
            nn.ReLU(),
            nn.Linear(self.moe_d_model, self.moe_d_model),
        )
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
        self.moe_decode = nn.Linear(self.moe_d_model, self.channels)

        self.channel_conv_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(2 * self.channels, self.channels, kernel_size=1),
                nn.BatchNorm1d(self.channels),
                nn.ReLU(),
                nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels),
                nn.BatchNorm1d(self.channels),
                nn.ReLU(),
            )
            for _ in range(3)
        ])

        self.channel_fusion_linear = nn.Linear(2 * self.channels, self.channels)

    def prepare_dataset(self, train_data, valid_data, test_data):
        self.rt.prepare_dataset(train_data)

        self.retrieval_dict = {}

        print('Doing Train Retrieval')
        train_rt = self.rt.retrieve_all(train_data, train=True, device=self.device)

        print('Doing Valid Retrieval')
        valid_rt = self.rt.retrieve_all(valid_data, train=False, device=self.device)

        print('Doing Test Retrieval')
        test_rt = self.rt.retrieve_all(test_data, train=False, device=self.device)

        del self.rt
        torch.cuda.empty_cache()

        self.retrieval_dict['train'] = train_rt.detach()
        self.retrieval_dict['valid'] = valid_rt.detach()
        self.retrieval_dict['test'] = test_rt.detach()

    def encoder(self, x, index, mode):
        index = index.to(self.device)

        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len)
        assert(channels == self.channels)

        x_offset = x.mean(dim=1, keepdim=True).detach()
        x_norm = x - x_offset

        x_pred_from_x = self.linear_x(x_norm.permute(0, 2, 1)).permute(0, 2, 1)

        index_cpu = index.cpu() if index.device != torch.device('cpu') else index

        retrieval_size = self.retrieval_dict[mode].size(1)
        index_cpu = torch.clamp(index_cpu, 0, retrieval_size - 1)

        pred_from_retrieval = self.retrieval_dict[mode][:, index_cpu]
        pred_from_retrieval = pred_from_retrieval.to(self.device)

        seasonal_retrieval = pred_from_retrieval[0]
        trend_retrieval = pred_from_retrieval[1]
        original_retrieval = pred_from_retrieval[2]

        seasonal_encoded = self.encoder_seasonal(seasonal_retrieval.permute(0, 2, 1)).permute(0, 2, 1)
        trend_encoded = self.encoder_trend(trend_retrieval.permute(0, 2, 1)).permute(0, 2, 1)
        original_encoded = self.encoder_original(original_retrieval.permute(0, 2, 1)).permute(0, 2, 1)

        retrieval_encoded = torch.cat([seasonal_encoded, trend_encoded, original_encoded], dim=1)

        concat_all = torch.cat([x_pred_from_x, retrieval_encoded], dim=1)

        fusion_method = 'concat'

        if fusion_method == 'concat':
            pred = torch.cat([x_pred_from_x, retrieval_encoded], dim=1)
            pred = self.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1).reshape(bsz, self.pred_len, self.channels)

        elif fusion_method == 'gate':
            pred_flat = x_pred_from_x.reshape(bsz, -1)
            retrieval_flat = retrieval_encoded.reshape(bsz, -1)

            def compute_channel_stats(tensor):
                mean_val = tensor.mean(dim=(1,), keepdim=False)
                std_val = tensor.std(dim=(1,), keepdim=False)
                min_val = torch.amin(tensor, dim=(1,))
                max_val = torch.amax(tensor, dim=(1,))
                return torch.cat([mean_val, std_val, min_val, max_val], dim=-1)

            stats_input = compute_channel_stats(x_norm)
            stats_x_pred = compute_channel_stats(x_pred_from_x)
            stats_seasonal = compute_channel_stats(seasonal_encoded)
            stats_trend = compute_channel_stats(trend_encoded)
            stats_original = compute_channel_stats(original_encoded)

            gate_input = torch.cat([
                pred_flat,
                retrieval_flat,
                stats_input,
                stats_x_pred,
                stats_seasonal,
                stats_trend,
                stats_original,
            ], dim=-1)

            gate = self.gate_layer(gate_input) + self.gate_linear1(gate_input)
            gate = torch.sigmoid(self.gate_linear2(gate))

            gate = gate.unsqueeze(1).expand(-1, self.pred_len, 1)

            retrieval_agg = retrieval_encoded.mean(dim=1, keepdim=True)

            pred = gate * retrieval_agg + (1.0 - gate) * x_pred_from_x

        elif fusion_method == 'moe':
            experts = [
                x_pred_from_x.unsqueeze(1),
                seasonal_encoded.unsqueeze(1),
                trend_encoded.unsqueeze(1),
                original_encoded.unsqueeze(1),
            ]
            all_experts = torch.cat(experts, dim=1)

            B, E, P, C = all_experts.shape

            all_flat = all_experts.reshape(-1, C)
            enc_flat = self.encode_mlp(all_flat)
            enc = enc_flat.reshape(B, E, P, self.moe_d_model)

            enc = enc.permute(0, 2, 1, 3).contiguous()
            enc_bp = enc.reshape(B * P, E, self.moe_d_model)

            att_out, _ = self.mha(enc_bp, enc_bp, enc_bp)

            att_out = att_out + self.ffn(att_out)

            gate_scores = self.moe_gate_layer(att_out)
            alpha = F.softmax(gate_scores, dim=1)

            fused = torch.sum(alpha * att_out, dim=1)
            fused = fused.reshape(B, P, self.moe_d_model)

            pred = self.moe_decode(fused)

        elif fusion_method == 'channel_conv':
            def apply_channel_conv(x_pred, retrieval_comp, fusion_module):
                concat = torch.cat([x_pred, retrieval_comp], dim=-1)
                concat_t = concat.permute(0, 2, 1)
                fused_t = fusion_module(concat_t)
                fused = fused_t.permute(0, 2, 1)
                return fused

            seasonal_fused = apply_channel_conv(x_pred_from_x, seasonal_encoded, self.channel_conv_fusion[0])
            trend_fused = apply_channel_conv(x_pred_from_x, trend_encoded, self.channel_conv_fusion[1])
            original_fused = apply_channel_conv(x_pred_from_x, original_encoded, self.channel_conv_fusion[2])

            decomposition_fused = seasonal_fused + trend_fused
            retrieval_fused = torch.stack([decomposition_fused, original_fused], dim=1).mean(dim=1)

            concat_final = torch.cat([x_pred_from_x, retrieval_fused], dim=-1)
            B, P, _ = concat_final.shape
            pred = self.channel_fusion_linear(concat_final.reshape(B * P, -1))
            pred = pred.reshape(B, P, -1)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        pred = pred + x_offset

        return pred

    def forecast(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def imputation(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def anomaly_detection(self, x_enc, index, mode):
        return self.encoder(x_enc, index, mode)

    def classification(self, x_enc, index, mode):
        enc_out = self.encoder(x_enc, index, mode)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, index, mode='train'):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, index, mode)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, index, mode)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc, index, mode)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, index, mode)
            return dec_out
        return None
