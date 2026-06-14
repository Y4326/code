import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

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
        sim_type='pearson',
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)

        self.temperature = temperature
        self.topm = topm

        self.with_dec = with_dec
        self.return_key = return_key

        self.sim_type = sim_type

        kernel_size = 25
        self.series_decomp = series_decomp(kernel_size)

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
        self.train_data_all_mg, _ = self.decompose_mg(self.train_data_all)

        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()
        self.y_data_all_mg, _ = self.decompose_mg(self.y_data_all)

        self.n_train = self.train_data_all.shape[0]

    def decompose_mg(self, data_all, remove_offset=True):
        data_all_copy = copy.deepcopy(data_all)
        seasonal, trend = self.series_decomp(data_all_copy)
        mg = torch.stack([seasonal, trend, data_all_copy], dim=0)

        if remove_offset:
            offset = []
            for i, data_p in enumerate(mg):
                cur_offset = data_p.mean(dim=1, keepdim=True)
                mg[i] = data_p - cur_offset
                offset.append(cur_offset)
            offset = torch.stack(offset, dim=0)
        else:
            offset = None

        return mg, offset

    def periodic_batch_corr(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

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

    def _pearson_similarity(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        bx_mean = key.mean(dim=2, keepdim=True)
        bx_std = key.std(dim=2, keepdim=True) + 1e-5
        bx = (key - bx_mean) / bx_std

        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)

            ax_mean = cur_data.mean(dim=2, keepdim=True)
            ax_std = cur_data.std(dim=2, keepdim=True) + 1e-5
            ax = (cur_data - ax_mean) / ax_std

            cur_sim = torch.bmm(F.normalize(bx, dim=2),
                               F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    def _cosine_similarity(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)

            cur_sim = torch.bmm(F.normalize(key, dim=2),
                               F.normalize(cur_data, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    def _neg_l2_similarity(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)

            key_expanded = key.unsqueeze(3)
            data_expanded = cur_data.unsqueeze(2)

            diff = key_expanded - data_expanded.permute(0, 3, 2, 1)
            l2_dist = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-5)

            cur_sim = -l2_dist
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    def _dtw_similarity(self, data_all, key, in_bsz=64):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        seq_len = features // self.channels if self.channels > 0 else features

        iters = math.ceil(train_len / in_bsz)
        sim = []

        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)

            cur_data_reshaped = cur_data.reshape(cur_data.shape[0], cur_data.shape[1], seq_len, self.channels)
            key_reshaped = key.reshape(key.shape[0], key.shape[1], seq_len, self.channels)

            cur_sim = self._compute_dtw_vectorized(key_reshaped, cur_data_reshaped)

            sim.append(-cur_sim)

        sim = torch.cat(sim, dim=2)
        return sim

    def _compute_dtw_vectorized(self, key_reshaped, cur_data_reshaped):
        G, B, S, C = key_reshaped.shape
        T = cur_data_reshaped.shape[1]

        key_exp = key_reshaped.unsqueeze(2)
        cur_exp = cur_data_reshaped.unsqueeze(1)

        dist = torch.sqrt(torch.sum((key_exp - cur_exp) ** 2, dim=-1) + 1e-5)

        dtw_distances = torch.zeros(G, B, T, S, device=key_reshaped.device)

        dtw_distances[:, :, :, 0] = dist[:, :, :, 0]
        dtw_distances[:, :, 0, :] = dist[:, :, 0, :]

        for s in range(1, S):
            for t in range(1, T):
                dtw_distances[:, :, t, s] = dist[:, :, t, s] + torch.minimum(
                    torch.minimum(
                        dtw_distances[:, :, t-1, s],
                        dtw_distances[:, :, t, s-1]
                    ),
                    dtw_distances[:, :, t-1, s-1]
                )

        return dtw_distances[:, :, :, -1]

    def retrieve(self, x, index, train=True):
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape
        assert(seq_len == self.seq_len)
        assert(channels == self.channels)

        x_mg, mg_offset = self.decompose_mg(x)
        num_components = x_mg.shape[0]

        sim = self.periodic_batch_corr(
            self.train_data_all_mg.flatten(start_dim=2),
            x_mg.flatten(start_dim=2),
        )

        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(num_components, 1, 1)

            sim = sim.masked_fill_(self_mask.bool(), float('-inf'))

        sim = sim.reshape(num_components * bsz, self.n_train)

        topm_index = torch.topk(sim, self.topm, dim=1).indices

        ranking_sim = torch.ones_like(sim) * float('-inf')

        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)

        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        num_components = x_mg.shape[0]
        sim = sim.reshape(num_components, bsz, self.n_train)
        ranking_sim = ranking_sim.reshape(num_components, bsz, self.n_train)

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2).detach().cpu()

        y_data_all = self.y_data_all_mg

        preds_per_comp = []
        for comp in range(num_components):
            comp_y = y_data_all[comp].reshape(self.n_train, -1)
            pred_comp = torch.matmul(ranking_prob[comp], comp_y)
            preds_per_comp.append(pred_comp.unsqueeze(0))

        pred_from_retrieval = torch.cat(preds_per_comp, dim=0).reshape(num_components, bsz, self.pred_len, channels)
        pred_from_retrieval = pred_from_retrieval.to(x.device)

        return pred_from_retrieval

    def retrieve_all(self, data, train=False, device=torch.device('cpu')):
        assert(self.train_data_all_mg != None)

        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

        retrievals = []

        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                pred_from_retrieval = self.retrieve(batch_x.float().to(device), index, train=train)
                pred_from_retrieval = pred_from_retrieval.cpu()
                retrievals.append(pred_from_retrieval)

        retrievals = torch.cat(retrievals, dim=1)

        return retrievals
