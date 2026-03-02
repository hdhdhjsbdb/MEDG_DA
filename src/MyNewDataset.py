from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from collections import Counter

def _to_tensor(x):
    # 兼容 transform 输出 numpy 或 torch
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif torch.is_tensor(x):
        return x.float()
    else:
        raise TypeError(f"transform output type not supported: {type(x)}")


class ConDataset(Dataset):
    """
    用于moco对比学习的Dataset
    """
    def __init__(self, x_path, y_path, info_path, transform_q=None, transform_k=None,
                 filter_domains=None):
        #以只读方式打开文件
        self.x = np.load(x_path, mmap_mode='r') #(num_samples,6,2048)
        self.y = np.load(y_path)  #(num_samples,)
        self.info = np.load(info_path) ##(num_samples,2)

        #创建掩码过滤
        mask = np.ones(len(self.y), dtype=bool)

        # ======= 改动：工况过滤改为元组列表 filter_domains =======
        # filter_domains: [(speed, load), ...]
        if filter_domains is not None:
            filter_domains = np.asarray(filter_domains, dtype=np.int64)  # (m,2)
            in_domain = np.zeros(len(self.y), dtype=bool)
            for sp, ld in filter_domains:
                in_domain |= ((self.info[:, 0] == sp) & (self.info[:, 1] == ld))
            mask &= in_domain
        # =========================================================

        #获取符合条件的索引
        self.indices = np.where(mask)[0]

        #两种增强
        self.transform_q = transform_q
        self.transform_k = transform_k
        print(f"Selected {len(self.indices)} samples out of {len(self.y)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.x[real_idx].copy()
        label = self.y[real_idx]
        x_q = self.transform_q(sample) if self.transform_q else sample
        x_k = self.transform_k(sample) if self.transform_k else sample
        #转换为torch张量
        x_q = torch.from_numpy(x_q).float()
        x_k = torch.from_numpy(x_k).float()

        return x_q, x_k , label

class ConDataset4New(Dataset):
    """
    用于 MoCo 域感知对比学习的 Dataset
    支持返回 (x_q, x_k, label, domain) 用于同域正样本、异域负样本的对比
    """
    def __init__(self, x_path, y_path, info_path, 
                 transform_q=None, transform_k=None,
                 filter_domains=None,
                 domain_mode='combined'):  # 新增参数
        """
        Args:
            domain_mode: 
                - 'combined': 将 speed 和 load 组合成唯一 domain ID (默认)
                - 'speed': 仅用转速作为 domain
                - 'load': 仅用负载作为 domain  
                - None: 不返回 domain 标签（兼容旧版）
        """
        # 以只读方式打开文件（适合大文件）
        self.x = np.load(x_path, mmap_mode='r')  # (num_samples, 6, 2048)
        self.y = np.load(y_path)                 # (num_samples,)
        self.info = np.load(info_path)           # (num_samples, 2) -> [speed, load]
        
        self.transform_q = transform_q
        self.transform_k = transform_k
        self.domain_mode = domain_mode

        # ====== 工况过滤（保持您的原有逻辑）======
        mask = np.ones(len(self.y), dtype=bool)
        
        if filter_domains is not None:
            filter_domains = np.asarray(filter_domains, dtype=np.int64)  # (m, 2)
            in_domain = np.zeros(len(self.y), dtype=bool)
            for sp, ld in filter_domains:
                in_domain |= ((self.info[:, 0] == sp) & (self.info[:, 1] == ld))
            mask &= in_domain
        
        # 获取符合条件的索引
        self.indices = np.where(mask)[0]
        
        # ====== 新增：生成 Domain 标签 ======
        if domain_mode is not None:
            self.domains = self._create_domain_ids()
        else:
            self.domains = None
            
        print(f"Selected {len(self.indices)} samples out of {len(self.y)}")
        if domain_mode is not None:
            unique_domains = np.unique(self.domains[self.indices])
            print(f"Domain mode: {domain_mode}, Unique domains: {len(unique_domains)} | {unique_domains}")

    def _create_domain_ids(self):
        """
        根据 domain_mode 生成域标签
        """
        if self.domain_mode == 'combined':
            # 将 (speed, load) 组合映射为唯一整数 ID
            # 例如: (1500, 0.5)->0, (1500, 1.0)->1, (3000, 0.5)->2
            info_filtered = self.info[self.indices] if len(self.indices) < len(self.info) else self.info
            unique_pairs = np.unique(self.info[:, :2], axis=0)  # 基于全部数据创建映射
            domain_map = {tuple(pair): idx for idx, pair in enumerate(unique_pairs)}
            return np.array([domain_map[tuple(row[:2])] for row in self.info])
            
        elif self.domain_mode == 'speed':
            # 仅用转速作为域标签
            unique_speeds = np.unique(self.info[:, 0])
            speed_map = {val: idx for idx, val in enumerate(unique_speeds)}
            return np.array([speed_map[val] for val in self.info[:, 0]])
            
        elif self.domain_mode == 'load':
            # 仅用负载作为域标签
            unique_loads = np.unique(self.info[:, 1])
            load_map = {val: idx for idx, val in enumerate(unique_loads)}
            return np.array([load_map[val] for val in self.info[:, 1]])
        else:
            raise ValueError(f"Unknown domain_mode: {self.domain_mode}")

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # 加载原始样本
        sample = self.x[real_idx].copy()
        
        # 应用不同的数据增强（Query vs Key）
        x_q = self.transform_q(sample) if self.transform_q else sample
        x_k = self.transform_k(sample) if self.transform_k else sample
        
        # 转换为 Tensor
        x_q = torch.from_numpy(x_q).float()
        x_k = torch.from_numpy(x_k).float()
        
        # 故障标签（用于后续线性评估或监督微调）
        label = torch.tensor(self.y[real_idx], dtype=torch.long)
        
        if self.domain_mode is not None:
            # 返回域标签，用于 MoCo 的域感知对比
            domain = torch.tensor(self.domains[real_idx], dtype=torch.long)
            return x_q, x_k, label, domain
        else:
            # 兼容旧版接口
            return x_q, x_k, label

class NormalDataset(Dataset):
    """
    用于分类训练的Dataset（支持：过滤后按域均匀采样 + 返回领域索引）
    info[:,0]=speed, info[:,1]=load
    """
    def __init__(self, x_path, y_path, info_path,
                 transform=None,
                 filter_domains=None,
                 filter_classes=None,       # ✅ 新增：只保留这些故障类别
                 exclude_classes=None,      # ✅ 可选：排除这些故障类别
                 mmap_mode='r'):
        self.x = np.load(x_path, mmap_mode=mmap_mode)
        self.y = np.load(y_path, mmap_mode=mmap_mode)
        self.info = np.load(info_path, mmap_mode=mmap_mode)

        n = len(self.y)
        mask = np.ones(n, dtype=bool)

        # ===== ✅ 新增：故障类别过滤 =====
        if filter_classes is not None and len(filter_classes) > 0:
            fc = np.asarray(filter_classes, dtype=np.int64).reshape(-1)
            mask &= np.isin(self.y.astype(np.int64), fc)

        if exclude_classes is not None and len(exclude_classes) > 0:
            exc = np.asarray(exclude_classes, dtype=np.int64).reshape(-1)
            mask &= ~np.isin(self.y.astype(np.int64), exc)
        # =================================

        # ===== 快速域过滤：np.isin + structured view（避免 Python for）=====
        if filter_domains is not None and len(filter_domains) > 0:
            fd = np.asarray(filter_domains, dtype=np.int64).reshape(-1, 2)
            info2 = np.asarray(self.info[:, :2], dtype=np.int64)

            dt = np.dtype([('sp', np.int64), ('ld', np.int64)])
            info_view = info2.view(dt).reshape(-1)
            fd_view = fd.view(dt).reshape(-1)
            mask &= np.isin(info_view, fd_view)
        # ================================================================

        self.indices = np.where(mask)[0]
        self.transform = transform

        dom_arr = np.asarray(self.info[self.indices, :2], dtype=np.int64)  # (N,2)

        domains_unique, inv = np.unique(dom_arr, axis=0, return_inverse=True)
        self.domains = [tuple(map(int, d)) for d in domains_unique.tolist()]
        self.domain_to_id = {dom: i for i, dom in enumerate(self.domains)}
        self.id_to_domain = {i: dom for dom, i in self.domain_to_id.items()}

        self.sample_domain_id = inv.astype(np.int64)

        self.domain_id_to_pos = {}
        for did in range(len(self.domains)):
            pos = np.where(self.sample_domain_id == did)[0]
            if pos.size > 0:
                self.domain_id_to_pos[did] = pos

        print(f"Selected {len(self.indices)} samples out of {n}")
        print(f"Remaining domains: {len(self.domains)} (mapped to 0-{len(self.domains)-1})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.x[real_idx].copy()
        label = int(self.y[real_idx])

        # idx 是过滤后索引，直接查对应 domain_id
        dom_id = int(self.sample_domain_id[idx])

        x = self.transform(sample) if self.transform else sample
        x = torch.from_numpy(np.asarray(x)).float()
        y = torch.tensor(label, dtype=torch.long)
        d = torch.tensor(dom_id, dtype=torch.long)
        return x, y, d

    def _domain_to_id(self, domain):
        # 支持 tuple/list/np array
        dom = tuple(map(int, domain))
        try:
            return self.domain_to_id[dom]
        except KeyError:
            raise ValueError(f"Domain {dom} not in remaining domains.")

    def sample_domain_indices(self, domain, n, replace=True):
        """
        返回原始样本索引（ridx），用于直接索引 self.x/self.y
        """
        did = self._domain_to_id(domain)
        pos_pool = self.domain_id_to_pos.get(did, None)

        if pos_pool is None or pos_pool.size == 0:
            raise ValueError(
                f"No samples found for domain_id={did}, domain={self.id_to_domain[did]}"
            )

        chosen_pos = np.random.choice(pos_pool, size=n, replace=replace)  # positions in filtered set
        return self.indices[chosen_pos]  # map to original indices

    def get_uniform_domain_batch(self, domains, k_per_domain, replace=True):
        """
        返回: batch_x, batch_y (故障标签), batch_d (领域标签)
        """
        dids = [self._domain_to_id(dom) for dom in domains]

        all_real_indices = []
        all_domain_ids = []

        for did in dids:
            pos_pool = self.domain_id_to_pos.get(did, None)
            if pos_pool is None or pos_pool.size == 0:
                raise ValueError(
                    f"No samples found for domain_id={did}, domain={self.id_to_domain[did]}"
                )

            chosen_pos = np.random.choice(pos_pool, size=k_per_domain, replace=replace)
            all_real_indices.append(self.indices[chosen_pos])
            all_domain_ids.append(np.full(k_per_domain, did, dtype=np.int64))

        all_real_indices = np.concatenate(all_real_indices, axis=0)
        all_domain_ids = np.concatenate(all_domain_ids, axis=0)

        # 同步打乱
        perm = np.random.permutation(all_real_indices.shape[0])
        all_real_indices = all_real_indices[perm]
        all_domain_ids = all_domain_ids[perm]

        batch = self.x[all_real_indices].copy()
        labels = self.y[all_real_indices].astype(np.int64)

        if self.transform:
            batch_x = torch.stack([
                torch.from_numpy(np.asarray(self.transform(b))).float()
                for b in batch
            ])
        else:
            batch_x = torch.from_numpy(batch).float()

        batch_y = torch.from_numpy(labels).long()
        batch_d = torch.from_numpy(all_domain_ids).long()
        return batch_x, batch_y, batch_d

    def get_meta_batches(self, meta_train_domains, meta_test_domain, k, replace=True):
        """
        一次返回包含领域索引的元训练和元测试 Batch
        """
        x_tr, y_tr, d_tr = self.get_uniform_domain_batch(meta_train_domains, k_per_domain=k, replace=replace)
        x_te, y_te, d_te = self.get_uniform_domain_batch(meta_test_domain, k_per_domain=k, replace=replace)
        return (x_tr, y_tr, d_tr), (x_te, y_te, d_te)
    def apply_global_map(self, global_domain_to_id: dict):
        """
        把本 dataset 的局部 domain_id 映射到全局 domain_id，并重建采样池。
        global_domain_to_id: {(speed,load): gid, ...}
        """
        # local did -> global gid
        local_to_global = np.array([global_domain_to_id[dom] for dom in self.domains], dtype=np.int64)

        # 每个样本的 domain_id 改成全局
        self.sample_domain_id = local_to_global[self.sample_domain_id]

        # 更新映射表（_domain_to_id 会用到）
        self.domain_to_id = global_domain_to_id
        self.id_to_domain = {i: dom for dom, i in global_domain_to_id.items()}

        # 重建采样池：gid -> pos_in_filtered
        self.domain_id_to_pos = {}
        for gid in np.unique(self.sample_domain_id):
            pos = np.where(self.sample_domain_id == gid)[0]
            if pos.size > 0:
                self.domain_id_to_pos[int(gid)] = pos


class TargetDataset(Dataset):
    """
    无标签目标域 Dataset
    - 返回 (x, d) 其中 d 是 domain_id
    - info[:,0]=speed, info[:,1]=load
    """
    def __init__(self, x_path, info_path,
                 transform=None, filter_domains=None,
                 mmap_mode='r'):
        self.x = np.load(x_path, mmap_mode=mmap_mode)
        self.info = np.load(info_path, mmap_mode=mmap_mode)

        n = len(self.info)
        mask = np.ones(n, dtype=bool)

        # ===== 快速域过滤：np.isin + structured view =====
        if filter_domains is not None and len(filter_domains) > 0:
            fd = np.asarray(filter_domains, dtype=np.int64).reshape(-1, 2)
            info2 = np.asarray(self.info[:, :2], dtype=np.int64)

            dt = np.dtype([('sp', np.int64), ('ld', np.int64)])
            info_view = info2.view(dt).reshape(-1)
            fd_view = fd.view(dt).reshape(-1)
            mask &= np.isin(info_view, fd_view)
        # ===============================================

        self.indices = np.where(mask)[0]  #生成过滤后的样本索引
        self.transform = transform

        # 只取剩余样本的 domain (speed,load)
        dom_arr = np.asarray(self.info[self.indices, :2], dtype=np.int64)  # (N,2)

        # domains & 每个样本的局部 domain_id
        domains_unique, inv = np.unique(dom_arr, axis=0, return_inverse=True)
        self.domains = [tuple(map(int, d)) for d in domains_unique.tolist()]  # list of tuples
        self.domain_to_id = {dom: i for i, dom in enumerate(self.domains)}
        self.id_to_domain = {i: dom for dom, i in self.domain_to_id.items()}

        self.sample_domain_id = inv.astype(np.int64)  # (len(self.indices),)

        # domain_id -> pos_in_filtered
        self.domain_id_to_pos = {}
        for did in range(len(self.domains)):
            pos = np.where(self.sample_domain_id == did)[0]
            if pos.size > 0:
                self.domain_id_to_pos[did] = pos

        print(f"[Target] Selected {len(self.indices)} samples out of {n}")
        print(f"[Target] Remaining domains: {len(self.domains)} (mapped to 0-{len(self.domains)-1})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.x[real_idx].copy()

        dom_id = int(self.sample_domain_id[idx])

        x = self.transform(sample) if self.transform else sample
        x = torch.from_numpy(np.asarray(x)).float()
        d = torch.tensor(dom_id, dtype=torch.long)
        return x, d

    def _domain_to_id(self, domain):
        dom = tuple(map(int, domain))
        try:
            return self.domain_to_id[dom]
        except KeyError:
            raise ValueError(f"Domain {dom} not in remaining domains.")

    def get_uniform_domain_batch(self, domains, k_per_domain, replace=True):
        """
        返回: batch_x, batch_d
        """
        dids = [self._domain_to_id(dom) for dom in domains]

        all_real_indices = []
        all_domain_ids = []

        for did in dids:
            pos_pool = self.domain_id_to_pos.get(did, None)
            if pos_pool is None or pos_pool.size == 0:
                raise ValueError(
                    f"No samples found for domain_id={did}, domain={self.id_to_domain[did]}"
                )

            chosen_pos = np.random.choice(pos_pool, size=k_per_domain, replace=replace)
            all_real_indices.append(self.indices[chosen_pos])
            all_domain_ids.append(np.full(k_per_domain, did, dtype=np.int64))

        all_real_indices = np.concatenate(all_real_indices, axis=0)
        all_domain_ids = np.concatenate(all_domain_ids, axis=0)

        perm = np.random.permutation(all_real_indices.shape[0])
        all_real_indices = all_real_indices[perm]
        all_domain_ids = all_domain_ids[perm]

        batch = self.x[all_real_indices].copy()

        if self.transform:
            batch_x = torch.stack([
                torch.from_numpy(np.asarray(self.transform(b))).float()
                for b in batch
            ])
        else:
            batch_x = torch.from_numpy(batch).float()

        batch_d = torch.from_numpy(all_domain_ids).long()
        return batch_x, batch_d

    def apply_global_map(self, global_domain_to_id: dict):
        """
        把局部 domain_id 映射到全局 domain_id，并重建采样池
        """
        local_to_global = np.array([global_domain_to_id[dom] for dom in self.domains], dtype=np.int64)
        self.sample_domain_id = local_to_global[self.sample_domain_id]

        self.domain_to_id = global_domain_to_id
        self.id_to_domain = {i: dom for dom, i in global_domain_to_id.items()}

        self.domain_id_to_pos = {}
        for gid in np.unique(self.sample_domain_id):
            pos = np.where(self.sample_domain_id == gid)[0]
            if pos.size > 0:
                self.domain_id_to_pos[int(gid)] = pos
    def sample_batch(self, batch_size):
        """
        按 domain 均匀采样 target batch
        返回:
            x: Tensor [B, ...]
            d: Tensor [B]  (global domain_id)
        """
        # 所有可用 domain_id（已经是 global 的，如果 apply_global_map 过）
        domain_ids = list(self.domain_id_to_pos.keys())
        num_domains = len(domain_ids)

        if num_domains == 0:
            raise RuntimeError("No domains available in TargetDataset.")

        # 每个 domain 至少采多少
        per_domain = batch_size // num_domains
        remainder = batch_size % num_domains

        all_real_indices = []
        all_domain_ids = []

        # -------- 均匀采样 --------
        for did in domain_ids:
            pos_pool = self.domain_id_to_pos[did]

            # 样本不足则允许 replace
            replace = pos_pool.size < per_domain
            chosen_pos = np.random.choice(pos_pool, size=per_domain, replace=replace)

            all_real_indices.append(self.indices[chosen_pos])
            all_domain_ids.append(np.full(per_domain, did, dtype=np.int64))

        # -------- 处理余数 --------
        if remainder > 0:
            extra_dids = np.random.choice(domain_ids, size=remainder, replace=True)
            for did in extra_dids:
                pos_pool = self.domain_id_to_pos[did]
                chosen_pos = np.random.choice(pos_pool, size=1, replace=True)
                all_real_indices.append(self.indices[chosen_pos])
                all_domain_ids.append(np.array([did], dtype=np.int64))

        all_real_indices = np.concatenate(all_real_indices, axis=0)
        all_domain_ids = np.concatenate(all_domain_ids, axis=0)

        # -------- 同步打乱 --------
        perm = np.random.permutation(all_real_indices.shape[0])
        all_real_indices = all_real_indices[perm]
        all_domain_ids = all_domain_ids[perm]

        batch = self.x[all_real_indices].copy()

        if self.transform:
            batch_x = torch.stack([
                torch.from_numpy(np.asarray(self.transform(b))).float()
                for b in batch
            ])
        else:
            batch_x = torch.from_numpy(batch).float()

        batch_d = torch.from_numpy(all_domain_ids).long()
        return batch_x, batch_d


