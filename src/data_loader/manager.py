import numpy as np
from pytorch_lightning import LightningDataModule
from .templates import get_template, TrainDataset, TestDataset
from .processors import get_processor
from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from datasets import load_from_disk
from pathlib import Path
import os
import shutil
import time

class DataManager(LightningDataModule):
    def __init__(self, data_config, model_config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_config = data_config
        self.model_config = model_config
        # config
        self.data_name = getattr(data_config, 'dataset_name', 'tfns')
        self.data_path = getattr(data_config, 'data_path', "./data/tfns")
        self.model_type = model_config.model_type
        self.max_len = data_config.max_len
        # partition methods
        self._partition_methods = {
            'uniform': self._partition_uniform,
            'noniid': self._partition_noniid,
            'long_tail': self._partition_long_tail,
            "clustering": self._partition_clustering
        }
        self.cache_root = Path(__file__).resolve().parents[2] / "partition_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def _path_tag(path_value: str) -> str:
        raw = str(path_value or "")
        stem = Path(raw).name if raw else "unknown_path"
        safe_stem = stem.replace("/", "_").replace("\\", "_")
        return safe_stem

    def _build_partition_cache_dir(
        self,
        num_clients: int,
        partition_method: str,
    ) -> Path:
        dataset_tag = str(self.data_name).replace("/", "_")
        split_tag = str(getattr(self.data_config, "data_split", "all")).replace("/", "_")
        method_tag = str(partition_method).replace("/", "_")
        data_path_tag = self._path_tag(self.data_path)
        cache_dir = self.cache_root / dataset_tag / data_path_tag / split_tag / f"{method_tag}_n{int(num_clients)}"
        return cache_dir

    @staticmethod
    def _cache_complete(cache_dir: Path, num_clients: int) -> bool:
        if not (cache_dir / "_SUCCESS").exists():
            return False
        for client_id in range(num_clients):
            train_dir = cache_dir / f"client_{client_id}" / "train"
            test_dir = cache_dir / f"client_{client_id}" / "test"
            if not train_dir.exists():
                return False
            if not test_dir.exists():
                return False
            if not (train_dir / "state.json").exists():
                return False
            if not (test_dir / "state.json").exists():
                return False
        return True

    @staticmethod
    def _load_cached_partitions(cache_dir: Path, num_clients: int):
        train_parts, test_parts = [], []
        for client_id in range(num_clients):
            client_dir = cache_dir / f"client_{client_id}"
            train_parts.append(load_from_disk(str(client_dir / "train")))
            test_parts.append(load_from_disk(str(client_dir / "test")))
        return train_parts, test_parts

    def _save_cached_partitions(
        self,
        cache_dir: Path,
        num_clients: int,
        train_parts: List[Any],
        test_parts: List[Any],
    ) -> None:
        temp_cache_dir = cache_dir.parent / f"{cache_dir.name}.tmp_{os.getpid()}"
        if temp_cache_dir.exists():
            shutil.rmtree(temp_cache_dir)
        temp_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            for client_id in range(num_clients):
                client_dir = temp_cache_dir / f"client_{client_id}"
                client_dir.mkdir(parents=True, exist_ok=True)
                train_path = client_dir / "train"
                test_path = client_dir / "test"
                train_parts[client_id].save_to_disk(str(train_path))
                test_parts[client_id].save_to_disk(str(test_path))
            (temp_cache_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            temp_cache_dir.rename(cache_dir)
        except Exception:
            if temp_cache_dir.exists():
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
            raise

    @staticmethod
    def _rank_id() -> int:
        return next((int(v) for k in ("RANK", "LOCAL_RANK", "SLURM_PROCID") if (v := os.getenv(k)) and v.isdigit()), 0)

    def _wait_for_cache_ready(
        self,
        cache_dir: Path,
        num_clients: int,
        timeout_seconds: int = 600,
        poll_seconds: float = 1.0,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() <= deadline:
            if self._cache_complete(cache_dir, num_clients):
                return True
            time.sleep(poll_seconds)
        return False

    def _build_client_partitions(
        self,
        train_source: Any,
        test_source: Any,
        num_clients: int,
        partition_method: str,
    ):
        train_client_indices = self._partition_data(
            train_source,
            num_clients,
            partition_method,
            alpha=self.data_config.dirichlet_alpha
        )
        train_parts, test_parts = [], []
        for indices in train_client_indices:
            train_indices = indices.tolist() if isinstance(indices, np.ndarray) else indices
            train_subset = train_source.select(train_indices)
            test_subset = test_source
            train_parts.append(train_subset)
            test_parts.append(test_subset)
        return train_parts, test_parts
        
    def _partition_data(self, data: Any, num_clients: int, method: str = 'uniform', **kwargs) -> Union[List[np.ndarray], List[List[int]]]:
        if method not in self._partition_methods:
            raise ValueError(f"Unknown partition method: {method}. Available: {list(self._partition_methods.keys())}")
        return self._partition_methods[method](data, num_clients, **kwargs)

    def _partition_uniform(self, data, num_clients, **kwargs) -> List[List[int]]:
        samples_per_client = len(data) // num_clients
        client_indices = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < num_clients - 1 else len(data)
            client_indices.append(list(range(start, end)))
        return client_indices

    @staticmethod
    def _has_column(data, column_name: str) -> bool:
        if isinstance(data, dict):
            return column_name in data
        return column_name in getattr(data, "column_names", [])
        
    def _resolve_partition_label_key(self, data, label_key: str) -> str:
        if self._has_column(data, "task_type"):
            return "task_type"
        if self._has_column(data, label_key):
            return label_key
        if self._has_column(data, "label"):
            return "label"
        raise KeyError(f"Cannot find a valid partition label column from '{label_key}'.")

    def _partition_noniid(self, data, num_clients, alpha=0.5, label_key='answer', **kwargs) -> List[List[int]]:
        label_key = self._resolve_partition_label_key(data, label_key)
        labels = np.array(data[label_key])
        unique_labels = np.unique(labels)
        class_indices = {c: np.where(labels == c)[0] for c in unique_labels}
        client_indices = [[] for _ in range(num_clients)]
        for c, idxs in class_indices.items():
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            split_parts = np.split(idxs, split_points)
            for i, part in enumerate(split_parts):
                client_indices[i].extend(part.tolist())
        for i in range(num_clients):
            np.random.shuffle(client_indices[i])
        return client_indices

    def _partition_long_tail(self, data, num_clients, **kwargs) -> List[List[int]]:
        total = len(data)
        remaining = total
        current = 0
        client_indices = []
        for i in range(num_clients):
            take = max(1, remaining // 2) if i == 0 else max(1, remaining // (2 ** (i + 1)))
            if current + take > total: take = total - current
            client_indices.append(list(range(current, current + take)))
            current += take; remaining -= take
            if remaining <= 0 or current >= total: break
        if remaining > 0 and client_indices:
            client_indices[-1].extend(list(range(current, total)))
        while len(client_indices) < num_clients:
            client_indices.append([])
        return client_indices
    
    def _partition_clustering(self, data, num_clients, n_clusters=6, alpha=0.5, **kwargs) -> List[List[int]]:
        if self.data_name == "alpaca" and self._has_column(data, "task_type"):
            return self._partition_noniid(data, num_clients, alpha=alpha, label_key="task_type")
        else:
            text_key = "answer"
            if self._has_column(data, "question"):
                text_key = "question"
            texts = [item[text_key] for item in data]
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
            pseudo_labels = kmeans.fit_predict(tfidf_matrix)
            temp_data = {'pseudo_label': pseudo_labels}
            return self._partition_noniid(temp_data, num_clients, alpha=alpha, label_key='pseudo_label')

    # def _partition_noniid(self, data, num_clients, alpha=0.5, label_key='answer', **kwargs) -> List[List[int]]:
    #     labels = np.array(data[label_key])
    #     unique_labels = np.unique(labels)
    #     class_indices = {c: np.where(labels == c)[0] for c in unique_labels}
    #     client_indices = [[] for _ in range(num_clients)]
    #     for c, idxs in class_indices.items():
    #         np.random.shuffle(idxs)
    #         proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
    #         split_points = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    #         split_parts = np.split(idxs, split_points)
    #         for i, part in enumerate(split_parts):
    #             client_indices[i].extend(part.tolist())
    #     for i in range(num_clients):
    #         np.random.shuffle(client_indices[i])
    #     return client_indices

    # def _partition_long_tail(self, data, num_clients, **kwargs) -> List[List[int]]:
    #     total = len(data)
    #     remaining = total
    #     current = 0
    #     client_indices = []
    #     for i in range(num_clients):
    #         take = max(1, remaining // 2) if i == 0 else max(1, remaining // (2 ** (i + 1)))
    #         if current + take > total: take = total - current
    #         client_indices.append(list(range(current, current + take)))
    #         current += take; remaining -= take
    #         if remaining <= 0 or current >= total: break
    #     if remaining > 0 and client_indices:
    #         client_indices[-1].extend(list(range(current, total)))
    #     while len(client_indices) < num_clients:
    #         client_indices.append([])
    #     return client_indices
    
    # def _partition_clustering(self, data, num_clients, n_clusters=6, alpha=0.5, **kwargs) -> List[List[int]]:
    #     texts = [item["answer"] for item in data]
    #     vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    #     tfidf_matrix = vectorizer.fit_transform(texts)
    #     kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    #     pseudo_labels = kmeans.fit_predict(tfidf_matrix)
    #     temp_data = {'pseudo_label': pseudo_labels}
    #     return self._partition_noniid(temp_data, num_clients, alpha=alpha, label_key='pseudo_label')
    
    def _get_data_loader(self, num_clients=10, partition_method='uniform'):
        # Load and process data
        processor = get_processor(self.data_name, self.data_path, test_rate=getattr(self.data_config, 'test_rate', 0.2), 
                                  div_test_data_rate=getattr(self.data_config, 'div_test_data_rate', 1.0))
        (train_source, test_source), label_map = processor.load_and_process()
        # template  
        template = get_template(self.model_type)
        cache_dir = self._build_partition_cache_dir(
            num_clients=num_clients,
            partition_method=partition_method,
        )
        rank_id = self._rank_id()
        train_parts, test_parts = None, None
        if self._cache_complete(cache_dir, num_clients):
            try:
                train_parts, test_parts = self._load_cached_partitions(cache_dir, num_clients)
                print(f"[DataManager] Loaded cached client partitions: {cache_dir}")
            except Exception as exc:
                print(f"[DataManager] Failed to load cache ({cache_dir}), rebuilding. Error: {exc}")
                train_parts, test_parts = None, None

        if train_parts is None or test_parts is None:
            if rank_id == 0:
                train_parts, test_parts = self._build_client_partitions(
                    train_source=train_source,
                    test_source=test_source,
                    num_clients=num_clients,
                    partition_method=partition_method,
                )
                self._save_cached_partitions(cache_dir, num_clients, train_parts, test_parts)
                print(f"[DataManager] Saved client partitions to cache: {cache_dir}")
            else:
                cache_ready = self._wait_for_cache_ready(cache_dir, num_clients)
                if cache_ready:
                    try:
                        train_parts, test_parts = self._load_cached_partitions(cache_dir, num_clients)
                        print(f"[DataManager] Loaded cached client partitions after waiting: {cache_dir}")
                    except Exception as exc:
                        print(f"[DataManager] Failed to load ready cache ({cache_dir}), fallback to local partition. Error: {exc}")
            if train_parts is None or test_parts is None:
                train_parts, test_parts = self._build_client_partitions(
                    train_source=train_source,
                    test_source=test_source,
                    num_clients=num_clients,
                    partition_method=partition_method,
                )
                if rank_id != 0:
                    print(f"[DataManager] Fallback to rank-local partitions without cache write (rank={rank_id}).")
                    
        # partition data
        # train_client_indices = self._partition_data(train_source, num_clients, partition_method, alpha=self.data_config.dirichlet_alpha)
        # create data loaders
        # data_loaders = []
        # for _, indices in enumerate(train_client_indices):
        #     train_indices = indices.tolist() if isinstance(indices, np.ndarray) else indices
        #     train_ds = TrainDataset(
        #         data=train_source, 
        #         tokenizer=self.tokenizer,
        #         template=template,
        #         max_len=self.max_len,
        #         indices=train_indices,
        #         label_map=label_map
        #     )
        #     test_ds = TestDataset(
        #         data=test_source,
        #         tokenizer=self.tokenizer,
        #         template=template,
        #         max_len=self.max_len,
        #         label_map=label_map
        #     )
        #     data_loaders.append({
        #         'train': train_ds,
        #         'test': test_ds
        #     })
        # return data_loaders

        data_loaders = []
        for client_id in range(num_clients):
            train_ds = TrainDataset(
                data=train_parts[client_id],
                tokenizer=self.tokenizer,
                template=template,
                max_len=self.max_len,
                indices=None,
                label_map=label_map
            )
            test_ds = TestDataset(
                data=test_parts[client_id],
                tokenizer=self.tokenizer,
                template=template,
                max_len=self.max_len,
                label_map=label_map
            )
            data_loaders.append({
                'train': train_ds,
                'test': test_ds
            })
        return data_loaders