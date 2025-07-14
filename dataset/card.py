import os
from os import path
import pandas as pd
import numpy as np
import math
import tqdm
import pickle
import logging
from random import shuffle
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader
import tqdm
import torch
from torch.utils.data.dataset import Dataset

from misc.utils import divide_chunks
from dataset.vocab import Vocabulary
import os
import pickle
import torch
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)
log = logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TransactionDataset(Dataset):
    def __init__(self,
                 mlm,
                 user_ids=None,
                 seq_len=10,
                 num_bins=10,
                 cached=True,
                 root="./data/card/",
                 fname="card_trans",
                 vocab_dir="checkpoints",
                 fextension="",
                 nrows=None,
                 flatten=False,
                 stride=5,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=False, 
                 vocab=None):

        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.fextension = f'_{fextension}' if fextension else ''
        self.cached = cached
        self.user_ids = user_ids
        self.return_labels = return_labels
        self.skip_user = skip_user

        self.mlm = mlm
        self.trans_stride = stride

        self.flatten = flatten

        self.vocab = Vocabulary(adap_thres) if vocab is None else vocab
        self.seq_len = seq_len
        self.encoder_fit = {}

        self.trans_table = None
        self.data = []
        self.labels = []
        self.indices = []
        self.window_label = []

        self.ncols = None
        self.num_bins = num_bins
        self.encode_data()
        if vocab is None:
            self.init_vocab()
        self.prepare_samples()
        if vocab is None:
            self.save_vocab(vocab_dir)

    def __getitem__(self, index):
        real_index = self.indices[index]
        if self.flatten:
            return_data = torch.tensor(self.data[real_index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.data[real_index], dtype=torch.long).reshape(self.seq_len, -1)

        if self.return_labels:
            return_data = (return_data, torch.tensor(self.labels[real_index], dtype=torch.long))

        return return_data

    def __len__(self):
        return len(self.data)

    def save_vocab(self, vocab_dir):
        file_name = path.join(vocab_dir, f'vocab{self.fextension}.nb')
        log.info(f"saving vocab at {file_name}")
        self.vocab.save_vocab(file_name)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        X_hm = X['Time'].str.split(':', expand=True)
        d = pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=X_hm[0], minute=X_hm[1])).astype(
            int)
        return pd.DataFrame(d)

    @staticmethod
    def amountEncoder(X):
        amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        return pd.DataFrame(amt)

    @staticmethod
    def fraudEncoder(X):
        fraud = (X == 'Yes').astype(int)
        return pd.DataFrame(fraud)

    @staticmethod
    def nanNone(X):
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        return X.where(pd.notnull(X), 0)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return quant_inputs

    def user_level_data(self):
        fname = path.join(self.root, f"preprocessed/{self.fname}.user{self.fextension}.pkl")
        trans_data, trans_labels = [], []

        if self.cached and path.isfile(fname):
            log.info(f"loading cached user level data from {fname}")
            cached_data = pickle.load(open(fname, "rb"))
            trans_data = cached_data["trans"]
            trans_labels = cached_data["labels"]
            columns_names = cached_data["columns"]

        else:
            unique_users = self.trans_table["User"].unique()
            columns_names = list(self.trans_table.columns)

            for user in tqdm.tqdm(unique_users):
                user_data = self.trans_table.loc[self.trans_table["User"] == user]
                user_trans, user_labels = [], []
                for idx, row in user_data.iterrows():
                    row = list(row)

                    # assumption that user is first field
                    skip_idx = 1 if self.skip_user else 0

                    user_trans.extend(row[skip_idx:-1])
                    user_labels.append(row[-1])

                trans_data.append(user_trans)
                trans_labels.append(user_labels)

            if self.skip_user:
                columns_names.remove("User")

            with open(fname, 'wb') as cache_file:
                pickle.dump({"trans": trans_data, "labels": trans_labels, "columns": columns_names}, cache_file)

        # convert to str
        return trans_data, trans_labels, columns_names

    def format_trans(self, trans_lst, column_names):
        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 2))  # 2 to ignore isFraud and SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            if self.mlm:  # and self.flatten:  # only add [SEP] for BERT + flatten scenario
                vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

    def prepare_samples(self):
        log.info("preparing user level data...")
        trans_data, trans_labels, columns_names = self.user_level_data()

        log.info("creating transaction samples with vocab")
        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)

            user_labels = trans_labels[user_idx]

            bos_token = self.vocab.get_id(self.vocab.bos_token, special_token=True)  # will be used for GPT2
            eos_token = self.vocab.get_id(self.vocab.eos_token, special_token=True)  # will be used for GPT2
            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening
                if not self.mlm and self.flatten:  # for GPT2, need to add [BOS] and [EOS] tokens
                    ids = [bos_token] + ids + [eos_token]
                self.data.append(ids)

            for jdx in range(0, len(user_labels) - self.seq_len + 1, self.trans_stride):
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)

                fraud = 0
                if len(np.nonzero(ids)[0]) > 0:
                    fraud = 1
                self.window_label.append(fraud)
                
        self.indices = list(range(len(self.labels)))
        assert len(self.data) == len(self.labels) == len(self.indices)

        '''
            ncols = total fields - 1 (special tokens) - 1 (label)
            if bert:
                ncols += 1 (for sep)
        '''
        self.ncols = len(self.vocab.field_keys) - 2 + (1 if self.mlm else 0)
        log.info(f"ncols: {self.ncols}")
        log.info(f"no of samples {len(self.data)}")

    def get_csv(self, fname):
        data = pd.read_csv(fname, nrows=self.nrows)
        if self.user_ids:
            log.info(f'Filtering data by user ids list: {self.user_ids}...')
            self.user_ids = map(int, self.user_ids)
            data = data[data['User'].isin(self.user_ids)]

        self.nrows = data.shape[0]
        log.info(f"read data : {data.shape}")
        return data

    def write_csv(self, data, fname):
        log.info(f"writing to file {fname}")
        data.to_csv(fname, index=False)

    def init_vocab(self):
        column_names = list(self.trans_table.columns)
        if self.skip_user:
            column_names.remove("User")

        self.vocab.set_field_keys(column_names)

        for column in column_names:
            unique_values = self.trans_table[column].value_counts(sort=True).to_dict()  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}")
        log.info(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            log.info(f"column : {column}, vocab size : {vocab_size}")

            if vocab_size > self.vocab.adap_thres:
                log.info(f"\tsetting {column} for adaptive softmax")
                self.vocab.adap_sm_cols.add(column)

    def encode_data(self):
        dirname = path.join(self.root, "preprocessed")
        fname = f'{self.fname}{self.fextension}.encoded.csv'
        data_file = path.join(self.root, f"{self.fname}.csv")

        if self.cached and path.isfile(path.join(dirname, fname)):
            log.info(f"cached encoded data is read from {fname}")
            self.trans_table = self.get_csv(path.join(dirname, fname))
            encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
            self.encoder_fit = pickle.load(open(encoder_fname, "rb"))
            return

        data = self.get_csv(data_file)
        log.info(f"{data_file} is read.")

        log.info("nan resolution.")
        data['Errors?'] = self.nanNone(data['Errors?'])
        data['Is Fraud?'] = self.fraudEncoder(data['Is Fraud?'])
        data['Zip'] = self.nanZero(data['Zip'])
        data['Merchant State'] = self.nanNone(data['Merchant State'])
        data['Use Chip'] = self.nanNone(data['Use Chip'])
        data['Amount'] = self.amountEncoder(data['Amount'])

        sub_columns = ['Errors?', 'MCC', 'Zip', 'Merchant State', 'Merchant City', 'Merchant Name', 'Use Chip']

        log.info("label-fit-transform.")
        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data

        log.info("timestamp fit transform")
        timestamp = self.timeEncoder(data[['Year', 'Month', 'Day', 'Time']])
        timestamp_fit, timestamp = self.label_fit_transform(timestamp, enc_type="time")
        self.encoder_fit['Timestamp'] = timestamp_fit
        data['Timestamp'] = timestamp

        log.info("timestamp quant transform")
        coldata = np.array(data['Timestamp'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        data['Timestamp'] = self._quantize(coldata, bin_edges)
        self.encoder_fit["Timestamp-Quant"] = [bin_edges, bin_centers, bin_widths]

        log.info("amount quant transform")
        coldata = np.array(data['Amount'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        data['Amount'] = self._quantize(coldata, bin_edges)
        self.encoder_fit["Amount-Quant"] = [bin_edges, bin_centers, bin_widths]

        columns_to_select = ['User',
                             'Card',
                             'Timestamp',
                             'Amount',
                             'Use Chip',
                             'Merchant Name',
                             'Merchant City',
                             'Merchant State',
                             'Zip',
                             'MCC',
                             'Errors?',
                             'Is Fraud?']

        self.trans_table = data[columns_to_select]

        log.info(f"writing cached csv to {path.join(dirname, fname)}")
        if not path.exists(dirname):
            os.mkdir(dirname)
        self.write_csv(self.trans_table, path.join(dirname, fname))

        encoder_fname = path.join(dirname, f'{self.fname}{self.fextension}.encoder_fit.pkl')
        log.info(f"writing cached encoder fit to {encoder_fname}")
        pickle.dump(self.encoder_fit, open(encoder_fname, "wb"))
    def resample_train(self, train_indices, test_indices, eval_indices=[]):

        train_real_indices = [self.indices[i] for i in train_indices]
        test_real_indices = [self.indices[i] for i in test_indices]
        eval_real_indices = [self.indices[i] for i in eval_indices]
        train_labels = [self.labels[i] for i in train_real_indices]
        test_labels = [self.labels[i] for i in test_real_indices]
        eval_labels = [self.labels[i] for i in eval_real_indices]

        assert len(train_real_indices)+len(test_real_indices)+len(eval_real_indices) == len(self.indices)          
        assert len(train_labels)+len(test_labels)+len(eval_labels) == len(self.indices)          
        logger.info('Upsample training fraudulent samples.')
        train_real_indices = np.array(train_real_indices)
        train_labels = np.array(train_labels)
        logger.info(f'train labels shape: {train_labels.shape}')
        logger.info(f'train real indices shape: {train_real_indices.shape}')
        non_fraud_real_indices = train_real_indices[np.all(train_labels==0, axis=1)]
        non_fraud_labels = train_labels[np.all(train_labels==0, axis=1)]
        logger.info(f'non fraud indices shape: {non_fraud_real_indices.shape}')
        logger.info(f'non fraud labels shape: {non_fraud_labels.shape}')
        fraud_real_indices = train_real_indices[np.any(train_labels, axis=1)]
        fraud_labels = train_labels[np.any(train_labels, axis=1)]
        logger.info(f'fraud indices shape: {fraud_real_indices.shape}')
        logger.info(f'fraud labels shape: {fraud_labels.shape}')
        fraud_upsample_real_indices = resample(fraud_real_indices, replace=True, n_samples=non_fraud_labels.shape[0], random_state=2022)
        logger.info(f'fraud upsample indices shape: {fraud_upsample_real_indices.shape}')
        train_real_indices = np.concatenate((fraud_upsample_real_indices,non_fraud_real_indices))
        logger.info(f'new train indices shape: {train_real_indices.shape}')
        self.indices = list(train_real_indices)+eval_real_indices+test_real_indices
        train_indices = list(np.arange(train_real_indices.shape[0]))
        eval_indices = list(np.arange(train_real_indices.shape[0],train_real_indices.shape[0]+len(eval_real_indices)))
        test_indices = list(np.arange(train_real_indices.shape[0]+len(eval_real_indices),train_real_indices.shape[0]+len(eval_real_indices)+len(test_real_indices)))
        shuffle(train_indices)
        shuffle(eval_indices)
        shuffle(test_indices)
        logger.info(f'labels shape: {np.array(self.labels).shape}')
        logger.info(f'data shape: {np.array(self.data).shape}')
        logger.info(f'indices shape: {np.array(self.indices).shape}')
        assert len(self.data) == len(self.labels), f'data {len(self.data)} != labels {len(self.labels)}'
        assert len(self.indices) > len(self.labels), f'indices {len(self.indices)} <= labels {len(self.labels)}'
        return train_indices, test_indices, eval_indices


class TransactionDatasetEmbedded(TransactionDataset):
    def __init__(self,
                pretrained_model,              
                raw_dataset: TransactionDataset, 
                batch_size,
                cache_dir="./embeddings_cache",
                force_recompute=False):
        
        self.pretrained_model = pretrained_model.model.to(device)
        self.pretrained_model.eval()
        self.raw_dataset = raw_dataset
        assert self.raw_dataset.return_labels
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_recompute = force_recompute
        
        # Generate cache key based on dataset and model
        self.cache_key = self._generate_cache_key()
        self.cache_file = self.cache_dir / f"embeddings_{self.cache_key}.pkl"
        
        self.data = []
        self.labels = []
        self.indices = []
        
        # Try to load from cache first
        if self._load_embeddings():
            print("✓ Loaded embeddings from cache")
        else:
            print("✗ Cache miss - extracting embeddings")
            self.extract_embeddings(batch_size=batch_size)
            self._save_embeddings()
            print("✓ Saved embeddings to cache")

    def _generate_cache_key(self):
        """Generate a unique cache key based on dataset and model characteristics"""
        # Get dataset characteristics
        dataset_info = {
            'dataset_size': len(self.raw_dataset),
            'dataset_type': type(self.raw_dataset).__name__,
        }
        
        # Get model characteristics (simplified)
        model_info = {
            'model_type': type(self.pretrained_model).__name__,
            'config': str(self.pretrained_model.config) if hasattr(self.pretrained_model, 'config') else 'no_config',
        }
        
        # Combine info and hash
        cache_info = {**dataset_info, **model_info}
        cache_string = json.dumps(cache_info, sort_keys=True)
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()[:16]
        
        return cache_key

    def _save_embeddings(self):
        """Save embeddings, labels, and indices to cache"""
        try:
            cache_data = {
                'data': self.data,
                'labels': self.labels,
                'indices': self.indices,
                'metadata': {
                    'dataset_size': len(self.raw_dataset),
                    'embedding_count': len(self.data),
                    'cache_key': self.cache_key
                }
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Saved embeddings to: {self.cache_file}")
            print(f"Cache file size: {self.cache_file.stat().st_size / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"Warning: Failed to save embeddings to cache: {e}")

    def _load_embeddings(self):
        """Load embeddings from cache if available and valid"""
        if self.force_recompute:
            print("Force recompute enabled - skipping cache")
            return False
            
        if not self.cache_file.exists():
            print(f"Cache file not found: {self.cache_file}")
            return False
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache
            if not self._validate_cache(cache_data):
                print("Cache validation failed")
                return False
            
            # Load data
            self.data = cache_data['data']
            self.labels = cache_data['labels']
            self.indices = cache_data['indices']
            
            print(f"Loaded from cache: {len(self.data)} embeddings")
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def _validate_cache(self, cache_data):
        """Validate that cached data matches current dataset"""
        metadata = cache_data.get('metadata', {})
        
        # Check dataset size
        if metadata.get('dataset_size') != len(self.raw_dataset):
            print(f"Dataset size mismatch: cache={metadata.get('dataset_size')}, current={len(self.raw_dataset)}")
            return False
        
        # Check cache key
        if metadata.get('cache_key') != self.cache_key:
            print(f"Cache key mismatch: cache={metadata.get('cache_key')}, current={self.cache_key}")
            return False
        
        # Check data integrity
        if len(cache_data['data']) != len(cache_data['labels']):
            print("Data/labels length mismatch in cache")
            return False
        
        return True

    def extract_embeddings(self, batch_size=20):
        """Extract embeddings from the pretrained model"""
        self.pretrained_model.eval()
        print(f"Starting embedding extraction with batch_size={batch_size}")
        print(f"Total dataset size: {len(self.raw_dataset)}")
        
        # Create DataLoader
        dataloader = DataLoader(
            self.raw_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        with torch.no_grad():
            for batch_data in tqdm.tqdm(dataloader, desc="Processing batches"):
                # Unpack batch - assuming your dataset returns (transaction, labels) tuples
                transactions, labels = batch_data
                
                # Move to device
                transactions = transactions.to(device)
                
                # Get embeddings
                outputs = self.pretrained_model(transactions)
                
                # Store results
                embeddings = outputs.cpu()  # Move to CPU
                self.data.extend([embeddings[i] for i in range(embeddings.shape[0])])
                
                # Handle labels - convert to list if they're tensors
                if isinstance(labels, torch.Tensor):
                    self.labels.extend(labels.tolist())
                else:
                    self.labels.extend(labels)
                
                # Explicit cleanup
                del transactions, outputs, embeddings
                torch.cuda.empty_cache()
        
        self.indices = list(range(len(self.labels)))
        
        # Final summary
        print(f"\nEmbedding extraction completed!")
        print(f"Final counts:")
        print(f"  Data: {len(self.data)}")
        print(f"  Labels: {len(self.labels)}")
        print(f"  Indices: {len(self.indices)}")
        
        if len(self.data) > 0:
            print(f"Sample embedding shape: {self.data[0].shape}")
            print(f"Sample embedding dtype: {self.data[0].dtype}")
        
        if len(self.labels) > 0:
            sample_label = self.labels[0]
            print(f"Sample label type: {type(sample_label)}")
            print(f"Sample label shape/length: {sample_label.shape if hasattr(sample_label, 'shape') else len(sample_label) if hasattr(sample_label, '__len__') else 'scalar'}")
            print(f"Sample label content: {sample_label}")
        
        assert len(self.data) == len(self.labels) == len(self.indices)

    def clear_cache(self):
        """Clear the cache file"""
        if self.cache_file.exists():
            self.cache_file.unlink()
            print(f"Cleared cache: {self.cache_file}")
        else:
            print("No cache file to clear")

    @classmethod
    def clear_all_cache(cls, cache_dir="./embeddings_cache"):
        """Clear all cache files in the cache directory"""
        cache_path = Path(cache_dir)
        if cache_path.exists():
            cache_files = list(cache_path.glob("embeddings_*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            print(f"Cleared {len(cache_files)} cache files from {cache_dir}")
        else:
            print(f"Cache directory {cache_dir} does not exist")

    def get_cache_info(self):
        """Get information about the cache"""
        info = {
            'cache_file': str(self.cache_file),
            'cache_exists': self.cache_file.exists(),
            'cache_key': self.cache_key,
        }
        
        if self.cache_file.exists():
            info['cache_size_mb'] = self.cache_file.stat().st_size / (1024*1024)
            info['cache_modified'] = self.cache_file.stat().st_mtime
        
        return info

    def resample_train(self, train_indices, test_indices, eval_indices=[]):
        train_real_indices = [self.indices[i] for i in train_indices]
        test_real_indices = [self.indices[i] for i in test_indices]
        eval_real_indices = [self.indices[i] for i in eval_indices]
        
        # Fix: Handle labels properly - convert to numpy if they're lists or tensors
        train_labels = []
        for i in train_real_indices:
            label = self.labels[i]
            if isinstance(label, torch.Tensor):
                train_labels.append(label.numpy())
            elif isinstance(label, (list, np.ndarray)):
                train_labels.append(np.array(label))
            else:
                train_labels.append(label)
        
        test_labels = [self.labels[i] for i in test_real_indices]
        eval_labels = [self.labels[i] for i in eval_real_indices]
    
        assert len(train_real_indices)+len(test_real_indices)+len(eval_real_indices) == len(self.indices)          
        assert len(train_labels)+len(test_labels)+len(eval_labels) == len(self.labels)            
        
        logger.info('Upsample training fraudulent samples.')
        train_real_indices = np.array(train_real_indices)
        train_labels = np.array(train_labels)
        logger.info(f'train labels shape: {train_labels.shape}')
        logger.info(f'train real indices shape: {train_real_indices.shape}')
        
        non_fraud_real_indices = train_real_indices[np.all(train_labels==0, axis=1)]
        non_fraud_labels = train_labels[np.all(train_labels==0, axis=1)]
        logger.info(f'non fraud indices shape: {non_fraud_real_indices.shape}')
        logger.info(f'non fraud labels shape: {non_fraud_labels.shape}')
        
        fraud_real_indices = train_real_indices[np.any(train_labels, axis=1)]
        fraud_labels = train_labels[np.any(train_labels, axis=1)]
        logger.info(f'fraud indices shape: {fraud_real_indices.shape}')
        logger.info(f'fraud labels shape: {fraud_labels.shape}')
        
        fraud_upsample_real_indices = resample(fraud_real_indices, replace=True, n_samples=non_fraud_labels.shape[0], random_state=2022)
        logger.info(f'fraud upsample indices shape: {fraud_upsample_real_indices.shape}')
        
        train_real_indices = np.concatenate((fraud_upsample_real_indices,non_fraud_real_indices))
        logger.info(f'new train indices shape: {train_real_indices.shape}')
        
        self.indices = list(train_real_indices)+eval_real_indices+test_real_indices
        train_indices = list(np.arange(train_real_indices.shape[0]))
        eval_indices = list(np.arange(train_real_indices.shape[0],train_real_indices.shape[0]+len(eval_real_indices)))
        test_indices = list(np.arange(train_real_indices.shape[0]+len(eval_real_indices),train_real_indices.shape[0]+len(eval_real_indices)+len(test_real_indices)))
        
        shuffle(train_indices)
        shuffle(eval_indices)
        shuffle(test_indices)
        
        logger.info(f'labels shape: {np.array(self.labels).shape}')
        logger.info(f'data count: {len(self.data)}, sample shape: {self.data[0].shape if len(self.data) > 0 else "empty"}')
        logger.info(f'indices shape: {np.array(self.indices).shape}')
        
        assert len(self.data) == len(self.labels), f'data {len(self.data)} != labels {len(self.labels)}'
        assert len(self.indices) > len(self.labels), f'indices {len(self.indices)} <= labels {len(self.labels)}'
        
        return train_indices, test_indices, eval_indices

    def __getitem__(self, index):

        real_index = self.indices[index]
        # out_dict = {'input_ids': self.data[real_index], 'labels': self.labels[real_index]}
        return (self.data[real_index], torch.tensor(self.labels[real_index], dtype=torch.long))


    