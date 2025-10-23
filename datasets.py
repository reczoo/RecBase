import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):
    def __init__(self,data_path):

        self.data_path = data_path
        if 'pretrain_ds/amazonkdd23' in data_path:
            amazonkdd23_path = [
            ]
            embs = []
            for data in amazonkdd23_path:
                emb = np.load(data)
                embs.append(emb)
            self.embeddings = np.concatenate(embs, axis=0)
        else:
            self.embeddings = np.load(data_path)
        self.embeddings = self.embeddings[:5000]
        self.dim = 4096

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)

file_path=[]
class AllEmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.embeddings=[]
        self.data_path = data_path
        for path in file_path:
            array=np.load(path)
            self.embeddings.extend(array)
            print(path)
            # import pdb;pdb.set_trace()
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.dim = 4096

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class AllEmbDataset_all(data.Dataset):
    def __init__(self,data_path):
        self.file_paths = file_path
        self.index_map = []
        self._build_index_map()
        self.dim = 4096
    def _build_index_map(self):
        for file_idx, path in enumerate(self.file_paths):
            # import pdb;pdb.set_trace()
            array = np.load(path, mmap_mode='r') 
            for item_idx in range(array.shape[0]):
                self.index_map.append((file_idx, item_idx))

    def __getitem__(self, index):
        file_idx, item_idx = self.index_map[index]
        file_path = self.file_paths[file_idx]
        array = np.load(file_path, mmap_mode='r')
        emb = array[item_idx]
        
        writable_emb = np.array(emb, copy=True)
        
        tensor_emb = torch.tensor(writable_emb, dtype=torch.float32)
        return tensor_emb
    def __len__(self):
        return len(self.index_map)