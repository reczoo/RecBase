This repository contains the  implementation of our paper:
**RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation**.

---


## ⚙️ Installation
```bash
git clone https://github.com/reczoo/RecBase.git
cd RecBase
pip install -r requirements.txt      
```

---

## 🗂️ Project Layout
```
RecBase/
├── data_process/               # data & embedding pipeline
│   ├── amazon18_data_process.py
│   ├── amazon_text_emb.py
│   └── utils.py
├── models/                     # CL-VAE model & trainer
│   ├── clvae.py
│   ├── layers.py
│   └── ...
├── train.sh                    # one-click training
├── README.md                   
└── requirements.txt
```

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Data Pre-processing
```bash
# (1) ID + text cleaning + k-core filtering + chronological train/valid/test split
python data_process/amazon18_data_process.py \
  --dataset Games \
  --input_path  /raw/amazon2018 \
  --output_path data/Games \
  --user_k 5 --item_k 5

# (2) LLM text embedding (LLaMA-2 by default)
python data_process/amazon_text_emb.py \
  --dataset Games \
  --root data/Games \
  --plm_checkpoint meta-llama/Llama-2-7b-chat-hf \
  --gpu_id 0
```
After finishing you will obtain  
`data/Games/Games.emb-llama2-td.npy` (item text vectors) and all atomic files RecBole / SR-GNN ready.

---

### 2️⃣ Train CL-VAE
Edit `scripts/train.sh`:
```bash
--data_path data/path_to_emb-llama2-td.npy
```
Then run:
The script automatically loads embeddings → trains CL-VAE → saves latent codes & codebook.  
Check-points are stored in `ckpts` by default.

---

### 3️⃣ Use Generated Semantic Codes
Two files are ready after training:

| File | Description |
|---|---|
| `ckpts/Games/index.npy` | discrete semantic code for each item (shape: [num_items, code_len]) |
| `ckpts/Games/codebook.npy` | learnable codebook vectors |

---
The trained uniform index codebook will be used to further pre-train the dedicated llm.

## 📚 Citation
If you use RecBase, please cite our paper:

```bibtex
@misc{zhou2025recbasegenerativefoundationmodel,
      title={RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation}, 
      author={Sashuai Zhou and Weinan Gan and Qijiong Liu and Ke Lei and Jieming Zhu and Hai Huang and Yan Xia and Ruiming Tang and Zhenhua Dong and Zhou Zhao},
      year={2025},
      eprint={2509.03131},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2509.03131}, 
}
```

---

## 🤝 Contributing
Issues, PRs, new datasets & models are welcome! Let's make text semantics a first-class citizen in recommendation.