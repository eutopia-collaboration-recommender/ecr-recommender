
Install PyTorch:
```bash
pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Installation due to error: Faiss assertion 'err == CUBLAS_STATUS_SUCCESS' failed:
```
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install faiss-cpu==1.7.3
```


Install PyTorch Geometric:
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```