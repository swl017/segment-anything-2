# Torch with CUDA 12.1
```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

# from sam2 import _C error
```
python setup.py build_ext --inplace
```
> Confirm if running pip install -e . was executed correctly, or use python setup.py build_ext to compile _C.so only. If you prefer to use the local code instead of installing the library, then use python setup.py build_ext --inplace, and you will get _C.so in 'segment-anything-2/sam2/'.