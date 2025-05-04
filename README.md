# IMC2025
Image Matching Challenge 2025 on Kaggle

```
git clone git@github.com:Wuqiqi123/IMC2025.git
cd IMC2025/docker
docker build -f Dockerfile -t mast3r:dev .
docker run -it --gpus all -v /YOUR/PATH/IMC2025/:/workspace/IMC2025 --network host --name mast mast3r:dev  /bin/bash

```

# Install dependencies
```
cd /workspace/IMC2025
cd asmk/cython/
cythonize *.pyx
cd ..
pip install . 
cd ..

cd dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../
```


ln -s /your/path/image-matching-challenge-2025 data/image-matching-challenge-2025
```
