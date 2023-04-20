# Segmentation-free direct iris localization networks

&copy; 2023 NEC Corporation

This repository is an official implementation of our paper ["Segmentation-free Direct Iris Localization Networks"](https://openaccess.thecvf.com/content/WACV2023/html/Toizumi_Segmentation-Free_Direct_Iris_Localization_Networks_WACV_2023_paper.html), WACV2023. 

## Notice
- We publish PyTorch implementation of the iris localization network (ILN) model for evaluation. If you need our pre-trained ILN model, please contact us.
- The output is limited to iris and pupil circles (6 dimensions). 
- It can used for research purpose. 
- ___Do Not Use It for Commercial Purpose___.

# License
This software is released under the NEC Corporation License. See [LICENSE](./LICENSE.txt) before using the code. If you use our model or codes for your research, please cite following paper.
```
@InProceedings{Toizumi_2023_WACV,
    author    = {Toizumi, Takahiro and Takahashi, Koichi and Tsukada, Masato},
    title     = {Segmentation-Free Direct Iris Localization Networks},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    pages     = {991-1000}
}
```

# Requirements

```
torch > 1.7.1
numpy > 1.19.2
opencv-python > 4.3.0.36
matplotlib > 3.2.2
```

# Pre-trained model
- If you need pre-trained model for your research, please contact to us. (Our email address is written in the paper.)
- The NEC license also extends to our pre-trained model. See [LICENSE](./LICENSE.txt) before using our model.  

# How to use
1. Prepare pre-trained model by yourself or get it from authors.
2. Put your dataset images in "img/\<your path\>" directory.
3. Rewrite following line in test.py to your image directory.
```python
    #files = sorted(glob('img/sample/**/*.png', recursive=True))
    files = sorted(glob('img/<your path>/**/*.<your extension (png, jpg, ...)>', recursive=True))
```
4. Run "test.py".

# Contributors
- Takahiro Toizumi, NEC Corporation. 
- Koichi Takahashi, NEC Corporation.
- Masato Tsukada, Tsukuba University.

