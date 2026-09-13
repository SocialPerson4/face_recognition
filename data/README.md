# Dataset layout

原始人脸数据只保存在本地，不提交 GitHub。

```text
data/
└── raw/
    ├── orl/
    │   ├── s1/
    │   │   ├── 1.pgm
    │   │   └── ...
    │   └── ...
    └── lfw/
        ├── lfw_funneled/
        ├── pairsDevTrain.txt
        ├── pairsDevTest.txt
        └── pairs.txt
```

## ORL

第一阶段使用。预期包含 40 个身份，每个身份 10 张 `92 × 112` 灰度图像。

来源：https://cam-orl.co.uk/facedatabase.html

## LFW Funneled

第三阶段使用。下载和实验协议记录在 `docs/research_thinking.md`，第一阶段不依赖该数据。

来源：http://vis-www.cs.umass.edu/lfw/

