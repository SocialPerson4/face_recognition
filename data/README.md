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

第三阶段使用。标准文件名为 `lfw-funneled.tgz`，解压后的图片目录为 `lfw_funneled/`。本地审计确认包含 5,749 个身份和 13,233 张 `250 × 250` RGB JPEG；官方三份配对文件没有缺失引用。下载和实验协议记录在 `docs/research_thinking.md`。

来源：http://vis-www.cs.umass.edu/lfw/
