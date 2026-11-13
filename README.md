# 磁瓦表面缺陷：协同检测、分割与神经网络样本生成

公开磁瓦数据集上的轻量 CPU 研究原型。提供七组对比、真实测试结果、训练权重、120 张 VAE 合成缺陷局部图，以及超过 4,000 中文字、含五幅图的项目报告。

## 数据

- 来源：[Magnetic Tile Defect Dataset](https://github.com/abin24/Magnetic-tile-defect-datasets.)，作者 Huang 等，论文 *Saliency of magnetic tile surface defects*。
- 本次版本：`cc4eb530b57b9cb7225bf76023873043ffd50655`。
- 1,344 张图像；正常 952，缺陷 392；五种缺陷合并为单个 defect 类。
- 训练/验证/测试固定分层划分见 [split.json](artifacts/split.json)。图像级划分不能保证生产批次独立。
- 原始数据不重复上传；请遵守原作者使用说明并引用论文。首次下载需要 GitHub 访问。

## 安装与复现

```bash
bash setup.sh
.venv/bin/python tests_metrics.py
.venv/bin/python src/project.py train --data data/magnetic-tiles --epochs 40
.venv/bin/python src/project.py generate --data data/magnetic-tiles --epochs 60
.venv/bin/python src/ensemble.py data/magnetic-tiles
MPLCONFIGDIR=/tmp/defect-matplotlib .venv/bin/python src/report.py data/magnetic-tiles
```

当前云环境已安装 `/workspace/ml-env/venv`，原始数据位于 `/workspace/data/magnetic-tiles`；使用该解释器和数据路径可以直接运行，无需重新下载。CPU 训练需要数分钟至数十分钟，依机器性能而定；没有 GPU 要求。复现命令会覆盖同名实验产物，请先备份需要保留的结果。

## 推理

```bash
.venv/bin/python src/infer.py path/to/image.jpg --mode ensemble --output prediction
```

输出 `mask.png` 与 `detections.json`（原图坐标、置信度）。已保存权重可直接使用。输入为灰度图，模型工作分辨率 64×64，输出放大不代表恢复了原始细节。

## 方法与实验

- `det`：独立稠密区域检测；`seg`：独立分割。
- `joint`：共享编码器双任务；`fusion`：增加分割概率到检测头的交互。
- `full`：再增加包含关系约束及分割引导的检测区域细化。
- `ensemble`：完整协同模型加独立分割专家的概率融合；验证集校准，增加参数与计算量，不能将其收益完全归因于共享编码器。
- `balanced`：保持完整结构，将分割损失权重提高为 4，检验任务权重平衡。默认训练命令运行六组；可用 `--modes balanced` 只追加指定组。
- 检测框由真值掩膜连通区域产生，检测头预测矩形覆盖图，再提取连通区域外接框。**这是区域式检测原型，不是 YOLO 或标准框回归检测器。**
- 检测指标为单类 mAP@0.5、mAP@0.5:0.95（101 点插值），分割为全测试集累积前景 IoU、前景与背景 mIoU。正常图上的误检计入评价。
- 统一训练预算和种子；只在验证集选择权重与阈值。测试结果见 [results.json](artifacts/results.json)，不要将单次结果解读为统计显著性。
- VAE 只使用训练集缺陷局部图；输出为 64×64 合成局部图，未附人工掩膜，存在模糊样本，不保证每张缺陷均清晰，也未证明能够改善下游任务。

## 交付物

- [中文项目报告（Markdown）](artifacts/report/项目报告.md)
- [中文项目报告（Word）](artifacts/report/项目报告.docx)
- [120 张生成图片](artifacts/generated/)
- [消融指标](artifacts/results.json)、[训练历史](artifacts/history.json)、[运行配置](artifacts/run.json)
- `artifacts/*.pt`：六组任务权重与 VAE 权重。
- `artifacts/*_predictions.npz`：测试集概率图与索引，方便重新评价。

推送包含代码、权重、结果和报告；不包含可重新下载的原始数据或 Python 环境。

## 本次实测结果

测试集 269 张，单次固定种子结果；全部七组见报告和 results.json。

|方案|mAP@0.5|mAP@0.5:0.95|缺陷 IoU|mIoU|
|---|---:|---:|---:|---:|
|独立检测|0.0606|0.0172|—|—|
|独立分割|—|—|0.4368|0.7091|
|单骨干完整协同|0.1996|0.0580|0.3920|0.6845|
|双专家协同融合|0.1241|0.0541|0.4988|0.7412|

最终融合方案相对各自单任务基线同时提升两项任务，但包含额外分割专家；单骨干完整协同的分割仍低于独立分割。未进行多种子显著性验证。检测采用连通区域式原型，低分辨率和小目标限制使绝对 mAP 较低，不宜直接作为工业部署依据。

已保存权重和预测后，可运行 `.venv/bin/python tests_inference.py data/magnetic-tiles` 检查 CLI 输出与保存测试预测的一致性。报告正文中文字数及图片数量见 `artifacts/report/report_checks.json`。
