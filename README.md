# Assessment Automation
```Introdution```: 对于社区热门API比较的时候，我们以往是通过定性分析，
通过模型输出人工一张张去统计计算出精确度，这样的工作量大而且对于精度等级相同的模型很难客观的评价出哪个模型更佳，
所以我们想通过对coco eval接口重新封装，计算每张图所有关键点的OKS相似度得到10个IOU阈值下的AR和AP，最后把输出的mAR和mAR综合计算得到F1，
并把所有输出最后保存到excel文档方便数据整理。

下面是演示代码：

# **Run The Test**
```
 cd assessment_automation
 python ./scripts/eval_mediapipe-video.py
```
