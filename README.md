# 该项目主要有三个功能：
1. 遍历视频流，用```Mediapipe```得到关键点，对每个视频输出结果分别保存一个json文件并存储在```./output```
2. 利用```COCO eval```接口把Mediapipe输出结果分别与GT进行比较计算AP， AR，然后把结果打印到终端窗口
3. 把所有过程数据记录在```./output/total.xlsx```

下面是演示代码：

# **Test**
```
 cd assessment_automation
 python ./scripts/main.py
```
