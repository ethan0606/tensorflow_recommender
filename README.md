# tensorflow_recommender

a series of recommender model implemented by tensorflow

设计思路

整体分为三块

1. 特征Feed

特征主要分为两大类特征
numeric：数值型特征直接喂入
category: 类别型特征，过embedding layer

2. 使用yaml配置文件管理特征

tf.placeholder(feature_name)

category特征过embedding layer后再和numeric特征concat一起

3. 使用tf estimator 来进行训练

