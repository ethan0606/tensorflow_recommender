# 使用tensorflow构建一系列常用的推荐模型

规范化tensorflow训练模型的一些常用步骤，包括数据读取，特征转换，构建计算图，运行模型，保存模型和在tensorboard展示


## 特征
特征使用Feature来构建，包含了一些列特征的基础属性和转换。可以直接读取一个json文件来配置

## 数据读取
数据使用tf.dataset API 来读取，格式默认为tf record. feature_spec可以由FeatureLayer输出


## 构建计算图
计算图使用tf.estimator的model_fn来构建，使用tf.keras.layer来构建层


## 运行
训练使用tf.estimator的train_and_eval来，使用tf.save_model来保存模型，输出tensorboard log来监控训练



# 模型支持








