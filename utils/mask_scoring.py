# utils/mask_scoring.py
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, GlobalAveragePooling2D,
                                     Concatenate, MaxPooling2D, Layer, TimeDistributed, Reshape)


class MaskScoringHead(Layer):
    """掩码评分头部网络 - 适配Mask RCNN"""

    def __init__(self, num_classes=1, pool_size=7, **kwargs):
        super(MaskScoringHead, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.pool_size = pool_size

    def build(self, input_shape):
        # 特征提取卷积
        self.conv1 = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.conv2 = TimeDistributed(Conv2D(256, (3, 3), padding='same', activation='relu'))

        # 全局池化
        self.global_pool = TimeDistributed(GlobalAveragePooling2D())

        # 全连接层
        self.dense1 = TimeDistributed(Dense(128, activation='relu'))
        self.dense2 = TimeDistributed(Dense(self.num_classes, activation='sigmoid'))

        self.built = True

    def call(self, inputs):
        # 输入: [roi_features, mask_predictions]
        # roi_features: [batch, num_rois, pool_size, pool_size, channels]
        # mask_predictions: [batch, num_rois, 2*pool_size, 2*pool_size, num_classes]
        roi_features, mask_predictions = inputs

        # 池化掩码预测到与roi_features相同的空间尺寸
        mask_pooled = TimeDistributed(MaxPooling2D((2, 2)))(mask_predictions)  # 28x28 -> 14x14

        # 如果还需要进一步池化到7x7
        if self.pool_size == 7:
            mask_pooled = TimeDistributed(MaxPooling2D((2, 2)))(mask_pooled)  # 14x14 -> 7x7

        # 拼接特征
        concat_features = Concatenate(axis=-1)([roi_features, mask_pooled])

        # 特征提取
        x = self.conv1(concat_features)
        x = self.conv2(x)

        # 全局池化 + 全连接
        x = self.global_pool(x)
        x = self.dense1(x)
        mask_scores = self.dense2(x)

        return mask_scores


class MaskScoringRCNN(Layer):
    """包含掩码评分的Mask RCNN改进版本"""

    def __init__(self, num_classes, pool_size=7, **kwargs):
        super(MaskScoringRCNN, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.mask_scoring_head = MaskScoringHead(num_classes, pool_size)

    def call(self, inputs):
        # 输入: [rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, feature_maps, image_meta]
        rois, mrcnn_class, mrcnn_bbox, mrcnn_mask, feature_maps, image_meta = inputs

        # 原有的检测逻辑保持不变
        # ...

        # 添加掩码评分
        if len(inputs) > 4:  # 如果有特征图输入
            # 使用PyramidROIAlign获取ROI特征
            from nets.layers import PyramidROIAlign
            roi_align = PyramidROIAlign([self.pool_size, self.pool_size])
            roi_features = roi_align([rois, image_meta] + feature_maps)

            # 计算掩码评分
            mask_scores = self.mask_scoring_head([roi_features, mrcnn_mask])

            return mrcnn_class, mrcnn_bbox, mrcnn_mask, mask_scores
        else:
            return mrcnn_class, mrcnn_bbox, mrcnn_mask


def apply_mask_scoring(config, model, feature_maps):
    """
    在现有Mask RCNN模型上应用掩码评分
    """
    # 获取模型输出
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox = model.outputs[:3]
    mrcnn_mask = model.outputs[3]

    # 创建掩码评分头
    mask_scoring_head = MaskScoringHead(
        num_classes=config.NUM_CLASSES,
        pool_size=config.POOL_SIZE
    )

    # 获取ROI特征
    from nets.layers import PyramidROIAlign
    roi_align = PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE])

    # 这里需要根据实际模型结构调整输入
    # 假设我们有rois和image_meta作为输入
    rois = model.get_layer('ROI').output
    image_meta = model.get_layer('input_image_meta').output

    roi_features = roi_align([rois, image_meta] + feature_maps)

    # 计算掩码评分
    mask_scores = mask_scoring_head([roi_features, mrcnn_mask])

    return mask_scores