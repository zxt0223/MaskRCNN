# utils/attention_modules.py
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Dense, Multiply,
                                     Concatenate, Reshape, Layer, TimeDistributed)


class ChannelAttention(Layer):
    """通道注意力机制 - 适配Mask RCNN"""

    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        # 共享的MLP
        self.dense1 = Dense(self.channels // self.ratio, activation='relu')
        self.dense2 = Dense(self.channels)
        self.built = True

    def call(self, inputs):
        # 全局平均池化
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, self.channels))(avg_pool)

        # 全局最大池化
        max_pool = GlobalMaxPooling2D()(inputs)
        max_pool = Reshape((1, 1, self.channels))(max_pool)

        # 共享MLP
        avg_out = self.dense2(self.dense1(avg_pool))
        max_out = self.dense2(self.dense1(max_pool))

        # 合并并激活
        channel_weights = tf.sigmoid(avg_out + max_out)

        return Multiply()([inputs, channel_weights])


class SpatialAttention(Layer):
    """空间注意力机制 - 适配Mask RCNN"""

    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv2D(1, kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        # 通道维度平均池化
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # 通道维度最大池化
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # 拼接
        concat = Concatenate()([avg_pool, max_pool])

        # 卷积生成空间权重
        spatial_weights = self.conv(concat)

        return Multiply()([inputs, spatial_weights])


class CBAM(Layer):
    """混合注意力机制 - 适配Mask RCNN"""

    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x


class TimeDistributedCBAM(Layer):
    """时间分布的CBAM注意力，用于ROI特征"""

    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(TimeDistributedCBAM, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.cbam = CBAM(self.ratio, self.kernel_size)
        self.built = True

    def call(self, inputs):
        # inputs shape: [batch, num_rois, height, width, channels]
        batch_size = tf.shape(inputs)[0]
        num_rois = tf.shape(inputs)[1]
        h, w, c = inputs.shape[2], inputs.shape[3], inputs.shape[4]

        # 重塑为2D卷积格式
        x_reshaped = tf.reshape(inputs, [batch_size * num_rois, h, w, c])

        # 应用CBAM
        x_attention = self.cbam(x_reshaped)

        # 重塑回原始形状
        output = tf.reshape(x_attention, [batch_size, num_rois, h, w, c])

        return output


def add_attention_to_fpn(feature_maps, attention_type='cbam'):
    """
    在FPN特征图上添加注意力机制
    """
    attended_features = []

    for feature in feature_maps:
        if attention_type == 'cbam':
            attention_layer = CBAM(ratio=8)
        elif attention_type == 'channel':
            attention_layer = ChannelAttention(ratio=8)
        elif attention_type == 'spatial':
            attention_layer = SpatialAttention(kernel_size=7)
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")

        attended_feature = attention_layer(feature)
        attended_features.append(attended_feature)

    return attended_features