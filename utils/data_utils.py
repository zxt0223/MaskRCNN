# utils/data_utils.py
import json
import numpy as np
import cv2
import os
from tensorflow.keras.utils import Sequence


class SimpleMaskRCNNDataGenerator(Sequence):
    """简化的Mask RCNN数据生成器 - TF 2.4兼容"""

    def __init__(self, annotation_file, image_dir, batch_size=1, image_size=512,
                 max_instances=100, shuffle=True, augment=False):
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_instances = max_instances
        self.shuffle = shuffle
        self.augment = augment

        # 加载标注
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

        self.images = self.data['images']
        self.annotations = self.data['annotations']

        # 创建图像ID到标注的映射
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.image_ids = list(self.img_to_anns.keys())
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = []
        batch_bboxes = []
        batch_class_ids = []
        batch_masks = []

        for img_id in batch_ids:
            image, bboxes, class_ids, masks = self._load_single_image(img_id)
            batch_images.append(image)
            batch_bboxes.append(bboxes)
            batch_class_ids.append(class_ids)
            batch_masks.append(masks)

        # 转换为numpy数组
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_bboxes = np.array(batch_bboxes, dtype=np.float32)
        batch_class_ids = np.array(batch_class_ids, dtype=np.int32)
        batch_masks = np.array(batch_masks, dtype=np.float32)

        return batch_images, {
            'rpn_class': batch_class_ids,
            'rpn_bbox': batch_bboxes,
            'mrcnn_class': batch_class_ids,
            'mrcnn_bbox': batch_bboxes,
            'mrcnn_mask': batch_masks
        }

    def _load_single_image(self, img_id):
        """加载单张图像数据"""
        # 查找图像信息
        img_info = None
        for img in self.images:
            if img['id'] == img_id:
                img_info = img
                break

        if img_info is None:
            raise ValueError(f"Image {img_id} not found")

        # 加载图像
        image_path = os.path.join(self.image_dir, img_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_h, original_w = image.shape[:2]

        # 调整图像大小
        image = self._resize_image(image, self.image_size)

        # 加载标注
        anns = self.img_to_anns[img_id]
        bboxes = []
        class_ids = []
        masks = []

        for ann in anns:
            # 边界框 [x, y, width, height]
            x, y, w, h = ann['bbox']

            # 调整到新尺寸
            scale = self.image_size / max(original_h, original_w)
            x1 = x * scale
            y1 = y * scale
            x2 = (x + w) * scale
            y2 = (y + h) * scale

            bbox = [y1, x1, y2, x2]  # [y1, x1, y2, x2]
            class_id = ann['category_id']

            # 创建简单掩码（实际使用中应该使用真实的掩码）
            mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            y1_int, x1_int = max(0, int(y1)), max(0, int(x1))
            y2_int, x2_int = min(self.image_size, int(y2)), min(self.image_size, int(x2))

            if y1_int < y2_int and x1_int < x2_int:
                mask[y1_int:y2_int, x1_int:x2_int] = 1.0

            bboxes.append(bbox)
            class_ids.append(class_id)
            masks.append(mask)

        # 填充到固定长度
        bboxes = self._pad_array(bboxes, (self.max_instances, 4), 0.0)
        class_ids = self._pad_array(class_ids, (self.max_instances,), 0)
        masks = self._pad_array(masks, (self.max_instances, self.image_size, self.image_size), 0.0)

        # 归一化图像
        image = image.astype(np.float32) / 255.0

        return image, bboxes, class_ids, masks

    def _resize_image(self, image, target_size):
        """调整图像大小"""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h))

        # 填充到正方形
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2

        image = np.pad(image,
                       ((pad_h, target_size - new_h - pad_h),
                        (pad_w, target_size - new_w - pad_w),
                        (0, 0)),
                       mode='constant', constant_values=0)

        return image

    def _pad_array(self, array, target_shape, fill_value):
        """填充数组到目标形状"""
        if len(array) < target_shape[0]:
            if len(target_shape) == 1:
                padded = np.full(target_shape, fill_value, dtype=type(fill_value))
                padded[:len(array)] = array
            else:
                padded = np.full(target_shape, fill_value, dtype=type(fill_value))
                padded[:len(array)] = array
            return padded
        else:
            return np.array(array[:target_shape[0]])

    def on_epoch_end(self):
        """每个epoch结束时调用"""
        if self.shuffle:
            np.random.shuffle(self.image_ids)