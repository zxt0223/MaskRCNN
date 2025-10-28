import os
import json
import shutil
import glob
from datetime import datetime


def create_directory_structure(base_dir):
    """创建所需的目录结构"""
    directories = [
        "annotations",
        "train300/images",  # 调整为300张训练
        "test30/images",  # 调整为30张测试
        "test29/images",  # 调整为29张测试
        "test30_2/images",  # 另一个30张测试集
        "unlabeled_images"
    ]

    for directory in directories:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
        print(f"创建目录: {os.path.join(base_dir, directory)}")


def labelme_to_coco_by_split(source_dir, target_base_dir):
    """
    将LabelMe标注转换为COCO格式并按划分放到对应目录

    Args:
        source_dir: 包含LabelMe JSON文件和原图的源目录
        target_base_dir: 目标data目录
    """

    # 创建目录结构
    create_directory_structure(target_base_dir)

    # 获取所有LabelMe JSON文件并排序
    json_files = sorted(glob.glob(os.path.join(source_dir, "*.json")))

    print(f"找到 {len(json_files)} 个LabelMe标注文件")

    if len(json_files) != 389:
        print(f"警告: 期望389个标注文件，实际找到{len(json_files)}个")

    # 重新定义划分 (389张已标注)
    splits = {
        "train300": (0, 300),  # 300张训练
        "test30": (300, 330),  # 30张测试
        "test29": (330, 359),  # 29张测试
        "test30_2": (359, 389)  # 30张测试
    }

    # 处理每个划分
    for split_name, (start_idx, end_idx) in splits.items():
        print(f"\n处理 {split_name}...")

        split_files = json_files[start_idx:end_idx]

        # COCO格式模板
        coco_data = {
            "info": {
                "description": f"{split_name} dataset converted from LabelMe",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"url": "", "id": 1, "name": ""}],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": 1,
                    "name": "stone",
                    "supercategory": "none"
                }
            ]
        }

        # 计数器
        image_id = 0
        annotation_id = 0

        for json_file in split_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)

            image_filename = labelme_data["imagePath"]
            image_src_path = os.path.join(os.path.dirname(json_file), image_filename)
            image_dst_path = os.path.join(target_base_dir, split_name, "images", image_filename)

            if not os.path.exists(image_src_path):
                print(f"警告: 图像文件不存在: {image_src_path}")
                continue

            # 复制图像文件
            shutil.copy2(image_src_path, image_dst_path)

            # 添加图像信息
            image_info = {
                "id": image_id,
                "file_name": image_filename,
                "height": labelme_data["imageHeight"],
                "width": labelme_data["imageWidth"],
                "license": 1,
                "url": "",
                "date_captured": ""
            }
            coco_data["images"].append(image_info)

            # 处理标注
            for shape in labelme_data.get("shapes", []):
                if shape["shape_type"] == "polygon":
                    segmentation = []
                    for point in shape["points"]:
                        segmentation.extend([point[0], point[1]])

                    x_coords = [point[0] for point in shape["points"]]
                    y_coords = [point[1] for point in shape["points"]]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    area = (x_max - x_min) * (y_max - y_min)

                    annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": [segmentation],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }

                    coco_data["annotations"].append(annotation)
                    annotation_id += 1

            image_id += 1
            print(f"处理完成: {image_filename}")

        # 保存COCO格式的标注文件
        output_json_path = os.path.join(target_base_dir, "annotations", f"{split_name}.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"{split_name} 转换完成!")
        print(f"图像数量: {len(coco_data['images'])}")
        print(f"标注数量: {len(coco_data['annotations'])}")


def copy_unlabeled_images(source_dir, target_base_dir):
    """
    复制未标注图像到unlabeled_images目录

    Args:
        source_dir: 包含所有图像的源目录
        target_base_dir: 目标data目录
    """
    print(f"\n处理未标注图像...")

    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(source_dir, ext)))

    # 获取所有已处理的图像（通过JSON文件）
    json_files = glob.glob(os.path.join(source_dir, "*.json"))
    processed_images = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
        image_path = os.path.join(source_dir, labelme_data["imagePath"])
        processed_images.append(image_path)

    # 找出未标注的图像（不包括已处理的）
    unlabeled_images = [img for img in all_images if img not in processed_images]

    print(f"找到 {len(unlabeled_images)} 个未标注图像")

    # 复制未标注图像
    for i, image_path in enumerate(unlabeled_images):
        filename = os.path.basename(image_path)
        dst_path = os.path.join(target_base_dir, "unlabeled_images", filename)
        shutil.copy2(image_path, dst_path)
        print(f"未标注图像已复制: {filename}")

    print(f"已复制 {len(unlabeled_images)} 个未标注图像")


def verify_structure(target_base_dir):
    """
    验证生成的目录结构是否正确
    """
    print(f"\n验证目录结构...")

    # 检查目录是否存在
    required_dirs = [
        "annotations",
        "train300/images",
        "test30/images",
        "test29/images",
        "test30_2/images",
        "unlabeled_images"
    ]

    for directory in required_dirs:
        dir_path = os.path.join(target_base_dir, directory)
        if not os.path.exists(dir_path):
            print(f"错误: 目录不存在: {dir_path}")
            return False
        print(f"✓ {directory}")

    # 检查COCO标注文件
    required_annotations = [
        "train300.json",
        "test30.json",
        "test29.json",
        "test30_2.json"
    ]

    for ann_file in required_annotations:
        ann_path = os.path.join(target_base_dir, "annotations", ann_file)
        if not os.path.exists(ann_path):
            print(f"错误: 标注文件不存在: {ann_path}")
            return False

        # 验证JSON文件格式
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ {ann_file} (图像: {len(data['images'])}张, 标注: {len(data['annotations'])}个)")
        except Exception as e:
            print(f"错误: {ann_file} 格式不正确: {e}")
            return False

    # 检查图像数量
    splits = {
        "train300": 300,
        "test30": 30,
        "test29": 29,
        "test30_2": 30
    }

    for split_name, expected_count in splits.items():
        image_dir = os.path.join(target_base_dir, split_name, "images")
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            actual_count = len(image_files)
            if actual_count == expected_count:
                print(f"✓ {split_name}/images (图像: {actual_count}张)")
            else:
                print(f"警告: {split_name}/images 图像数量不符 (期望: {expected_count}, 实际: {actual_count})")

    # 检查未标注图像
    unlabeled_dir = os.path.join(target_base_dir, "unlabeled_images")
    if os.path.exists(unlabeled_dir):
        unlabeled_files = [f for f in os.listdir(unlabeled_dir) if
                           f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"✓ unlabeled_images (图像: {len(unlabeled_files)}张)")

    print("✓ 目录结构验证完成!")
    return True


if __name__ == "__main__":
    # 配置路径 - 根据你的实际情况修改
    source_directory = r"D:\MaskRCNN_TF2\Stone\mask-rcnn-tf2-master-1 - 2-1\datasets\before"  # 源文件夹A路径
    target_base_directory = r"D:\MaskRCNN_TF2\Stone\mask-rcnn-tf2-master-1 - 2-1\data"  # 目标data目录

    print("开始转换LabelMe标注到COCO格式...")
    print(f"源目录: {source_directory}")
    print(f"目标目录: {target_base_directory}")

    # 检查源目录是否存在
    if not os.path.exists(source_directory):
        print(f"错误: 源目录不存在: {source_directory}")
        exit(1)

    # 转换标注并复制图像
    labelme_to_coco_by_split(source_directory, target_base_directory)

    # 复制未标注图像
    copy_unlabeled_images(source_directory, target_base_directory)

    # 验证目录结构
    verify_structure(target_base_directory)

    print(f"\n所有处理完成!")
    print(f"数据划分总结:")
    print(f"- train300: 300张训练图像")
    print(f"- test30: 30张测试图像")
    print(f"- test29: 29张测试图像")
    print(f"- test30_2: 30张测试图像")
    print(f"- 总计: 300+30+29+30 = 389张已标注图像")
    print(f"- unlabeled_images: 未标注图像")
    print(f"数据已保存到: {target_base_directory}")