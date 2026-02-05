#!/usr/bin/env python3
"""
High-throughput local batch background removal using a local BiRefNet model.
Processes all images in an input directory and saves them with transparent backgrounds.
Supports GPU, batch inference, and half precision.
"""

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from wdd.file_utils import init_logging, logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def setup_model(local_checkpoint_path: str, device: str):
    """
    Load BiRefNet from local checkpoint (.pth file) using original BiRefNet code.
    """
    from models.birefnet import BiRefNet
    from utils import check_state_dict

    model_path = (ROOT_DIR / local_checkpoint_path).resolve()

    logging.info(f"Loading BiRefNet model from {model_path} on {device}...")

    torch.set_float32_matmul_precision("high")

    # Initialize model structure
    model = BiRefNet(bb_pretrained=False)

    # Load local weights
    state_dict = torch.load(str(model_path), map_location="cpu")
    state_dict = check_state_dict(state_dict)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def get_transform(img_size=(1024, 1024)):
    """Get the image transformation pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def enhance_hair_details(mask_np, img_np):
    """增强头发丝细节"""
    # 提取高频细节（头发丝通常为高频）
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    high_freq = cv2.subtract(gray, gray_blur)

    # 增强高频部分
    high_freq_enhanced = cv2.multiply(high_freq, 1.5)
    high_freq_enhanced = np.clip(high_freq_enhanced, 0, 255)

    # 提取可能的头发丝区域（高频且亮度适中）
    _, hair_candidate = cv2.threshold(high_freq_enhanced, 15, 255, cv2.THRESH_BINARY)

    # 只保留mask边缘附近的头发丝
    mask_edges = cv2.Canny((mask_np * 255).astype(np.uint8), 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    mask_edges_dilated = cv2.dilate(mask_edges, kernel, iterations=2)

    # 提取并增强头发丝
    hair_details = cv2.bitwise_and(hair_candidate, mask_edges_dilated)

    # 将头发丝细节添加到mask
    mask_with_hair = cv2.bitwise_or((mask_np * 255).astype(np.uint8), hair_details)

    return mask_with_hair.astype(np.float32) / 255.0


def remove_background_batch(
    images: list,
    model,
    transform,
    device,
    autocast_ctx,
    max_workers=8,
    hair_detail=True,  # 是否增强头发丝细节
    threshold=0.45,  # 人物通常需要稍低的阈值保留发丝
):
    """
    Remove background from an image using BiRefNet.

    Args:
        image: Input PIL Image
        model: BiRefNet model
        transform: Image transformation pipeline
        device: torch device (cuda/cpu)
        hair_detail: 增强头发丝细节
        threshold: 分割阈值

    Returns:
        PIL Image with transparent background
    """

    # Step 1: GPU batch inference
    batch_tensors = []

    for img in images:
        tensor = transform(img)
        batch_tensors.append(tensor)

    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad(), autocast_ctx:
        preds = model(batch_tensor)[-1].sigmoid()  # (B,C,H,W)

    # Step 2: CPU post-processing per image
    results = [None] * len(images)
    marks = [None] * len(images)

    def process_single(i_pred_img):
        i, (pred, img) = i_pred_img

        # 获取原图
        img_np = np.array(img.convert("RGB"))
        h, w = img.height, img.width

        # 处理预测张量
        if pred.ndim == 3:
            pred = pred.squeeze(0)
        elif pred.ndim != 2:
            raise ValueError(f"Unexpected pred shape: {pred.shape}")

        pred = pred.unsqueeze(0).unsqueeze(0)

        # 上采样到原图尺寸 - 使用bicubic保留更多细节
        mask = (
            F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False)
            .squeeze()
            .cpu()
            .numpy()
        )

        # 1. 基础阈值化（针对人物调整阈值）
        mask_binary = np.where(mask > threshold, 1.0, 0.0).astype(np.float32)

        # 2. 轻微高斯模糊去噪
        mask_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0.5)

        # 3. 形态学操作 - 填充小的孔洞（如头发间的空隙）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_closed = cv2.morphologyEx(mask_blur, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # 头发丝细节增强
        if hair_detail:
            mask_enhanced = enhance_hair_details(mask_clean, img_np)
        else:
            mask_enhanced = mask_clean

        # 确保值范围在0-1
        mask_enhanced = np.clip(mask_enhanced, 0.0, 1.0)

        # 生成最终结果
        mask_u8 = (mask_enhanced * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_u8, mode="L")

        # 生成透明图
        rgba_img = Image.new("RGBA", (w, h))
        rgba_img.paste(img.convert("RGBA"), (0, 0), mask_img)  # 使用掩码作为 alpha 通道

        return i, rgba_img, mask_img

    # 使用线程池处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(process_single, enumerate(zip(preds, images)))
        for i, rgba_img, mask_img in futures:
            results[i] = rgba_img
            marks[i] = mask_img

    del batch_tensor, preds

    return results, marks


def get_safe_max_workers(safety_factor=0.5, max_limit=16):
    """
    根据 CPU 核心数计算安全线程数，避免占满系统。

    Args:
        safety_factor: 使用核心数比例（0~1），默认 0.5
        max_limit: 最大线程数上限，默认 16
    Returns:
        int: 安全线程数
    """
    cpu_count = os.cpu_count() or 1
    safe_workers = max(1, int(cpu_count * safety_factor))
    return min(safe_workers, max_limit)


def load_image_safe(path):
    """安全加载图片并返回副本，失败返回 None"""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")  # RGB 规范化
            return img.copy()
    except Exception as e:
        logging.warning(f"Failed to load {path}: {e}")
        return None


def load_images_threaded(image_paths, max_workers=8):
    """
    使用线程池加载一批图像，并保持原始顺序
    Args:
        image_paths: list[Path] 图片路径列表
        max_workers: 最大线程数
    Returns:
        list[PIL.Image], list[Path]  成功加载的图像及其路径
    """
    images = []
    loaded_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map 会按 image_paths 的顺序返回结果
        for img, path in zip(executor.map(load_image_safe, image_paths), image_paths):
            if img is not None:
                images.append(img)
                loaded_paths.append(path)

    return images, loaded_paths


def save_batch_results_threaded(
    batch_names, batch_results, batch_marks, output_path, save_mark=False, max_workers=8
):
    """
    多线程保存批量 RGBA 图像和 mask 图。

    Args:
        batch_names: 图片名称列表（不带后缀）
        batch_results: RGBA PIL.Image 列表
        batch_marks: mask PIL.Image 列表或 None
        output_path: 输出目录 Path 对象
        save_mark: 是否保存 mask 图
        max_workers: 线程数
    Returns:
        successful: 成功保存的数量
    """
    successful = 0

    def worker(args):
        name, img_out, mark_out = args
        try:
            img_out.save(output_path / f"{name}.png")
            if save_mark and mark_out:
                mark_out.save(output_path / f"mark{name}.png")
            return 1
        except Exception as e:
            logging.warning(f"Failed to save {name}: {e}")
            return 0

    tasks = list(zip(batch_names, batch_results, batch_marks))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(worker, tasks):
            successful += res

    return successful


def get_person_matting_params(person_type="general"):
    """根据人物类型返回优化的参数"""
    params_map = {
        "general": {  # 通用人物（默认）
            "hair_detail": True,  # 开启头发细节
            "threshold": 0.45,  # 中等阈值
        },
        "portrait": {  # 肖像特写（面部特写）
            "hair_detail": True,  # 重要！面部特写需要头发细节
            "threshold": 0.42,  # 稍低阈值，保留面部细节
        },
        "full_body": {  # 全身照
            "hair_detail": False,  # 关闭头发细节，提高处理速度
            "threshold": 0.5,  # 稍高阈值，全身照轮廓需要更清晰
        },
        "long_hair": {  # 长发人物
            "hair_detail": True,  # 重要！长发需要细节
            "threshold": 0.38,  # 较低阈值，保留发丝
        },
        "group": {  # 多人合照
            "hair_detail": False,  # 关闭头发细节，提高处理速度
            "threshold": 0.5,  # 较高阈值，确保每个人物轮廓清晰
        },
        "id_photo": {  # 证件照
            "hair_detail": False,  # 证件照通常发型整齐，不需要细节增强
            "threshold": 0.55,  # 高阈值，确保轮廓非常清晰
        },
        "artistic": {  # 艺术照/写真
            "hair_detail": True,  # 需要头发细节
            "threshold": 0.4,  # 低阈值保留更多艺术细节
        },
        "ecommerce": {  # 电商模特图
            "hair_detail": True,  # 需要头发细节
            "threshold": 0.45,  # 中等阈值
        },
    }

    return params_map.get(person_type, params_map["general"])


def process_directory(
    input_dir: str,
    output_dir: str,
    model,
    transform,
    device: str = "cuda",
    batch_size: int = 4,
    mixed_precision: str = "fp16",
    save_mark: bool = False,
    person_type="full_body",
):
    """
    Process all images in input directory and save to output directory.

    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for processed images
        device: Device to run model on ('cuda' or 'cpu')
    """
    # 记录开始时间
    start_time = time.time()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    max_workers = get_safe_max_workers(safety_factor=0.5, max_limit=16)
    params = get_person_matting_params(person_type)

    logging.info(
        f"Using up to {max_workers} threads for image loading/saving."
        f"\nparams: {params}"
    )

    # Setup autocast context
    if mixed_precision == "fp16":
        mixed_dtype = torch.float16
    elif mixed_precision == "bf16":
        mixed_dtype = torch.bfloat16
    else:
        mixed_dtype = None

    batch_autocast = (
        torch.amp.autocast(device_type="cuda", dtype=mixed_dtype)
        if mixed_dtype and device == "cuda"
        else nullcontext()
    )

    single_autocast = nullcontext()  # 单张处理更稳，关闭 autocast

    # Gather all images
    image_files = [
        p
        for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        logging.warning(f"No images found in {input_dir}")
        return

    logging.info(f"Found {len(image_files)} images to process.")

    # Batch processing
    successful, failed = 0, 0
    for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_paths = image_files[i : i + batch_size]

        # 线程池加载，保持顺序
        batch_images, paths = load_images_threaded(batch_paths, max_workers=max_workers)
        batch_names = [p.stem for p in paths]

        try:
            batch_results, batch_marks = remove_background_batch(
                batch_images,
                model,
                transform,
                device,
                batch_autocast,
                max_workers=max_workers,
                **params,
            )

            successful += save_batch_results_threaded(
                batch_names=batch_names,
                batch_results=batch_results,
                batch_marks=batch_marks,
                output_path=output_path,
                save_mark=save_mark,
                max_workers=max_workers,
            )
        except Exception as e:
            logging.warning(f"Batch failed, fallback to single-image processing: {e}")

            if device == "cuda":
                torch.cuda.empty_cache()

            # 回退到逐张处理
            for img, name in zip(batch_images, batch_names):
                try:
                    results, marks = remove_background_batch(
                        [img],
                        model,
                        transform,
                        device,
                        single_autocast,
                        max_workers=max_workers,
                        **params,
                    )
                    single_result = results[0]
                    single_mark = marks[0]

                    single_result.save(output_path / f"{name}.png")
                    if save_mark:
                        single_mark.save(output_path / f"mark{name}.png")
                    successful += 1
                except Exception as e2:
                    logging.warning(f"Failed on {name}: {e2}")
                    failed += 1
                finally:
                    del img, results, marks
        finally:
            del batch_results, batch_marks, batch_images
            if device == "cuda" and i > 0 and i % 8 == 0:
                torch.cuda.empty_cache()

    # 计算耗时
    elapsed = time.time() - start_time

    logging.info(f"Processing complete in {elapsed} seconds.")
    logging.info(f"Successful: {successful}")
    if failed > 0:
        logging.warning(f"Failed: {failed}")
    logging.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch background removal using local BiRefNet checkpoint"
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save output images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoint/BiRefNet-general-epoch_244.pth",
        help="Path to local BiRefNet checkpoint (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run model on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["fp16", "bf16", "none"],
        default="fp16",
        help="Automatic mixed precision: fp16, bf16, or none",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1024x1024",
        help="Input image resolution (e.g., 1024x1024 or 512x512)",
    )
    parser.add_argument(
        "--save_mark", action="store_true", default=False, help="save mark"
    )
    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = [int(x) for x in args.resolution.lower().split("x")]
    except Exception as e:
        logging.warning(f"Invalid resolution '{args.resolution}', using 1024x1024")
        width, height = 1024, 1024

    try:
        init_logging(str(ROOT_DIR))

        # Setup model and transform
        model = setup_model(args.checkpoint, args.device)
        transform = get_transform((width, height))

        process_directory(
            args.input_dir,
            args.output_dir,
            model,
            transform,
            args.device,
            batch_size=args.batch_size,
            mixed_precision=args.mixed_precision,
            save_mark=args.save_mark,
        )
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
        sys.exit(1)
    finally:
        # 释放模型和 GPU
        try:
            del model
        except NameError:
            pass
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
