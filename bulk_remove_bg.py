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
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from wdd.file_utils import delete_old_files_and_folders, logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def setup_model(local_checkpoint_path: str, device: str):
    """
    Load BiRefNet from local checkpoint (.pth file) using original BiRefNet code.
    """
    from models.birefnet import BiRefNet
    from utils import check_state_dict

    current_dir = Path(__file__).resolve().parent
    model_path = (current_dir / local_checkpoint_path).resolve()

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


def remove_background_batch(images: list, model, transform, device, autocast_ctx):
    """
    Remove background from an image using BiRefNet.

    Args:
        image: Input PIL Image
        model: BiRefNet model
        transform: Image transformation pipeline
        device: torch device (cuda/cpu)

    Returns:
        PIL Image with transparent background
    """
    batch_tensors = []
    orig_sizes = []

    for img in images:
        orig_sizes.append(img.size)  # (W,H)
        img_rgb = img.convert("RGB")
        tensor = transform(img_rgb)

        batch_tensors.append(tensor)

    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad(), autocast_ctx:
        preds = model(batch_tensor)[-1].sigmoid()  # (B,C,H,W)

    results = []
    for i, pred in enumerate(preds):
        mask = pred[0]  # single channel
        mask_pil = transforms.ToPILImage()(mask)
        mask_pil = mask_pil.resize(orig_sizes[i])  # restore original size

        # Apply alpha channel
        img_out = images[i].convert("RGBA")
        img_out.putalpha(mask_pil)

        results.append(img_out)

    del batch_tensor, preds
    torch.cuda.empty_cache()

    return results


def process_directory(
    input_dir: str,
    output_dir: str,
    model,
    transform,
    device: str = "cuda",
    batch_size: int = 4,
    mixed_precision: str = "fp16",
):
    """
    Process all images in input directory and save to output directory.

    Args:
        input_dir: Path to input directory containing images
        output_dir: Path to output directory for processed images
        device: Device to run model on ('cuda' or 'cpu')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
        batch_images = []
        batch_names = []

        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    batch_images.append(img.copy())
                batch_names.append(p.stem)
            except Exception as e:
                logging.warning(f"Error loading {p.name}: {e}")
                failed += 1

        if not batch_images:
            continue

        try:
            batch_results = remove_background_batch(
                batch_images, model, transform, device, batch_autocast
            )
            for name, img_out in zip(batch_names, batch_results):
                img_out.save(output_path / f"{name}.png")
                successful += 1
        except Exception as e:
            logging.warning(f"Batch failed, fallback to single-image processing: {e}")

            if device == "cuda":
                torch.cuda.empty_cache()

            # 回退到逐张处理
            for img, name in zip(batch_images, batch_names):
                try:
                    single_result = remove_background_batch(
                        [img], model, transform, device, single_autocast
                    )[0]

                    single_result.save(output_path / f"{name}.png")
                    successful += 1
                except Exception as e2:
                    logging.warning(f"Failed on {name}: {e2}")
                    failed += 1

    logging.info(f"Processing complete!")
    logging.info(f"Successful: {successful}")
    if failed > 0:
        logging.warning(f"Failed: {failed}")
    logging.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch background removal using local BiRefNet checkpoint"
    )
    parser.add_argument("input_dir", type=str, help="Directory containing input images")
    parser.add_argument("output_dir", type=str, help="Directory to save output images")
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
    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = [int(x) for x in args.resolution.lower().split("x")]
    except Exception as e:
        logging.warning(f"Invalid resolution '{args.resolution}', using 1024x1024")
        width, height = 1024, 1024

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
    )


if __name__ == "__main__":
    main()
