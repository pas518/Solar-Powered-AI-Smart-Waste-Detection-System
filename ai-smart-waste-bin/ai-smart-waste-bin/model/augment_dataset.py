"""
augment_dataset.py — Dataset Augmentation Pipeline
====================================================
Augments an existing labelled image dataset to increase sample count
and improve model robustness to lighting, orientation, and material variation.

Input structure:
  data/train/bio/   *.jpg/*.png
  data/train/rec/
  data/train/haz/

Output:
  data/augmented/bio/
  data/augmented/rec/
  data/augmented/haz/

Usage:
  python augment_dataset.py --input data/train --output data/augmented --factor 8
"""

from __future__ import annotations
import argparse, logging, shutil
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("augment")

CATEGORIES = ["bio", "rec", "haz"]


class AugmentPipeline:

    def __init__(self, output_size: tuple[int,int] = (224, 224)):
        self.output_size = output_size

    def apply(self, img: np.ndarray) -> list[np.ndarray]:
        """Apply all augmentation variants to a single image."""
        variants = []
        h, w = img.shape[:2]
        rng   = np.random.default_rng()

        # 1. Original (resized)
        variants.append(cv2.resize(img, self.output_size))

        # 2. Horizontal flip
        variants.append(cv2.resize(cv2.flip(img, 1), self.output_size))

        # 3. Rotation ±15°
        for angle in [rng.uniform(-15, -5), rng.uniform(5, 15)]:
            M   = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            variants.append(cv2.resize(rot, self.output_size))

        # 4. Brightness variants
        for beta in [-35, -15, 15, 35]:
            bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
            variants.append(cv2.resize(bright, self.output_size))

        # 5. Contrast variants
        for alpha in [0.75, 1.35]:
            con = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            variants.append(cv2.resize(con, self.output_size))

        # 6. HSV jitter
        for _ in range(2):
            hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.int16)
            hsv[:,:,0] = np.clip(hsv[:,:,0] + rng.integers(-10, 10), 0, 179)
            hsv[:,:,1] = np.clip(hsv[:,:,1] + rng.integers(-30, 30), 0, 255)
            jit = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            variants.append(cv2.resize(jit, self.output_size))

        # 7. Gaussian noise
        noise = rng.normal(0, 10, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        variants.append(cv2.resize(noisy, self.output_size))

        # 8. Random crop (90% of image, random corner)
        for _ in range(2):
            cx  = rng.integers(0, max(1, w//10))
            cy  = rng.integers(0, max(1, h//10))
            crop = img[cy:h-cy, cx:w-cx]
            if crop.size > 0:
                variants.append(cv2.resize(crop, self.output_size))

        # 9. Blur (simulate motion / out-of-focus)
        blurred = cv2.GaussianBlur(img, (7, 7), 0)
        variants.append(cv2.resize(blurred, self.output_size))

        # 10. Sharpen
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        sharp  = cv2.filter2D(img, -1, kernel)
        variants.append(cv2.resize(sharp, self.output_size))

        return variants


def augment_dataset(
    input_dir:  str,
    output_dir: str,
    factor:     int = 8,
    copy_original: bool = True,
):
    """
    Augment all images in input_dir/*/  →  output_dir/*/

    factor : approximate multiplication of dataset size
    """
    pipeline = AugmentPipeline()
    inp      = Path(input_dir)
    out      = Path(output_dir)
    totals   = {}

    for cat in CATEGORIES:
        src_dir = inp / cat
        dst_dir = out / cat
        dst_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(src_dir.glob("*.jpg")) +
                           list(src_dir.glob("*.jpeg")) +
                           list(src_dir.glob("*.png")))
        if not img_files:
            log.warning("No images found in %s", src_dir)
            continue

        count = 0
        for fp in img_files:
            img = cv2.imread(str(fp))
            if img is None:
                continue

            # Copy original
            if copy_original:
                dst = dst_dir / fp.name
                shutil.copy2(fp, dst)
                count += 1

            # Apply augmentations (up to factor variants)
            variants = pipeline.apply(img)
            for i, v in enumerate(variants[:factor - 1]):
                out_name = dst_dir / f"{fp.stem}_aug{i:02d}.jpg"
                cv2.imwrite(str(out_name), v, [cv2.IMWRITE_JPEG_QUALITY, 92])
                count += 1

        totals[cat] = count
        log.info("  %s: %d → %d images", cat.upper(), len(img_files), count)

    log.info("Augmentation complete: %s", totals)
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   default="data/train")
    ap.add_argument("--output",  default="data/augmented")
    ap.add_argument("--factor",  default=8, type=int)
    ap.add_argument("--no-orig", action="store_true",
                    help="Do not copy originals to output dir")
    args = ap.parse_args()

    totals = augment_dataset(
        args.input, args.output, args.factor,
        copy_original=not args.no_orig,
    )
    total_images = sum(totals.values())
    print(f"\n✓ Augmentation complete — {total_images} total images in {args.output}")
