"""
rename_polygen_masks.py
-----------------------
Rename PolyGen masks from `{name}_mask.jpg` → `{name}.jpg`
so that filenames in images/ and masks/ match exactly (required by CleanPolypDataset).

Usage:
    # Dry-run (preview only, no changes):
    python rename_polygen_masks.py --root datasets/PolyGen --dry-run

    # Actually rename:
    python rename_polygen_masks.py --root datasets/PolyGen

    # Only one split:
    python rename_polygen_masks.py --root datasets/PolyGen --splits Train
"""

import argparse
import os
import sys


def _rename_masks(masks_dir: str, images_dir: str, dry_run: bool) -> tuple[int, int, int]:
    """
    Rename mask files to match image filenames.

    Matching rule:
        mask_stem = image_stem + "_mask"   → rename to image_stem + image_ext

    Returns (renamed, skipped_already_ok, errors).
    """
    if not os.path.isdir(images_dir):
        print(f"  [ERROR] images dir not found: {images_dir}")
        return 0, 0, 1
    if not os.path.isdir(masks_dir):
        print(f"  [ERROR] masks dir not found: {masks_dir}")
        return 0, 0, 1

    image_files = sorted(f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)))
    mask_files_set = set(os.listdir(masks_dir))

    renamed = 0
    skipped = 0
    errors = 0

    for img_name in image_files:
        img_stem, img_ext = os.path.splitext(img_name)  # e.g. "100H0050", ".jpg"

        # Target: mask should be named same as image
        target_name = img_name  # e.g. "100H0050.jpg"

        # If already correctly named — skip
        if target_name in mask_files_set:
            skipped += 1
            continue

        # Expected current mask name pattern: {stem}_mask{ext}
        candidates = [
            f"{img_stem}_mask{img_ext}",  # same extension  e.g. 100H0050_mask.jpg
            f"{img_stem}_mask.jpg",
            f"{img_stem}_mask.png",
        ]
        found = None
        for cand in candidates:
            if cand in mask_files_set:
                found = cand
                break

        if found is None:
            # Fallback: try cleaning invalid chars from stem (e.g. "file].jpg" -> "file.jpg")
            clean_stem = "".join(c for c in img_stem if c.isalnum() or c in "_-.")
            if clean_stem != img_stem:
                fallback_candidates = [
                    f"{clean_stem}_mask{img_ext}",
                    f"{clean_stem}_mask.jpg",
                    f"{clean_stem}_mask.png",
                ]
                for cand in fallback_candidates:
                    if cand in mask_files_set:
                        found = cand
                        # Also rename the image to use the clean name
                        clean_img_name = clean_stem + img_ext
                        img_src = os.path.join(images_dir, img_name)
                        img_dst = os.path.join(images_dir, clean_img_name)
                        if dry_run:
                            print(f"  [DRY  ] image rename: {img_name}  ->  {clean_img_name}  (cleaned invalid chars)")
                        else:
                            os.rename(img_src, img_dst)
                            print(f"  [OK   ] image rename: {img_name}  ->  {clean_img_name}  (cleaned invalid chars)")
                        target_name = clean_img_name
                        break

        if found is None:
            print(f"  [WARN ] No mask found for image: {img_name}")
            errors += 1
            continue

        src = os.path.join(masks_dir, found)
        dst = os.path.join(masks_dir, target_name)

        if dry_run:
            print(f"  [DRY  ] {found}  ->  {target_name}")
        else:
            os.rename(src, dst)
            # Update set so subsequent lookups are accurate
            mask_files_set.discard(found)
            mask_files_set.add(target_name)
            print(f"  [OK   ] {found}  ->  {target_name}")
        renamed += 1

    return renamed, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Rename PolyGen masks to match image filenames")
    parser.add_argument(
        "--root",
        default="datasets/PolyGen",
        help="Root folder containing Train/ and Test/ subdirs",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Test"],
        help="Which splits to process (default: Train Test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without making any changes",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.root):
        args.root = os.path.join(repo_root, args.root)

    if not os.path.isdir(args.root):
        print(f"[ERROR] Root not found: {args.root}")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN — no files will be changed ===\n")

    total_renamed = total_skipped = total_errors = 0

    for split in args.splits:
        split_dir = os.path.join(args.root, split)
        images_dir = os.path.join(split_dir, "images")
        masks_dir = os.path.join(split_dir, "masks")

        print(f"[{split}] images: {images_dir}")
        print(f"[{split}] masks : {masks_dir}")

        renamed, skipped, errors = _rename_masks(masks_dir, images_dir, dry_run=args.dry_run)
        total_renamed += renamed
        total_skipped += skipped
        total_errors += errors

        print(f"[{split}] renamed={renamed}  already_ok={skipped}  warnings={errors}\n")

    print("=" * 50)
    print(f"TOTAL  renamed={total_renamed}  already_ok={total_skipped}  warnings={total_errors}")

    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")

    if total_errors > 0:
        print(f"\n[WARNING] {total_errors} image(s) have no matching mask — check the files above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
