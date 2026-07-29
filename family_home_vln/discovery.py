"""Category-free object discovery from a G1-D RGB survey.

Florence-2 receives only task tokens (object detection / dense region caption)
and the RGB image.  No household category list, scene truth, USD prim name, or
object coordinate is supplied to the model.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable


TASKS = ("<OD>", "<DENSE_REGION_CAPTION>")
DISCOVERY_PIPELINE_VERSION = 4
STRUCTURAL_LABELS = {
    "airplane",
    "airplane wing",
    "building",
    "ceiling",
    "floor",
    "furniture",
    "green diagonal line",
    "green cord",
    "green line",
    "green pen",
    "green plastic sticks",
    "green plastic stick",
    "green plastic straw",
    "green plastic tube",
    "green straw",
    "green string",
    "green string on wooden floor",
    "green string on wooden table",
    "green tube",
    "green toy",
    "green triangle",
    "green ruler",
    "green straight line",
    "green wire",
    "ground",
    "house",
    "paneling",
    "room",
    "wall",
    "white wall",
    "window",
    "wood floor",
    "wood paneling",
    "wooden floor",
    "wooden paneling",
    "wooden wall",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def survey_signature(manifest_path: Path, rgb_dir: Path) -> str:
    digest = hashlib.sha256(manifest_path.read_bytes())
    for path in sorted(rgb_dir.glob("*.png")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()


def validate_survey(manifest_path: Path, rgb_dir: Path) -> tuple[dict, list[Path]]:
    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    rgb = sorted(rgb_dir.glob("*.png"))
    if not frames or len(frames) != len(rgb):
        raise ValueError(
            "RGB 巡检不完整：capture_manifest 有 "
            f"{len(frames)} 帧，但目录有 {len(rgb)} 张图；请重新运行 home-survey。"
        )
    expected = [str(item.get("image", "")).split("/")[-1] for item in frames]
    actual = [path.name for path in rgb]
    if expected != actual:
        raise ValueError("RGB 巡检帧名与 capture_manifest 不一致")
    if manifest.get("object_category_labels_supplied_to_perception") is not False:
        raise ValueError("巡检制品没有声明 category labels 未提供给感知模型")
    return manifest, rgb


def sample_frame_indices(frame_count: int, maximum: int) -> list[int]:
    if frame_count < 1 or maximum < 1:
        raise ValueError("frame_count and maximum must be positive")
    if frame_count <= maximum:
        return list(range(frame_count))
    values = {
        round(index * (frame_count - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return sorted(values)


def normalize_label(value: str) -> str:
    label = re.sub(r"\s+", " ", value.strip().casefold())
    label = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", label)
    label = re.sub(r"^(a|an|the)\s+", "", label)
    words = label.split()
    if len(words) > 1 and words[0] in {
        "black",
        "blue",
        "brown",
        "computer",
        "gray",
        "grey",
        "red",
        "studio",
        "white",
        "wooden",
        "yellow",
    }:
        label = " ".join(words[1:])
    if label.endswith("ies") and len(label) > 4:
        label = label[:-3] + "y"
    elif label.endswith("s") and not label.endswith(("ss", "us")) and len(label) > 3:
        label = label[:-1]
    return label


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def aggregate_detections(
    detections: Iterable[dict[str, Any]],
    *,
    image_size: tuple[int, int],
    min_frame_occurrences: int = 2,
    max_objects: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = image_size
    image_area = float(width * height)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_frame_label: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for raw in detections:
        label = normalize_label(str(raw.get("label", "")))
        bbox = [float(value) for value in raw.get("bbox", [])]
        if (
            not label
            or len(label) > 64
            or label in STRUCTURAL_LABELS
            or len(bbox) != 4
        ):
            rejected.append({**raw, "reason": "invalid_or_structural_label"})
            continue
        area_ratio = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]) / image_area
        if not 0.002 <= area_ratio <= 0.70:
            rejected.append({**raw, "reason": "bbox_area_outside_gate"})
            continue
        record = {
            "label": label,
            "frame_index": int(raw["frame_index"]),
            "bbox": bbox,
            "area_ratio": area_ratio,
            "task": str(raw.get("task", "")),
        }
        siblings = per_frame_label[(record["frame_index"], label)]
        if any(_iou(record["bbox"], item["bbox"]) >= 0.60 for item in siblings):
            continue
        siblings.append(record)
        by_label[label].append(record)

    candidates = []
    for label, items in by_label.items():
        frames = sorted({item["frame_index"] for item in items})
        if len(frames) < min_frame_occurrences:
            rejected.append(
                {
                    "label": label,
                    "frame_occurrences": len(frames),
                    "reason": "insufficient_distinct_frames",
                }
            )
            continue
        largest = max(item["area_ratio"] for item in items)
        strong = [
            item for item in items if item["area_ratio"] >= max(0.002, largest * 0.50)
        ]
        prompt_frame = min(item["frame_index"] for item in strong)
        example = max(items, key=lambda item: item["area_ratio"])
        candidates.append(
            {
                "label": label,
                "sam3_prompt": label,
                "prompt_frame": prompt_frame,
                "frame_occurrences": len(frames),
                "raw_detection_count": len(items),
                "median_area_ratio": sorted(item["area_ratio"] for item in items)[
                    len(items) // 2
                ],
                "example": {
                    "frame_index": example["frame_index"],
                    "bbox": example["bbox"],
                    "task": example["task"],
                },
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["frame_occurrences"],
            -item["median_area_ratio"],
            item["label"],
        )
    )
    return candidates[:max_objects], rejected


def _florence_backend(model_path: Path):
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    def infer(image_path: Path, task: str) -> list[dict[str, Any]]:
        image = Image.open(image_path).convert("RGB")
        views = [("full", image, 0, 0)]
        if task == "<DENSE_REGION_CAPTION>":
            crop_width = round(image.width * 0.62)
            crop_height = round(image.height * 0.72)
            for name, left, top in (
                ("top_left", 0, 0),
                ("top_right", image.width - crop_width, 0),
                ("bottom_left", 0, image.height - crop_height),
                (
                    "bottom_right",
                    image.width - crop_width,
                    image.height - crop_height,
                ),
            ):
                views.append(
                    (
                        name,
                        image.crop((left, top, left + crop_width, top + crop_height)),
                        left,
                        top,
                    )
                )
        detections = []
        for view_name, view, offset_x, offset_y in views:
            inputs = processor(text=task, images=view, return_tensors="pt")
            inputs = {
                key: (
                    value.to(device, dtype=dtype)
                    if key == "pixel_values"
                    else value.to(device)
                )
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=1,
                    do_sample=False,
                    use_cache=False,
                )
            generated = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]
            parsed = processor.post_process_generation(
                generated,
                task=task,
                image_size=(view.width, view.height),
            )[task]
            for label, bbox in zip(
                parsed.get("labels", []),
                parsed.get("bboxes", []),
            ):
                detections.append(
                    {
                        "label": label,
                        "bbox": [
                            float(bbox[0]) + offset_x,
                            float(bbox[1]) + offset_y,
                            float(bbox[2]) + offset_x,
                            float(bbox[3]) + offset_y,
                        ],
                        "view": view_name,
                    }
                )
        return detections

    return infer


def run_object_discovery(
    manifest_path: Path,
    rgb_dir: Path,
    output_file: Path,
    *,
    model_path: Path,
    maximum_frames: int = 80,
    min_frame_occurrences: int = 2,
    max_objects: int = 16,
    infer: Callable[[Path, str], list[dict[str, Any]]] | None = None,
) -> dict:
    manifest, rgb = validate_survey(manifest_path, rgb_dir)
    signature = survey_signature(manifest_path, rgb_dir)
    indices = sample_frame_indices(len(rgb), maximum_frames)
    backend = infer or _florence_backend(model_path)
    raw = []
    image_size = tuple(manifest["camera"]["resolution"])
    for sample_number, frame_index in enumerate(indices, start=1):
        image_path = rgb[frame_index]
        for task in TASKS:
            for item in backend(image_path, task):
                raw.append(
                    {
                        "frame_index": frame_index,
                        "image": image_path.name,
                        "task": task,
                        "label": str(item["label"]),
                        "bbox": [float(value) for value in item["bbox"]],
                        "view": str(item.get("view", "external_backend")),
                    }
                )
        print(
            f"[Home discovery] frame {sample_number}/{len(indices)} "
            f"index={frame_index} raw={len(raw)}"
        )
    objects, rejected = aggregate_detections(
        raw,
        image_size=image_size,
        min_frame_occurrences=min_frame_occurrences,
        max_objects=max_objects,
    )
    payload = {
        "schema_version": 1,
        "pipeline_version": DISCOVERY_PIPELINE_VERSION,
        "artifact_type": "category_free_rgb_object_discovery",
        "model": {
            "name": "microsoft/Florence-2-base-ft",
            "path": str(model_path.resolve()),
            "tasks": list(TASKS),
        },
        "input": {
            "source": "g1d_rgb_survey",
            "survey_signature": signature,
            "frame_count": len(rgb),
            "sampled_frame_indices": indices,
            "rgb_only": True,
        },
        "truth_boundary": {
            "category_prompt_list_supplied": False,
            "usd_semantics_read": False,
            "scene_object_coordinates_read": False,
            "labels_generated_by_model": True,
        },
        "quality_gates": {
            "min_frame_occurrences": min_frame_occurrences,
            "bbox_area_ratio": [0.002, 0.70],
            "max_objects": max_objects,
        },
        "objects": objects,
        "raw_detections": raw,
        "rejected": rejected,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[Home discovery] accepted {len(objects)} labels: "
        + ", ".join(item["label"] for item in objects)
    )
    return payload


__all__ = [
    "TASKS",
    "DISCOVERY_PIPELINE_VERSION",
    "aggregate_detections",
    "normalize_label",
    "run_object_discovery",
    "sample_frame_indices",
    "survey_signature",
    "validate_survey",
]
