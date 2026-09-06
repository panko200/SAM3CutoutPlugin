import sys
import gc
import inspect
import json
import os
import cv2
import numpy as np
import torch
import time
import base64
from PIL import Image
from huggingface_hub import login
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

# Optional: Sam3Model for text prompt support
try:
    from transformers import Sam3Model, Sam3Processor
    _HAS_SAM3_IMAGE_MODEL = True
except ImportError:
    _HAS_SAM3_IMAGE_MODEL = False


# =========================================================
# Device / dtype helpers
# =========================================================

def select_device(device_index: int) -> str:
    if device_index == 1:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if device == "cuda":
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


# =========================================================
# API互換ヘルパー
# =========================================================

def call_init_video_session(processor, **kwargs):
    sig = inspect.signature(processor.init_video_session)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return processor.init_video_session(**filtered)


def call_add_inputs(processor, **kwargs):
    sig = inspect.signature(processor.add_inputs_to_inference_session)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return processor.add_inputs_to_inference_session(**filtered)


def normalize_masks(m, obj_ids=None) -> torch.Tensor:
    if isinstance(m, np.ndarray):
        m = torch.from_numpy(m)
    if not isinstance(m, torch.Tensor):
        raise ValueError(f"mask must be Tensor/ndarray, got: {type(m)}")

    n = len(obj_ids) if obj_ids is not None else None

    if m.ndim == 4:
        if n is not None:
            if m.shape[0] == n and m.shape[1] == 1:
                m = m[:, 0]
            elif m.shape[0] == 1 and m.shape[1] == n:
                m = m[0]
            else:
                if m.shape[0] == 1:
                    m = m[0]
                elif m.shape[1] == 1:
                    m = m[:, 0]
                else:
                    raise ValueError(f"Unexpected 4D mask shape: {tuple(m.shape)} (obj_ids={obj_ids})")
        else:
            if m.shape[0] == 1:
                m = m[0]
            elif m.shape[1] == 1:
                m = m[:, 0]
            else:
                raise ValueError(f"Unexpected 4D mask shape: {tuple(m.shape)}")
    elif m.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected mask tensor shape: {tuple(m.shape)}")

    if m.ndim != 3:
        raise ValueError(f"Normalize failed: got {tuple(m.shape)}")
    return m


def to_int_list(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
        x = x[0]
    try:
        return [int(v) for v in x]
    except Exception:
        return None


# =========================================================
# テキストプロンプト用 Sam3Model (遅延読み込み)
# =========================================================

_image_model = None
_image_processor = None


def get_image_model(device, dtype):
    global _image_model, _image_processor
    if _image_model is None:
        if not _HAS_SAM3_IMAGE_MODEL:
            raise RuntimeError("Sam3Model is not available in this transformers version.")
        print("Loading Sam3Model for text prompt support...", flush=True)
        _image_model = Sam3Model.from_pretrained(
            "facebook/sam3", torch_dtype=dtype
        ).to(device).eval()
        _image_processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("Sam3Model loaded.", flush=True)
    return _image_model, _image_processor


def _generate_initial_mask_with_text_original(img, text_prompt, points, labels, boxes, height, width, device, dtype):
    img_model, img_processor = get_image_model(device, dtype)

    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
    if not prompts:
        prompts = [""]

    images = [img] * len(prompts)
    proc_kwargs = {"images": images, "text": prompts, "return_tensors": "pt"}

    if boxes:
        proc_kwargs["input_boxes"] = [[[float(coord) for coord in box] for box in boxes]] * len(prompts)
    elif points:
        boxes_mapped = []
        for p in points:
            x, y = float(p[0]), float(p[1])
            x1 = max(0.0, x - 2.0)
            y1 = max(0.0, y - 2.0)
            x2 = min(float(width - 1), x + 2.0)
            y2 = min(float(height - 1), y + 2.0)
            boxes_mapped.append([x1, y1, x2, y2])
        proc_kwargs["input_boxes"] = [boxes_mapped] * len(prompts)
        proc_kwargs["input_boxes_labels"] = [[int(l) for l in labels]] * len(prompts)

    inputs = img_processor(**proc_kwargs).to(device=img_model.device)

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(dtype=dtype)

    with torch.inference_mode():
        outputs = img_model(**inputs)

    results = img_processor.post_process_instance_segmentation(
        outputs=outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=[[height, width]] * len(prompts),
    )

    all_masks = []
    for res in results:
        m = res["masks"]
        if len(m) > 0:
            combined_m = m.any(dim=0)
            all_masks.append(combined_m.cpu())

    if all_masks:
        mask = torch.stack(all_masks).any(dim=0).numpy().astype(np.uint8)
    else:
        mask = np.zeros((height, width), dtype=np.uint8)

    return mask


def _generate_initial_mask_with_text_best_filtered(img, text_prompt, points, labels, boxes, height, width, device, dtype, is_fallback=False):
    img_model, img_processor = get_image_model(device, dtype)

    use_text = text_prompt.strip() and not is_fallback
    prompts = [p.strip() for p in text_prompt.split(",") if p.strip()] if use_text else [""]

    images = [img] * len(prompts)
    proc_kwargs = {"images": images, "text": prompts, "return_tensors": "pt"}

    if boxes:
        proc_kwargs["input_boxes"] = [[[float(coord) for coord in box] for box in boxes]] * len(prompts)
    elif points:
        boxes_mapped = []
        for p in points:
            x, y = float(p[0]), float(p[1])
            x1 = max(0.0, x - 2.0)
            y1 = max(0.0, y - 2.0)
            x2 = min(float(width - 1), x + 2.0)
            y2 = min(float(height - 1), y + 2.0)
            boxes_mapped.append([x1, y1, x2, y2])
        proc_kwargs["input_boxes"] = [boxes_mapped] * len(prompts)
        proc_kwargs["input_boxes_labels"] = [[int(l) for l in labels]] * len(prompts)

    inputs = img_processor(**proc_kwargs).to(device=img_model.device)

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            inputs[k] = v.to(dtype=dtype)

    with torch.inference_mode():
        outputs = img_model(**inputs)

    results = img_processor.post_process_instance_segmentation(
        outputs=outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=[[height, width]] * len(prompts),
    )

    selected_masks = []
    for res in results:
        pred_masks = res.get("masks", [])
        pred_boxes = res.get("boxes", [])
        pred_scores = res.get("scores", [])
        
        if len(pred_masks) == 0:
            continue
            
        num_preds = len(pred_masks)
        
        if isinstance(pred_masks, list):
            pred_masks_np = np.array([m.cpu().numpy() if hasattr(m, "cpu") else np.array(m) for m in pred_masks])
        elif torch.is_tensor(pred_masks):
            pred_masks_np = pred_masks.cpu().numpy()
        else:
            pred_masks_np = np.array(pred_masks)
            
        if isinstance(pred_boxes, list):
            pred_boxes_list = [b.cpu().tolist() if hasattr(b, "cpu") else b for b in pred_boxes]
        elif torch.is_tensor(pred_boxes):
            pred_boxes_list = pred_boxes.cpu().tolist()
        else:
            pred_boxes_list = [b.tolist() if hasattr(b, "tolist") else b for b in pred_boxes]
            
        if isinstance(pred_scores, list):
            pred_scores_list = [float(s.cpu().item() if hasattr(s, "cpu") else s) for s in pred_scores]
        elif torch.is_tensor(pred_scores):
            pred_scores_list = pred_scores.cpu().tolist()
        else:
            pred_scores_list = [float(s) for s in pred_scores]

        if boxes:
            for user_box in boxes:
                best_idx = -1
                best_score = -1.0
                u_x1, u_y1, u_x2, u_y2 = user_box
                u_area = (u_x2 - u_x1) * (u_y2 - u_y1)
                if u_area <= 0:
                    continue
                    
                for idx in range(num_preds):
                    p_x1, p_y1, p_x2, p_y2 = pred_boxes_list[idx]
                    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
                    if p_area <= 0:
                        continue
                        
                    ix1 = max(u_x1, p_x1)
                    iy1 = max(u_y1, p_y1)
                    ix2 = min(u_x2, p_x2)
                    iy2 = min(u_y2, p_y2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    
                    iou = inter / (u_area + p_area - inter) if (u_area + p_area - inter) > 0 else 0
                    io_user = inter / u_area
                    
                    if iou > 0.15 or io_user > 0.4:
                        score = pred_scores_list[idx]
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                            
                if best_idx != -1:
                    selected_masks.append(pred_masks_np[best_idx])

        elif points:
            pos_pts = [points[i] for i in range(len(points)) if labels[i] == 1]
            neg_pts = [points[i] for i in range(len(points)) if labels[i] == 0]
            
            candidates = []
            for idx in range(num_preds):
                mask = pred_masks_np[idx]
                pos_ok = True
                for pt in pos_pts:
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    if px < 0 or px >= width or py < 0 or py >= height or not mask[py, px]:
                        pos_ok = False
                        break
                if not pos_ok:
                    continue
                    
                neg_ok = True
                for pt in neg_pts:
                    px, py = int(round(pt[0])), int(round(pt[1]))
                    if 0 <= px < width and 0 <= py < height and mask[py, px]:
                        neg_ok = False
                        break
                if not neg_ok:
                    continue
                    
                candidates.append((idx, pred_scores_list[idx]))
                
            if candidates:
                best_idx = max(candidates, key=lambda x: x[1])[0]
                selected_masks.append(pred_masks_np[best_idx])

    if selected_masks:
        combined = np.any(selected_masks, axis=0)
        return combined.astype(np.uint8)

    if use_text and (boxes or points):
        print(f"[SAM3 Python] '{text_prompt}' matches no objects in the box/points. Retrying with visual-only segmentation.", flush=True)
        return _generate_initial_mask_with_text_best_filtered(
            img, text_prompt, points, labels, boxes, height, width, device, dtype, is_fallback=True
        )

    return np.zeros((height, width), dtype=np.uint8)


def generate_initial_mask_with_text(img, text_prompt, points, labels, boxes, height, width, device, dtype, text_mode="all"):
    if not boxes and not points:
        img_model, img_processor = get_image_model(device, dtype)

        prompts = [p.strip() for p in text_prompt.split(",") if p.strip()]
        if not prompts:
            prompts = [""]

        images = [img] * len(prompts)
        proc_kwargs = {"images": images, "text": prompts, "return_tensors": "pt"}

        inputs = img_processor(**proc_kwargs).to(device=img_model.device)

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                inputs[k] = v.to(dtype=dtype)

        with torch.inference_mode():
            outputs = img_model(**inputs)

        results = img_processor.post_process_instance_segmentation(
            outputs=outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=[[height, width]] * len(prompts),
        )

        if text_mode == "all":
            all_masks = []
            for res in results:
                m = res["masks"]
                if len(m) > 0:
                    combined_m = m.any(dim=0)
                    all_masks.append(combined_m.cpu())
            if all_masks:
                return torch.stack(all_masks).any(dim=0).numpy().astype(np.uint8)
            else:
                return np.zeros((height, width), dtype=np.uint8)
        else:
            best_score = -1.0
            best_mask = None
            for res in results:
                m = res["masks"]
                scores = res["scores"]
                if len(m) > 0:
                    max_idx = torch.argmax(scores).item()
                    score = scores[max_idx].item()
                    if score > best_score:
                        best_score = score
                        best_mask = m[max_idx].cpu()
            if best_mask is not None:
                return best_mask.numpy().astype(np.uint8)
            else:
                return np.zeros((height, width), dtype=np.uint8)

    if text_mode == "all":
        return _generate_initial_mask_with_text_original(img, text_prompt, points, labels, boxes, height, width, device, dtype)
    else:
        return _generate_initial_mask_with_text_best_filtered(img, text_prompt, points, labels, boxes, height, width, device, dtype)


def process_text_prompt(frames_pil, text_prompt, points, labels, boxes, output_dir,
                        height, width, device, dtype, action, output_format, fps, text_mode="all", request_id=""):
    img = frames_pil[0]
    mask = generate_initial_mask_with_text(img, text_prompt, points, labels, boxes, height, width, device, dtype, text_mode)

    if action == "preview":
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[mask > 0] = [255, 100, 50, 128]

        out_path = os.path.join(output_dir, "preview_mask.png")
        is_success, im_buf_arr = cv2.imencode(".png", overlay)
        if is_success:
            im_buf_arr.tofile(out_path)
        print(f"PREVIEW_RESPONSE:{request_id}:{out_path}", flush=True)

    elif action == "track":
        timestamp = int(time.time())
        output_filename = f"cutout_{timestamp}"

        if output_format in ("MaskMP4", "MaskPNG"):
            final_frame = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
        else:
            frame_rgb = np.array(frames_pil[0])
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            b, g, r_ch = cv2.split(frame_bgr)
            b_m = cv2.bitwise_and(b, b, mask=mask)
            g_m = cv2.bitwise_and(g, g, mask=mask)
            r_m = cv2.bitwise_and(r_ch, r_ch, mask=mask)
            rgba = [b_m, g_m, r_m, mask * 255]
            final_frame = cv2.merge(rgba)

        out_path = os.path.join(output_dir, f"{output_filename}.png")
        is_success, im_buf_arr = cv2.imencode(".png", final_frame)
        if is_success:
            im_buf_arr.tofile(out_path)
        print(f"OUTPUT_IMAGE:{request_id}:{out_path}", flush=True)


# =========================================================
# マスクユーティリティ
# =========================================================

def _decode_mask_value(m, h=None, w=None):
    if m is None:
        return None
    if isinstance(m, tuple) and len(m) >= 4 and m[0] == "p":
        _, ph, pw, bits = m[0], m[1], m[2], m[3]
        arr = np.unpackbits(np.ascontiguousarray(bits), axis=1)
        if pw is not None:
            arr = arr[:, :pw]
        return np.ascontiguousarray(arr).astype(np.uint8)
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    arr = np.asarray(m)
    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    return np.ascontiguousarray(arr).astype(np.uint8)


def _encode_mask_value(mask_u8):
    arr = np.ascontiguousarray((np.asarray(mask_u8) > 0).astype(np.uint8))
    h, w = arr.shape
    bits = np.packbits(arr, axis=1)
    return ("p", h, w, bits)


def get_union_mask(masks_by_frame, frame_idx, h, w):
    masks = masks_by_frame.get(frame_idx, {})
    union = None
    for m in masks.values():
        mm = _decode_mask_value(m, h, w)
        if mm is None:
            continue
        mm = (mm > 0).astype(np.uint8)
        if mm.shape != (h, w):
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
        union = mm if union is None else (union | mm)
    if union is None:
        union = np.zeros((h, w), dtype=np.uint8)
    return union.astype(bool)


MASK_COLORS = [
    [255, 100, 0, 128],
    [0, 0, 255, 128],
    [0, 200, 0, 128],
    [0, 255, 255, 128],
    [255, 0, 255, 128],
    [220, 220, 0, 128]
]


def save_preview_mask_multi_object(masks_by_frame, frame_idx, height, width):
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    masks_for_frame = masks_by_frame.get(frame_idx, {})
    
    for oid, mask in sorted(masks_for_frame.items()):
        mm = _decode_mask_value(mask, height, width)
        if mm is None:
            continue
        mm = (mm > 0).astype(np.uint8)
        if mm.shape != (height, width):
            mm = cv2.resize(mm, (width, height), interpolation=cv2.INTER_NEAREST)
        color_idx = (int(oid) - 1) % len(MASK_COLORS)
        color = MASK_COLORS[color_idx]
        overlay[mm > 0] = color

    _, mask_buf = cv2.imencode(".png", overlay)
    return base64.b64encode(mask_buf).decode("utf-8")


def force_release_ram():
    try:
        import ctypes
        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
    except Exception:
        pass


# =========================================================
# 伝播ヘルパー
# =========================================================

def run_propagation_single_direction(model, processor, session, frames_pil, masks_by_frame, start_frame_idx=0, target_frame_idx=None, reverse=False):
    seen_frames = {start_frame_idx}

    with torch.inference_mode():
        propagate_fn = model.propagate_in_video_iterator
        kwargs = {
            "inference_session": session,
            "start_frame_idx": start_frame_idx,
            "reverse": reverse
        }

        try:
            sig = inspect.signature(propagate_fn)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            supports_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in sig.parameters.values()
            )
            if "start_frame_idx" not in sig.parameters and not supports_var_kwargs:
                kwargs.pop("start_frame_idx", None)
            if "reverse" not in sig.parameters and not supports_var_kwargs:
                kwargs.pop("reverse", None)

        try:
            iterator = propagate_fn(**kwargs)
        except TypeError:
            kwargs.pop("reverse", None)
            try:
                iterator = propagate_fn(**kwargs)
            except TypeError:
                kwargs.pop("start_frame_idx", None)
                iterator = propagate_fn(**kwargs)

        for out in iterator:
            frame_idx = int(out.frame_idx)
            if reverse and frame_idx > start_frame_idx:
                continue
            if not reverse and frame_idx < start_frame_idx:
                continue

            video_res_masks = processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[[session.video_height, session.video_width]],
            )[0]

            masks3 = normalize_masks(video_res_masks)
            prop_obj_ids = to_int_list(getattr(out, "object_ids", None))

            if prop_obj_ids is None or len(prop_obj_ids) != masks3.shape[0]:
                sess_obj_ids = [int(x) for x in getattr(session, "obj_ids", [])]
                prop_obj_ids = sess_obj_ids[:masks3.shape[0]]

            frame_merged = {}
            for i, temp_id in enumerate(prop_obj_ids):
                mask_2d = (masks3[i] > 0.0).to(torch.uint8).cpu().numpy()
                obj_id = temp_id // 1000
                if obj_id in frame_merged:
                    frame_merged[obj_id] = frame_merged[obj_id] | mask_2d
                else:
                    frame_merged[obj_id] = mask_2d

            masks_for_frame = masks_by_frame.setdefault(frame_idx, {})
            for obj_id, mask_2d in frame_merged.items():
                masks_for_frame[obj_id] = mask_2d

            seen_frames.add(frame_idx)
            if target_frame_idx is not None and frame_idx == target_frame_idx:
                break

    return seen_frames


def run_propagation_both_directions(model, processor, session, frames_pil, masks_by_frame, start_frame_idx=0):
    total = len(frames_pil)
    seen_frames = {start_frame_idx}

    with torch.inference_mode():
        for reverse in (False, True):
            propagate_fn = model.propagate_in_video_iterator
            kwargs = {
                "inference_session": session,
                "start_frame_idx": start_frame_idx,
                "reverse": reverse
            }

            try:
                sig = inspect.signature(propagate_fn)
            except (TypeError, ValueError):
                sig = None

            if sig is not None:
                supports_var_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in sig.parameters.values()
                )
                if "start_frame_idx" not in sig.parameters and not supports_var_kwargs:
                    kwargs.pop("start_frame_idx", None)
                if "reverse" not in sig.parameters and not supports_var_kwargs:
                    kwargs.pop("reverse", None)

            try:
                iterator = propagate_fn(**kwargs)
            except TypeError:
                kwargs.pop("reverse", None)
                try:
                    iterator = propagate_fn(**kwargs)
                except TypeError:
                    kwargs.pop("start_frame_idx", None)
                    iterator = propagate_fn(**kwargs)

            for out in iterator:
                frame_idx = int(out.frame_idx)
                if reverse and frame_idx > start_frame_idx:
                    continue
                if not reverse and frame_idx < start_frame_idx:
                    continue

                video_res_masks = processor.post_process_masks(
                    [out.pred_masks],
                    original_sizes=[[session.video_height, session.video_width]],
                )[0]

                masks3 = normalize_masks(video_res_masks)
                prop_obj_ids = to_int_list(getattr(out, "object_ids", None))

                if prop_obj_ids is None or len(prop_obj_ids) != masks3.shape[0]:
                    sess_obj_ids = [int(x) for x in getattr(session, "obj_ids", [])]
                    prop_obj_ids = sess_obj_ids[:masks3.shape[0]]

                frame_merged = {}
                for i, temp_id in enumerate(prop_obj_ids):
                    mask_2d = (masks3[i] > 0.0).to(torch.uint8).cpu().numpy()
                    obj_id = temp_id // 1000
                    if obj_id in frame_merged:
                        frame_merged[obj_id] = frame_merged[obj_id] | mask_2d
                    else:
                        frame_merged[obj_id] = mask_2d

                masks_for_frame = masks_by_frame.setdefault(frame_idx, {})
                for obj_id, mask_2d in frame_merged.items():
                    masks_for_frame[obj_id] = mask_2d

                seen_frames.add(frame_idx)
                covered = len(seen_frames)
                if covered % max(1, total // 20) == 0 or covered == total:
                    print(f"PROGRESS:{covered}/{total}", flush=True)

    return seen_frames


# =========================================================
# チャンク方式（省メモリ）サポート
# =========================================================

MODE_LEGACY = "legacy"
MODE_CHUNK = "chunk"
MODE_AUTO = "auto"

CHUNK_FRAMES = int(os.environ.get("SAM3_CHUNK_FRAMES", "180"))
CHUNK_OVERLAP = int(os.environ.get("SAM3_CHUNK_OVERLAP", "48"))
AUTO_LEGACY_MAX_FRAMES = int(os.environ.get("SAM3_AUTO_LEGACY_MAX", "240"))
_MEMLOG = os.environ.get("SAM3_MEMLOG", "0") in ("1", "true", "True")


class _ChunkFallback(Exception):
    pass


def _memlog(tag: str) -> None:
    if not _MEMLOG:
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        PROCESS_QUERY_INFORMATION = 0x0400
        h = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION, False, ctypes.windll.kernel32.GetCurrentProcessId()
        )
        if not h:
            return
        try:
            r = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
        if not r:
            return
        print(f"[MEMLOG:{tag}] rss={pmc.WorkingSetSize / 1048576.0:.1f}MB peak={pmc.PeakWorkingSetSize / 1048576.0:.1f}MB", file=sys.stderr, flush=True)
    except Exception:
        pass


class _SegmentFrameReader:
    def __init__(self, video_path: str, start_sec: float, end_sec: float):
        self.video_path = str(video_path)
        self.start_sec = float(start_sec)
        self.end_sec = float(end_sec)
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.positions = None
        self._cap = None
        self._cur = -1

    def probe_and_scan(self) -> int:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps) if fps and fps > 0 else 30.0
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            positions: list = []
            use_trim = total_frames > 1 and self.end_sec > self.start_sec
            if use_trim:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, self.start_sec) * 1000.0)
                start_frame_guess = int(round(self.start_sec * self.fps))
                end_frame_guess = int(round(self.end_sec * self.fps))
                cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                if cur_idx < start_frame_guess - 2:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_guess)
                while True:
                    cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    if cur_frame >= end_frame_guess:
                        break
                    ret, _frame = cap.read()
                    if not ret:
                        break
                    t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if t_msec and (t_msec / 1000.0) > self.end_sec:
                        break
                    positions.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))
            else:
                while True:
                    ret, _frame = cap.read()
                    if not ret:
                        break
                    positions.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0))
        finally:
            cap.release()

        if not positions:
            raise RuntimeError("No frames found in segment.")
        if self.width <= 0 or self.height <= 0:
            p0 = max(0, int(positions[0]) - 1)
            cap = cv2.VideoCapture(self.video_path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, p0)
                ret, frame0 = cap.read()
                if ret:
                    hh, ww = frame0.shape[:2]
                    self.height, self.width = int(hh), int(ww)
            finally:
                cap.release()
        self.positions = np.asarray(positions, dtype=np.int64)
        return int(len(positions))

    def _open(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video_path)
            self._cur = -1
            if self._cap is None or not self._cap.isOpened():
                raise RuntimeError(f"Failed to open segment video: {self.video_path}")

    def get(self, idx: int) -> Image.Image:
        if self.positions is None:
            raise RuntimeError("reader is not scanned")
        n = len(self.positions)
        i = int(idx)
        if i < 0:
            i = 0
        if i >= n:
            i = n - 1
        self._open()
        frame_bgr = None
        if self._cur >= 0 and i == self._cur + 1 and self._cap is not None:
            ret, frame_bgr = self._cap.read()
        else:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(self.positions[i]) - 1))
            ret, frame_bgr = self._cap.read()
            if not ret or frame_bgr is None:
                self._cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (float(i) / self.fps)) * 1000.0)
                ret, frame_bgr = self._cap.read()
        if not ret or frame_bgr is None:
            raise RuntimeError(f"Failed to decode frame at index={i}")
        self._cur = i
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def close(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


class _ChunkEngine:
    def __init__(self, model, processor, reader: _SegmentFrameReader, device: str, dtype,
                 height: int, width: int, fps: float, text_initial_mask_fn=None):
        self.model = model
        self.processor = processor
        self.reader = reader
        self.device = device
        self.dtype = dtype
        self.height = int(height)
        self.width = int(width)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.total = int(len(reader.positions)) if reader.positions is not None else 0
        self.text_initial_mask_fn = text_initial_mask_fn

        self.chunk = CHUNK_FRAMES
        self.overlap = CHUNK_OVERLAP

        self.session = None
        self.session_ws = -1
        self.session_we = -1

        self.masks = {}
        self.objects = None
        self.text_prompt = ""
        self.text_mode = "all"
        self.anchor = None
        self._baked = False
        self._cov_lo = None
        self._cov_hi = None
        self._last_report = 0
        self._report_every = max(1, self.total // 20) if self.total else 1

    def _release_session(self):
        self.session = None
        self.session_ws = -1
        self.session_we = -1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_window_session(self, ws: int, we: int):
        if self.session is not None and self.session_ws == ws and self.session_we == we:
            return
        self._release_session()
        if we <= ws:
            raise RuntimeError(f"invalid window [{ws},{we})")
        dev = torch.device(self.device)
        sess = call_init_video_session(
            self.processor,
            video=None,
            inference_device=dev,
            processing_device="cpu",
            video_storage_device="cpu",
            inference_state_device="cpu",
            dtype=self.dtype,
        )
        if hasattr(sess, "inference_device"):
            sess.inference_device = dev
        if hasattr(sess, "cache") and hasattr(sess.cache, "inference_device"):
            sess.cache.inference_device = dev
        try:
            if hasattr(sess, "video_height"):
                sess.video_height = int(self.height)
            if hasattr(sess, "video_width"):
                sess.video_width = int(self.width)
        except Exception:
            pass
        if not hasattr(sess, "add_new_frame"):
            raise _ChunkFallback("SAM3 video session does not support add_new_frame (chunk mode unavailable)")

        for local, gi in enumerate(range(ws, we)):
            pil = self.reader.get(gi)
            batch = self.processor.video_processor.preprocess(videos=[pil], return_tensors="pt")
            pv = batch["pixel_values_videos"][0, 0]
            sess.add_new_frame(pv, frame_idx=int(local))
            if (local + 1) % 60 == 0:
                gc.collect()
        self.session = sess
        self.session_ws = int(ws)
        self.session_we = int(we)
        _memlog(f"window[{ws},{we})")

    def _has_mask_at(self, frame_idx: int) -> bool:
        mf = self.masks.get(int(frame_idx))
        if not mf:
            return False
        for v in mf.values():
            arr = _decode_mask_value(v)
            if arr is not None and np.any(arr > 0):
                return True
        return False

    def _store_merged(self, frame_idx: int, out_obj_ids, masks_norm) -> None:
        frame_idx = int(frame_idx)
        mf = self.masks.setdefault(frame_idx, {})
        stored_any = False
        for k, temp_id in enumerate(out_obj_ids):
            if k >= masks_norm.shape[0]:
                break
            obj_id = int(temp_id) // 1000
            mask_2d = (masks_norm[k] > 0.0).to(torch.uint8).cpu().numpy()
            prev = mf.get(obj_id)
            if prev is not None:
                prev_arr = _decode_mask_value(prev)
                if prev_arr is not None and prev_arr.shape == mask_2d.shape:
                    mask_2d = (prev_arr | mask_2d).astype(np.uint8)
            mf[obj_id] = _encode_mask_value(mask_2d)
            if np.any(mask_2d > 0):
                stored_any = True
        if stored_any:
            if self._cov_lo is None:
                self._cov_lo = frame_idx
            else:
                self._cov_lo = min(self._cov_lo, frame_idx)
            if self._cov_hi is None:
                self._cov_hi = frame_idx
            else:
                self._cov_hi = max(self._cov_hi, frame_idx)

    def _mask_bbox(self, frame_idx: int, obj_id: int, pad: int = 2):
        mf = self.masks.get(int(frame_idx))
        if not mf:
            return None
        arr = _decode_mask_value(mf.get(int(obj_id)))
        if arr is None or not np.any(arr > 0):
            return None
        ys, xs = np.where(arr > 0)
        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(self.width - 1, int(xs.max()) + pad)
        y2 = min(self.height - 1, int(ys.max()) + pad)
        if x2 <= x1:
            x2 = min(self.width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(self.height - 1, y1 + 1)
        return [x1, y1, x2, y2]

    def _union_mask(self, frame_idx: int) -> np.ndarray:
        return get_union_mask(self.masks, frame_idx, self.height, self.width)

    def _accept_prompt(self, frame_index: int, objects, text_prompt: str, text_mode: str) -> bool:
        changed = objects != self.objects or text_prompt != self.text_prompt or text_mode != self.text_mode
        if changed or self.anchor is None:
            self.objects = dict(objects) if objects else {}
            self.text_prompt = text_prompt or ""
            self.text_mode = text_mode or "all"
            self.anchor = int(frame_index)
            self._baked = False
            self.masks.clear()
            self._cov_lo = None
            self._cov_hi = None
            self._release_session()
            return True
        return False

    def _register_prompt_into_window(self, local_frame: int):
        sess = self.session
        processor = self.processor
        frame_g = self.session_ws + int(local_frame)
        objects = self.objects or {}
        text_prompt = self.text_prompt
        text_mode = self.text_mode
        h, w = self.height, self.width

        temp_queries = []
        for obj_id_str, obj_data in objects.items():
            obj_id = int(obj_id_str)
            pts = obj_data.get("points", []) or []
            lbls = obj_data.get("labels", []) or []
            bxs = obj_data.get("boxes", []) or []
            if bxs:
                for idx, box in enumerate(bxs):
                    temp_id = obj_id * 1000 + idx
                    query_pts = pts if idx == 0 else []
                    query_lbls = lbls if idx == 0 else []
                    temp_queries.append({
                        "temp_id": temp_id,
                        "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "points": [[int(p[0]), int(p[1])] for p in query_pts] if query_pts else None,
                        "labels": [int(l) for l in query_lbls] if query_lbls else None,
                    })
            elif pts:
                temp_queries.append({
                    "temp_id": obj_id * 1000,
                    "box": None,
                    "points": [[int(p[0]), int(p[1])] for p in pts],
                    "labels": [int(l) for l in lbls],
                })
            else:
                temp_queries.append({
                    "temp_id": obj_id * 1000,
                    "box": None,
                    "points": None,
                    "labels": None,
                })

        is_first = True
        all_boxes_list = []
        all_box_obj_ids = []
        for q in temp_queries:
            if q["box"] is not None:
                all_boxes_list.append(q["box"])
                all_box_obj_ids.append(q["temp_id"])
        if all_boxes_list:
            call_add_inputs(
                processor,
                inference_session=sess,
                frame_idx=int(local_frame),
                obj_ids=all_box_obj_ids,
                input_boxes=[all_boxes_list],
                clear_old_inputs=is_first,
            )
            is_first = False

        all_points_list = []
        all_labels_list = []
        all_point_obj_ids = []
        for q in temp_queries:
            if q["points"] is not None:
                all_points_list.append(q["points"])
                all_labels_list.append(q["labels"])
                all_point_obj_ids.append(q["temp_id"])
        if all_points_list:
            call_add_inputs(
                processor,
                inference_session=sess,
                frame_idx=int(local_frame),
                obj_ids=all_point_obj_ids,
                input_points=[all_points_list],
                input_labels=[all_labels_list],
                clear_old_inputs=is_first,
            )
            is_first = False

        if text_prompt.strip() and self.text_initial_mask_fn is not None:
            pil_frame = self.reader.get(frame_g)
            for q in temp_queries:
                q_pts = q["points"] if q["points"] else []
                q_lbls = q["labels"] if q["labels"] else []
                q_bxs = [q["box"]] if q["box"] else []
                init_mask = self.text_initial_mask_fn(
                    pil_frame, text_prompt, q_pts, q_lbls, q_bxs,
                    h, w, self.device, self.dtype, text_mode,
                )
                call_add_inputs(
                    processor,
                    inference_session=sess,
                    frame_idx=int(local_frame),
                    obj_ids=[q["temp_id"]],
                    input_masks=init_mask,
                    clear_old_inputs=is_first,
                )
                is_first = False

        obj_ids_list = [q["temp_id"] for q in temp_queries]
        if not obj_ids_list:
            obj_ids_list = [1000]
        sess.obj_with_new_inputs = obj_ids_list
        with torch.inference_mode():
            outputs = self.model(inference_session=sess, frame_idx=int(local_frame))
        sess.obj_with_new_inputs = []

        processed = self.processor.post_process_masks(
            [outputs.pred_masks],
            [[self.height, self.width]],
            binarize=False,
        )[0]
        out_obj_ids = to_int_list(getattr(outputs, "object_ids", None))
        if out_obj_ids is None:
            out_obj_ids = [q["temp_id"] for q in temp_queries]
        masks_norm = normalize_masks(processed, out_obj_ids)

        mf = self.masks.setdefault(frame_g, {})
        for oid in objects.keys():
            if int(oid) not in mf:
                mf[int(oid)] = _encode_mask_value(np.zeros((h, w), dtype=np.uint8))
        self._store_merged(frame_g, out_obj_ids, masks_norm)

    # ★修正：前ウィンドウの追跡結果マスクそのものを注入（BBoxへの縮退による背景誤認識・反転を根絶）
    def _seed_window(self, ws: int, seed_global_frame: int) -> bool:
        sess = self.session
        if sess is None:
            return False
        local = int(seed_global_frame) - self.session_ws
        if local < 0 or local >= (self.session_we - self.session_ws):
            return False
        mf = self.masks.get(int(seed_global_frame))
        if not mf:
            return False

        masks_list = []
        obj_ids = []
        for oid in sorted(mf.keys()):
            arr = _decode_mask_value(mf.get(int(oid)), self.height, self.width)
            if arr is not None and np.any(arr > 0):
                masks_list.append((arr > 0).astype(np.uint8))
                obj_ids.append(int(oid) * 1000)

        if not masks_list:
            return False

        try:
            call_add_inputs(
                self.processor,
                inference_session=sess,
                frame_idx=int(local),
                obj_ids=obj_ids,
                input_masks=masks_list,
                clear_old_inputs=True,
            )
        except Exception:
            # 万が一の環境差互換フォールバック
            boxes = [self._mask_bbox(int(seed_global_frame), oid // 1000) for oid in obj_ids]
            valid_boxes = [b for b in boxes if b is not None]
            if not valid_boxes:
                return False
            call_add_inputs(
                self.processor,
                inference_session=sess,
                frame_idx=int(local),
                obj_ids=obj_ids[:len(valid_boxes)],
                input_boxes=[valid_boxes],
                clear_old_inputs=True,
            )

        sess.obj_with_new_inputs = list(obj_ids)
        with torch.inference_mode():
            outputs = self.model(inference_session=sess, frame_idx=int(local))
        sess.obj_with_new_inputs = []
        processed = self.processor.post_process_masks(
            [outputs.pred_masks],
            [[self.height, self.width]],
            binarize=False,
        )[0]
        out_obj_ids = to_int_list(getattr(outputs, "object_ids", None))
        if out_obj_ids is None:
            out_obj_ids = list(obj_ids)
        masks_norm = normalize_masks(processed, out_obj_ids)
        self._store_merged(int(seed_global_frame), out_obj_ids, masks_norm)
        return True

    def _propagate_window(self, ws: int, we: int, start_local: int, reverse: bool, report: bool = False) -> int:
        sess = self.session
        if sess is None:
            return 0
        with torch.inference_mode():
            propagate_fn = self.model.propagate_in_video_iterator
            kwargs = {
                "inference_session": sess,
                "start_frame_idx": int(start_local),
                "reverse": bool(reverse),
            }
            try:
                sig = inspect.signature(propagate_fn)
            except (TypeError, ValueError):
                sig = None
            if sig is not None:
                supports_var_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
                )
                if "start_frame_idx" not in sig.parameters and not supports_var_kwargs:
                    kwargs.pop("start_frame_idx", None)
                if "reverse" not in sig.parameters and not supports_var_kwargs:
                    kwargs.pop("reverse", None)
            try:
                iterator = propagate_fn(**kwargs)
            except TypeError:
                kwargs.pop("reverse", None)
                try:
                    iterator = propagate_fn(**kwargs)
                except TypeError:
                    kwargs.pop("start_frame_idx", None)
                    iterator = propagate_fn(**kwargs)

            before = len(self.masks)
            for out in iterator:
                frame_g = ws + int(out.frame_idx)
                if frame_g < ws or frame_g >= we:
                    continue
                video_res_masks = self.processor.post_process_masks(
                    [out.pred_masks],
                    original_sizes=[[self.height, self.width]],
                )[0]
                masks3 = normalize_masks(video_res_masks)
                out_ids = to_int_list(getattr(out, "object_ids", None))
                if out_ids is None or len(out_ids) != masks3.shape[0]:
                    sess_obj_ids = [int(x) for x in getattr(sess, "obj_ids", [])]
                    out_ids = sess_obj_ids[: masks3.shape[0]]
                self._store_merged(frame_g, out_ids, masks3)
                if report and len(self.masks) - before > 0:
                    covered = len(self.masks)
                    if covered >= self._last_report + self._report_every or covered >= self.total:
                        self._last_report = covered
                        print(f"PROGRESS:{min(covered, self.total)}/{self.total}", flush=True)
            return len(self.masks) - before

    def _bake(self):
        if self._baked:
            return
        if self.anchor is None:
            return
        anchor = int(self.anchor)
        n = self.total
        we = min(n, anchor + 1)
        if we > anchor:
            self._build_window_session(anchor, we)
            self._register_prompt_into_window(0)
        self._baked = True
        _memlog("bake-min")

    def _register_entry(self, ws: int, we: int, entry_global: int) -> bool:
        if entry_global == self.anchor and self.anchor is not None:
            local = entry_global - ws
            self._register_prompt_into_window(local)
            return True
        return self._seed_window(ws, entry_global)

    # ★修正：対称化（前進時に _cov_hi が動かなかった場合即時打ち切り）
    def _roll_forward(self, target: int = None, report: bool = False):
        n = self.total
        limit = n - 1 if target is None else min(n - 1, int(target))
        guard = 0
        while (self._cov_hi is not None and self._cov_hi < limit):
            guard += 1
            if guard > 10000:
                break
            ws = int(self._cov_hi)
            max_we = min(n, ws + self.chunk + self.overlap)
            we = min(max_we, limit + 1)
            if we - ws < 2:
                break
            self._build_window_session(ws, we)
            if not self._register_entry(ws, we, ws):
                break
            before_cov_hi = self._cov_hi
            advanced = self._propagate_window(ws, we, 0, reverse=False, report=report)
            if advanced <= 0 or self._cov_hi is None or self._cov_hi <= before_cov_hi:
                break
        if report and self._cov_hi is not None:
            covered = min(int(self._cov_hi) + 1, n)
            print(f"PROGRESS:{covered}/{n}", flush=True)

    def _roll_backward(self, target: int = None, report: bool = False):
        limit = 0 if target is None else max(0, int(target))
        guard = 0
        while (self._cov_lo is not None and self._cov_lo > limit):
            guard += 1
            if guard > 10000:
                break
            we = int(self._cov_lo) + 1
            min_ws = max(0, we - self.chunk - self.overlap)
            ws = target if (target is not None and target > min_ws) else min_ws
            if we - ws < 2:
                break
            self._build_window_session(ws, we)
            seed_local = we - 1 - ws
            if not self._register_entry(ws, we, we - 1):
                break
            before_cov_lo = self._cov_lo
            advanced = self._propagate_window(ws, we, seed_local, reverse=True, report=report)
            if advanced <= 0 or self._cov_lo is None or self._cov_lo >= before_cov_lo:
                break
        if report and self._cov_hi is not None:
            covered = min(int(self._cov_hi) + 1, self.total)
            print(f"PROGRESS:{covered}/{self.total}", flush=True)

    def _render_track(self, output_dir: str, output_format: str, request_id: str, output_filename: str):
        n = self.total
        out = None
        temp_frames_dir = ""
        if output_format in ("GreenScreenMP4", "MaskMP4"):
            out_path = os.path.join(output_dir, f"{output_filename}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            valid_fps = 30.0 if self.fps <= 0 else self.fps
            out = cv2.VideoWriter(out_path, fourcc, valid_fps, (self.width, self.height))
        else:
            temp_frames_dir = os.path.join(output_dir, "temp_masks")
            os.makedirs(temp_frames_dir, exist_ok=True)
            out_path = temp_frames_dir
        for i in range(n):
            mask = self._union_mask(i)
            mask_uint8 = mask.astype(np.uint8)
            if output_format == "MaskMP4":
                final_frame = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)
                out.write(final_frame)
            elif output_format == "GreenScreenMP4":
                frame_rgb = np.array(self.reader.get(i))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                bg = np.zeros_like(frame_bgr)
                bg[:] = (0, 255, 0)
                mask_3ch = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)
                fg = cv2.bitwise_and(frame_bgr, mask_3ch)
                bg_masked = cv2.bitwise_and(bg, cv2.bitwise_not(mask_3ch))
                final_frame = cv2.add(fg, bg_masked)
                out.write(final_frame)
            else:
                frame_rgb = np.array(self.reader.get(i))
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                b, g, r_ch = cv2.split(frame_bgr)
                b_m = cv2.bitwise_and(b, b, mask=mask_uint8)
                g_m = cv2.bitwise_and(g, g, mask=mask_uint8)
                r_m = cv2.bitwise_and(r_ch, r_ch, mask=mask_uint8)
                rgba = [b_m, g_m, r_m, mask_uint8 * 255]
                final_frame = cv2.merge(rgba)
                png_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png")
                is_success, im_buf_arr = cv2.imencode(".png", final_frame)
                if is_success:
                    im_buf_arr.tofile(png_path)
            if (i + 1) % max(1, n // 20) == 0 or (i + 1) == n:
                print(f"PROGRESS:{i + 1}/{n}", flush=True)
        if out is not None:
            out.release()
            print(f"OUTPUT_VIDEO:{request_id}:{out_path}", flush=True)
        else:
            print(f"OUTPUT_FRAMES_DIR:{request_id}:{temp_frames_dir}", flush=True)

    def _encode_jpeg_b64(self, pil) -> str:
        frame_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buf).decode("utf-8") if ok else ""

    def _overlay_b64(self, frame_idx: int) -> str:
        return save_preview_mask_multi_object(self.masks, frame_idx, self.height, self.width)

    def handle_line(self, line: str) -> bool:
        request_id = ""
        try:
            cmd = json.loads(line)
            action = cmd.get("action")
            request_id = cmd.get("request_id", "")
            if action == "exit":
                return False

            if action == "get_frame":
                frame_index = int(cmd.get("frame_index", 0))
                pil = self.reader.get(frame_index)
                b64_frame = self._encode_jpeg_b64(pil)
                b64_mask = "NONE"
                if self._has_mask_at(frame_index):
                    b64_mask = self._overlay_b64(frame_index)
                resp = json.dumps({"frame": b64_frame, "mask": b64_mask})
                print(f"GET_FRAME_RESPONSE:{request_id}:{resp}", flush=True)
                return True

            objects = cmd.get("objects", {}) or {}
            if not objects:
                root_pts = cmd.get("points", []) or []
                root_lbls = cmd.get("labels", []) or []
                root_bxs = cmd.get("boxes", []) or []
                if root_pts or root_bxs:
                    objects = {"1": {"points": root_pts, "labels": root_lbls, "boxes": root_bxs}}
            text_prompt = cmd.get("text_prompt", "") or ""
            if not objects and text_prompt.strip():
                objects = {"1": {"points": [], "labels": [], "boxes": []}}
            text_mode = cmd.get("text_mode", "all") or "all"
            frame_index = int(cmd.get("frame_index", 0))
            output_dir = cmd.get("output_dir", "") or ""
            output_format = cmd.get("output_format", "GreenScreenMP4")

            if not text_prompt.strip() and not objects:
                print(f"ERROR:{request_id}:No tracking targets specified", flush=True)
                return True

            changed = self._accept_prompt(frame_index, objects, text_prompt, text_mode)
            self._bake()

            if action == "preview":
                if not self._has_mask_at(frame_index):
                    if frame_index > (self._cov_hi if self._cov_hi is not None else -1):
                        self._roll_forward(target=frame_index)
                    elif frame_index < (self._cov_lo if self._cov_lo is not None else self.total + 1):
                        self._roll_backward(target=frame_index)
                pil = self.reader.get(frame_index)
                b64_frame = self._encode_jpeg_b64(pil)
                b64_mask = self._overlay_b64(frame_index)
                resp = json.dumps({"frame": b64_frame, "mask": b64_mask})
                print(f"PREVIEW_RESPONSE:{request_id}:{resp}", flush=True)
                return True

            if action == "track":
                t_start = time.time()
                print(f"PROGRESS:0/{self.total}", flush=True)
                if changed or not self._baked:
                    self._bake()
                self._roll_backward(target=0, report=True)
                self._roll_forward(target=self.total - 1, report=True)
                t_prop = time.time()
                print(f"Propagation done in {t_prop - t_start:.1f}s, rendering output...", flush=True)
                timestamp = int(time.time())
                output_filename = f"cutout_{timestamp}"
                self._render_track(output_dir, output_format, request_id, output_filename)
                t_end = time.time()
                print(f"Total: {t_end - t_start:.1f}s (propagate: {t_prop - t_start:.1f}s, render: {t_end - t_prop:.1f}s)", flush=True)
                self._release_session()
                self.reader.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                force_release_ram()
                _memlog("track-end")
                return True

            print(f"ERROR:{request_id}:unknown action {action}", flush=True)
            return True
        except _ChunkFallback:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR:{request_id}:{str(e)}", flush=True)
            return True


def _legacy_load_video_frames(video_path: str, start_sec: float, end_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames_pil = []
        if total_frames > 1 and end_sec > start_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)
            start_frame_guess = int(round(start_sec * fps))
            end_frame_guess = int(round(end_sec * fps))
            cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            if cur_idx < start_frame_guess - 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_guess)
            while True:
                cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                if cur_frame >= end_frame_guess:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                t_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                if t_msec and (t_msec / 1000.0) > end_sec:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_pil.append(Image.fromarray(frame_rgb))
                if len(frames_pil) % 120 == 0:
                    gc.collect()
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_pil.append(Image.fromarray(frame_rgb))
                if len(frames_pil) % 120 == 0:
                    gc.collect()
    finally:
        cap.release()
    if not frames_pil:
        raise RuntimeError("No frames found in segment.")
    if w <= 0 or h <= 0:
        w, h = frames_pil[0].size
    return frames_pil, float(fps), int(w), int(h)


def _legacy_build_session(processor, frames_pil, device: str, dtype):
    dev = torch.device(device)
    raw_video = [np.array(fr) for fr in frames_pil]
    kwargs = dict(
        video=raw_video,
        inference_device=dev,
        processing_device="cpu",
        video_storage_device="cpu",
        inference_state_device="cpu",
        dtype=dtype,
    )
    with torch.inference_mode():
        session = call_init_video_session(processor, **kwargs)
    if hasattr(session, "inference_device"):
        session.inference_device = dev
    if hasattr(session, "cache") and hasattr(session.cache, "inference_device"):
        session.cache.inference_device = dev
    del raw_video
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return session


def _run_legacy_stdin(model, processor, session, frames_pil, height, width, fps, masks_by_frame, device, dtype,
                     first_line=None):
    lines = [first_line] if first_line else []
    if first_line:
        for line in lines:
            if not process_command(line, model, processor, session, height, width,
                                   frames_pil, fps, masks_by_frame, device, dtype):
                return
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if not process_command(line, model, processor, session, height, width,
                               frames_pil, fps, masks_by_frame, device, dtype):
            return


# =========================================================
# グローバル キャッシュデータ
# =========================================================
_last_objects = {}
_last_text_prompt = ""
_last_text_mode = ""
_prompt_frame_idx = 0


# =========================================================
# メインコマンド処理
# =========================================================

def process_command(cmd_str, model, processor, session, height, width,
                    frames_pil, fps, masks_by_frame, device, dtype):
    global _last_objects, _last_text_prompt, _last_text_mode, _prompt_frame_idx
    request_id = ""
    try:
        cmd = json.loads(cmd_str)
        action = cmd.get("action")
        request_id = cmd.get("request_id", "")

        if action == "exit":
            return False

        if action == "get_frame":
            frame_index = cmd.get("frame_index", 0)
            frame_pil = frames_pil[frame_index]
            
            frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            _, im_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_frame = base64.b64encode(im_buf).decode("utf-8")
            
            b64_mask = "NONE"
            if frame_index in masks_by_frame:
                b64_mask = save_preview_mask_multi_object(masks_by_frame, frame_index, height, width)
                
            response_json = json.dumps({
                "frame": b64_frame,
                "mask": b64_mask
            })
            print(f"GET_FRAME_RESPONSE:{request_id}:{response_json}", flush=True)
            return True

        objects = cmd.get("objects", {})
        output_dir = cmd.get("output_dir", "")
        output_format = cmd.get("output_format", "GreenScreenMP4")
        text_prompt = cmd.get("text_prompt", "")
        text_mode = cmd.get("text_mode", "all")
        frame_index = cmd.get("frame_index", 0)

        if not objects:
            root_pts = cmd.get("points", [])
            root_lbls = cmd.get("labels", [])
            root_bxs = cmd.get("boxes", [])
            if root_pts or root_bxs:
                objects = {
                    "1": {
                        "points": root_pts,
                        "labels": root_lbls,
                        "boxes": root_bxs
                    }
                }

        if text_prompt.strip() and len(frames_pil) == 1:
            obj_pts = []
            obj_lbls = []
            obj_bxs = []
            if "1" in objects:
                obj_pts = objects["1"].get("points", [])
                obj_lbls = objects["1"].get("labels", [])
                obj_bxs = objects["1"].get("boxes", [])

            process_text_prompt(
                frames_pil, text_prompt, obj_pts, obj_lbls, obj_bxs,
                output_dir, height, width, device, dtype,
                action, output_format, fps, text_mode, request_id
            )
            return True

        if text_prompt.strip() and not objects:
            objects = {"1": {"points": [], "labels": [], "boxes": []}}

        if not text_prompt.strip() and not objects:
            print(f"ERROR:{request_id}:No tracking targets specified", flush=True)
            return True

        inputs_changed = (
            objects != _last_objects or
            text_prompt != _last_text_prompt or
            text_mode != _last_text_mode
        )

        if inputs_changed:
            if hasattr(session, "reset_tracking_data"):
                session.reset_tracking_data()
            masks_by_frame.clear()
          
            _prompt_frame_idx = frame_index
            _last_objects = objects
            _last_text_prompt = text_prompt
            _last_text_mode = text_mode
            is_first = True

            temp_queries = []
            for obj_id_str, obj_data in objects.items():
                obj_id = int(obj_id_str)
                pts = obj_data.get("points", [])
                lbls = obj_data.get("labels", [])
                bxs = obj_data.get("boxes", [])

                if bxs:
                    for idx, box in enumerate(bxs):
                        temp_id = obj_id * 1000 + idx
                        query_pts = pts if idx == 0 else []
                        query_lbls = lbls if idx == 0 else []
                        temp_queries.append({
                            "temp_id": temp_id,
                            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                            "points": [[int(p[0]), int(p[1])] for p in query_pts] if query_pts else None,
                            "labels": [int(l) for l in query_lbls] if query_lbls else None
                        })
                elif pts:
                    temp_id = obj_id * 1000
                    temp_queries.append({
                        "temp_id": temp_id,
                        "box": None,
                        "points": [[int(p[0]), int(p[1])] for p in pts],
                        "labels": [int(l) for l in lbls]
                    })
                else:
                    temp_id = obj_id * 1000
                    temp_queries.append({
                        "temp_id": temp_id,
                        "box": None,
                        "points": None,
                        "labels": None
                    })

            all_boxes_list = []
            all_box_obj_ids = []
            for q in temp_queries:
                if q["box"] is not None:
                    all_boxes_list.append(q["box"])
                    all_box_obj_ids.append(q["temp_id"])

            if all_boxes_list:
                nested_boxes = [all_boxes_list]
                call_add_inputs(
                    processor,
                    inference_session=session,
                    frame_idx=frame_index,
                    obj_ids=all_box_obj_ids,
                    input_boxes=nested_boxes,
                    clear_old_inputs=is_first,
                )
                is_first = False

            all_points_list = []
            all_labels_list = []
            all_point_obj_ids = []
            for q in temp_queries:
                if q["points"] is not None:
                    all_points_list.append(q["points"])
                    all_labels_list.append(q["labels"])
                    all_point_obj_ids.append(q["temp_id"])

            if all_points_list:
                nested_points = [all_points_list]
                nested_labels = [all_labels_list]
                call_add_inputs(
                    processor,
                    inference_session=session,
                    frame_idx=frame_index,
                    obj_ids=all_point_obj_ids,
                    input_points=nested_points,
                    input_labels=nested_labels,
                    clear_old_inputs=is_first,
                )
                is_first = False

            if text_prompt.strip():
                for q in temp_queries:
                    q_pts = q["points"] if q["points"] else []
                    q_lbls = q["labels"] if q["labels"] else []
                    q_bxs = [q["box"]] if q["box"] else []
                  
                    init_mask = generate_initial_mask_with_text(
                        frames_pil[frame_index], text_prompt, q_pts, q_lbls, q_bxs, height, width, device, dtype, text_mode
                    )
                  
                    call_add_inputs(
                        processor,
                        inference_session=session,
                        frame_idx=frame_index,
                        obj_ids=[q["temp_id"]],
                        input_masks=init_mask,
                        clear_old_inputs=is_first,
                    )
                    is_first = False

            with torch.inference_mode():
                obj_ids_list = [q["temp_id"] for q in temp_queries]
                if not obj_ids_list:
                    obj_ids_list = [1000]
                    
                session.obj_with_new_inputs = obj_ids_list
                outputs = model(
                    inference_session=session,
                    frame_idx=frame_index,
                )
                session.obj_with_new_inputs = []

            processed = processor.post_process_masks(
                [outputs.pred_masks],
                [[session.video_height, session.video_width]],
                binarize=False,
            )[0]

            out_obj_ids = to_int_list(getattr(outputs, "object_ids", None))
            if out_obj_ids is None:
                out_obj_ids = [q["temp_id"] for q in temp_queries]

            masks_norm = normalize_masks(processed, out_obj_ids)

            masks_for_frame = masks_by_frame.setdefault(frame_index, {})
            for oid in objects.keys():
                masks_for_frame[int(oid)] = np.zeros((height, width), dtype=np.uint8)

            for k, temp_id in enumerate(out_obj_ids):
                obj_id = temp_id // 1000
                mask_2d = (masks_norm[k] > 0.0).to(torch.uint8).cpu().numpy()
                masks_for_frame[obj_id] = masks_for_frame[obj_id] | mask_2d

        if action == "preview":
            frame_pil = frames_pil[frame_index]
            frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            _, im_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_frame = base64.b64encode(im_buf).decode("utf-8")

            if len(frames_pil) == 1:
                mask = get_union_mask(masks_by_frame, frame_index, height, width)
                overlay = np.zeros((height, width, 4), dtype=np.uint8)
                overlay[mask > 0] = [255, 100, 50, 128]

                out_path = os.path.join(output_dir, "preview_mask.png")
                is_success, im_buf_arr = cv2.imencode(".png", overlay)
                if is_success:
                    im_buf_arr.tofile(out_path)
                print(f"PREVIEW_RESPONSE:{request_id}:{out_path}", flush=True)
                return True

            if frame_index == _prompt_frame_idx:
                b64_mask = save_preview_mask_multi_object(masks_by_frame, frame_index, height, width)
            else:
                if frame_index not in masks_by_frame:
                    is_past = frame_index < _prompt_frame_idx
                    run_propagation_single_direction(
                        model, processor, session, frames_pil,
                        masks_by_frame, start_frame_idx=_prompt_frame_idx,
                        target_frame_idx=frame_index,
                        reverse=is_past
                    )
                b64_mask = save_preview_mask_multi_object(masks_by_frame, frame_index, height, width)

            response_json = json.dumps({
                "frame": b64_frame,
                "mask": b64_mask
            })
            print(f"PREVIEW_RESPONSE:{request_id}:{response_json}", flush=True)
            return True

        elif action == "track":
            timestamp = int(time.time())
            output_filename = f"cutout_{timestamp}"

            if len(frames_pil) == 1:
                mask = get_union_mask(masks_by_frame, _prompt_frame_idx, height, width)
                mask_uint8 = mask.astype(np.uint8)

                if output_format == "MaskPNG":
                    final_frame = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)
                elif output_format in ("TransparentPNG", "AutoMaskPNG"):
                    frame_rgb = np.array(frames_pil[0])
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    b, g, r_ch = cv2.split(frame_bgr)
                    b_m = cv2.bitwise_and(b, b, mask=mask_uint8)
                    g_m = cv2.bitwise_and(g, g, mask=mask_uint8)
                    r_m = cv2.bitwise_and(r_ch, r_ch, mask=mask_uint8)
                    rgba = [b_m, g_m, r_m, mask_uint8 * 255]
                    final_frame = cv2.merge(rgba)
                else:
                    final_frame = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)

                out_path = os.path.join(output_dir, f"{output_filename}.png")
                is_success, im_buf_arr = cv2.imencode(".png", final_frame)
                if is_success:
                    im_buf_arr.tofile(out_path)
                print(f"OUTPUT_IMAGE:{request_id}:{out_path}", flush=True)
                return True

            t_start = time.time()
            print(f"PROGRESS:0/{len(frames_pil)}", flush=True)

            seen_frames = run_propagation_both_directions(
                model, processor, session, frames_pil,
                masks_by_frame, start_frame_idx=_prompt_frame_idx
            )

            t_propagate = time.time()
            print(f"Propagation done in {t_propagate - t_start:.1f}s, rendering output...", flush=True)

            out = None
            temp_frames_dir = ""

            if output_format in ("GreenScreenMP4", "MaskMP4"):
                out_path = os.path.join(output_dir, f"{output_filename}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                valid_fps = 30.0 if fps <= 0 else fps
                out = cv2.VideoWriter(out_path, fourcc, valid_fps, (width, height))
            else:
                temp_frames_dir = os.path.join(output_dir, "temp_masks")
                os.makedirs(temp_frames_dir, exist_ok=True)
                out_path = temp_frames_dir

            total = len(frames_pil)
            for i in range(total):
                mask = get_union_mask(masks_by_frame, i, height, width)
                mask_uint8 = mask.astype(np.uint8)

                if output_format == "MaskMP4":
                    final_frame = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)
                    out.write(final_frame)
                elif output_format == "GreenScreenMP4":
                    frame_rgb = np.array(frames_pil[i])
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    bg = np.zeros_like(frame_bgr)
                    bg[:] = (0, 255, 0)
                    mask_3ch = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)
                    fg = cv2.bitwise_and(frame_bgr, mask_3ch)
                    bg_masked = cv2.bitwise_and(bg, cv2.bitwise_not(mask_3ch))
                    final_frame = cv2.add(fg, bg_masked)
                    out.write(final_frame)
                else:
                    frame_rgb = np.array(frames_pil[i])
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    b, g, r_ch = cv2.split(frame_bgr)
                    b_m = cv2.bitwise_and(b, b, mask=mask_uint8)
                    g_m = cv2.bitwise_and(g, g, mask=mask_uint8)
                    r_m = cv2.bitwise_and(r_ch, r_ch, mask=mask_uint8)
                    rgba = [b_m, g_m, r_m, mask_uint8 * 255]
                    final_frame = cv2.merge(rgba)

                    png_path = os.path.join(temp_frames_dir, f"frame_{i:06d}.png")
                    is_success, im_buf_arr = cv2.imencode(".png", final_frame)
                    if is_success:
                        im_buf_arr.tofile(png_path)

                if (i + 1) % max(1, total // 20) == 0 or (i + 1) == total:
                    print(f"PROGRESS:{i + 1}/{total}", flush=True)

            if out is not None:
                out.release()
                print(f"OUTPUT_VIDEO:{request_id}:{out_path}", flush=True)
            else:
                print(f"OUTPUT_FRAMES_DIR:{request_id}:{temp_frames_dir}", flush=True)

            t_end = time.time()
            print(f"Total: {t_end - t_start:.1f}s (propagate: {t_propagate - t_start:.1f}s, render: {t_end - t_propagate:.1f}s)", flush=True)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            force_release_ram()

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR:{request_id}:{str(e)}", flush=True)
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python processor.py <mode> ...")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "server":
        if len(sys.argv) < 4:
            print("Usage: python processor.py server <video_path> <token> [device_index] [start_sec] [end_sec]")
            sys.exit(1)

        video_path = sys.argv[2]
        hf_token = sys.argv[3]
        device_index = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
        start_sec = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.0
        end_sec = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.0
        mode_arg = sys.argv[7] if len(sys.argv) >= 8 else MODE_AUTO
        if mode_arg not in (MODE_AUTO, MODE_CHUNK, MODE_LEGACY):
            mode_arg = MODE_AUTO

        login(token=hf_token, add_to_git_credential=False)

        device = select_device(device_index)
        dtype = select_dtype(device)

        print(f"Device: {device}, Dtype: {dtype}, StartSec: {start_sec}, EndSec: {end_sec}", flush=True)

        model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3", torch_dtype=dtype
        ).to(device).eval()
        processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

        IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff", ".tif")
        
        is_image_file = False
        is_animated_image = False
        
        ext = os.path.splitext(video_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            try:
                with Image.open(video_path) as img:
                    is_animated_image = getattr(img, "is_animated", False) and getattr(img, "n_frames", 1) > 1
                    is_image_file = not is_animated_image
            except:
                is_image_file = True

        frames_pil = []
        fps = 30.0
        width = 0
        height = 0
        chunk_engine = None
        reader = None

        if is_image_file:
            try:
                orig_img = Image.open(video_path).convert("RGB")
                width, height = orig_img.size
                frames_pil.append(orig_img)
            except Exception as ex:
                print(f"ERROR: Failed to open image: {ex}", flush=True)
                sys.exit(1)
        elif is_animated_image:
            try:
                with Image.open(video_path) as img:
                    width, height = img.size
                    duration = img.info.get("duration", 40)
                    if duration and duration > 0:
                        fps = 1000.0 / duration
                    else:
                        fps = 30.0
                    for frame_idx in range(img.n_frames):
                        img.seek(frame_idx)
                        frames_pil.append(img.convert("RGB"))
            except Exception as ex:
                print(f"ERROR: Failed to open animated image: {ex}", flush=True)
                sys.exit(1)
        else:
            probe_cap = cv2.VideoCapture(video_path)
            if not probe_cap.isOpened():
                print(f"ERROR: Failed to open video: {video_path}", flush=True)
                sys.exit(1)
            fps_probe = probe_cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps_probe) if fps_probe and fps_probe > 0 else 30.0
            width = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames_probe = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            probe_cap.release()

            want_chunk = False
            if mode_arg == MODE_CHUNK:
                want_chunk = True
            elif mode_arg == MODE_AUTO:
                approx = total_frames_probe
                if end_sec > start_sec and total_frames_probe > 1:
                    approx = int(round((end_sec - start_sec) * fps))
                if approx <= 0:
                    approx = total_frames_probe
                want_chunk = approx > AUTO_LEGACY_MAX_FRAMES
            print(f"Mode: mode_arg={mode_arg} want_chunk={want_chunk}", file=sys.stderr, flush=True)

            if want_chunk:
                reader = _SegmentFrameReader(video_path, start_sec, end_sec)
                _memlog("scan-start")
                n_scanned = reader.probe_and_scan()
                _memlog("scan-end")
                if n_scanned <= 0:
                    print("ERROR: No frames found", flush=True)
                    sys.exit(1)
                if mode_arg == MODE_AUTO and n_scanned <= AUTO_LEGACY_MAX_FRAMES:
                    reader.close()
                    reader = None
                    want_chunk = False

            if want_chunk:
                fps = reader.fps
                width = reader.width
                height = reader.height
                chunk_engine = _ChunkEngine(
                    model, processor, reader, device, dtype,
                    height, width, fps,
                    text_initial_mask_fn=generate_initial_mask_with_text,
                )
                print(f"Loaded {chunk_engine.total} frames ({width}x{height} @ {fps:.2f}fps) [chunk mode]", flush=True)
                _memlog("ready")
                print(f"TOTAL_FRAMES:{chunk_engine.total}", flush=True)
                print("SERVER_READY", flush=True)
                force_release_ram()
                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if not chunk_engine.handle_line(line):
                            break
                    except _ChunkFallback as ex:
                        if mode_arg != MODE_AUTO:
                            print(f"ERROR::chunk mode unavailable (fallback disabled): {ex}", flush=True)
                            break
                        print(f"[SAM3] chunk mode failed ({ex}). Falling back to legacy mode...", file=sys.stderr, flush=True)
                        try:
                            legacy_frames, legacy_fps, legacy_w, legacy_h = _legacy_load_video_frames(
                                video_path, start_sec, end_sec)
                            legacy_session = _legacy_build_session(processor, legacy_frames, device, dtype)
                        except Exception as ex2:
                            import traceback
                            traceback.print_exc()
                            print(f"ERROR::fallback failed: {ex2}", flush=True)
                            break
                        legacy_masks = {}
                        print(f"[SAM3] switched to legacy mode ({len(legacy_frames)} frames)", file=sys.stderr, flush=True)
                        _run_legacy_stdin(
                            model, processor, legacy_session, legacy_frames,
                            legacy_h, legacy_w, legacy_fps, legacy_masks, device, dtype,
                            first_line=line,
                        )
                        break
                sys.exit(0)

            try:
                frames_pil, fps, width, height = _legacy_load_video_frames(video_path, start_sec, end_sec)
            except Exception as ex:
                print(f"ERROR: Video load failed: {ex}", flush=True)
                sys.exit(1)

        if len(frames_pil) == 0:
            print("ERROR: No frames found", flush=True)
            sys.exit(1)

        print(f"Loaded {len(frames_pil)} frames ({width}x{height} @ {fps:.2f}fps)", flush=True)
        _memlog("loaded")

        session = _legacy_build_session(processor, frames_pil, device, dtype)
        _memlog("session-built")

        masks_by_frame = {}

        print(f"TOTAL_FRAMES:{len(frames_pil)}", flush=True)
        print("SERVER_READY", flush=True)

        force_release_ram()

        with torch.inference_mode():
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                if not process_command(line, model, processor, session, height, width,
                                       frames_pil, fps, masks_by_frame, device, dtype):
                    break
        sys.exit(0)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    main()