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


# 元々の「全切り抜き(all)」用ロジック（従来通り動くよう完全保護）
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


# 「最も合うものを切り抜き(best)」用フィルタロジック
def _generate_initial_mask_with_text_best_filtered(img, text_prompt, points, labels, boxes, height, width, device, dtype, is_fallback=False):
    img_model, img_processor = get_image_model(device, dtype)

    # フォールバック処理中、またはプロンプトが空の場合は、テキストを使用せず純粋な形状切り抜きを行う
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
        
        # 安全な NumPy / List 変換
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
    # 1. テキストプロンプトのみ（ボックスもポイントも指定されていない完全にクリーンなテキスト専用モード）の場合
    # 画像全体の中から、高パフォーマンスかつ高互換性な元の推論・抽出ロジックに直接ルーティングします
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
            # bestモード
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

    # 2. ボックスまたはポイントが部分的にでも指定されている場合
    if text_mode == "all":
        # 合うもの全て切り抜き：元の処理（全出力結合）を完全復元
        return _generate_initial_mask_with_text_original(img, text_prompt, points, labels, boxes, height, width, device, dtype)
    else:
        # 最も合うものを切り抜き：フォールバック付きフィルタロジック
        return _generate_initial_mask_with_text_best_filtered(img, text_prompt, points, labels, boxes, height, width, device, dtype)


# 静止画テキストプロンプト処理
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
        # C#側のSendCommandAsyncがリクエストIDの一致を待ち受けられるようにプレフィックスをフォーマット
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
            # アルファマスク外の余計なピクセルを完全黒潰し（透過バグ解消）
            b_m = cv2.bitwise_and(b, b, mask=mask)
            g_m = cv2.bitwise_and(g, g, mask=mask)
            r_m = cv2.bitwise_and(r_ch, r_ch, mask=mask)
            rgba = [b_m, g_m, r_m, mask * 255]
            final_frame = cv2.merge(rgba)

        out_path = os.path.join(output_dir, f"{output_filename}.png")
        is_success, im_buf_arr = cv2.imencode(".png", final_frame)
        if is_success:
            im_buf_arr.tofile(out_path)
        # C#側のSendCommandAsyncが結果パスをキャッチできるようにリクエストIDを含めて出力
        print(f"OUTPUT_IMAGE:{request_id}:{out_path}", flush=True)


# =========================================================
# マスクユーティリティ
# =========================================================

def get_union_mask(masks_by_frame, frame_idx, h, w):
    masks = masks_by_frame.get(frame_idx, {})
    union = None
    for m in masks.values():
        if m is None:
            continue
        mm = m
        if isinstance(mm, torch.Tensor):
            mm = mm.detach().cpu().numpy()
        mm = np.asarray(mm)
        if mm.ndim == 3:
            mm = mm.squeeze()
        mm = (mm > 0).astype(np.uint8)
        if mm.shape != (h, w):
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
        union = mm if union is None else (union | mm)
    if union is None:
        union = np.zeros((h, w), dtype=np.uint8)
    return union.astype(bool)


# 6色周期マスクカラーテーブル (B, G, R, A)
MASK_COLORS = [
    [255, 100, 0, 128],   # 1: 青 (半透明)
    [0, 0, 255, 128],     # 2: 赤 (半透明)
    [0, 200, 0, 128],     # 3: 緑 (半透明)
    [0, 255, 255, 128],   # 4: 黄 (半透明)
    [255, 0, 255, 128],   # 5: 紫 (半透明)
    [220, 220, 0, 128]    # 6: 水 (半透明)
]


# 指定フレームの全オブジェクトマスクを、固有の色で重ね塗りして1枚のPNGに合成する
def save_preview_mask_multi_object(masks_by_frame, frame_idx, height, width):
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    
    masks_for_frame = masks_by_frame.get(frame_idx, {})
    
    for oid, mask in sorted(masks_for_frame.items()):
        if mask is None:
            continue
        mm = mask
        if isinstance(mm, torch.Tensor):
            mm = mm.detach().cpu().numpy()
        mm = np.asarray(mm)
        if mm.ndim == 3:
            mm = mm.squeeze()
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

# プレビュー用の一方向（シーク方向が未来なら順行、過去なら逆行）ピンポイント高速伝播メソッド
def run_propagation_single_direction(model, processor, session, frames_pil, masks_by_frame, start_frame_idx=0, target_frame_idx=None, reverse=False):
    total = len(frames_pil)
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
            
            # 安全ガード：進行方向と逆のフレームが万が一来ても破棄
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

            # 一時オブジェクトID（temp_id）から元のオブジェクトID（obj_id = temp_id // 1000）へマージして格納
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

            # シークターゲットに到達した瞬間に即座にブレイク
            if target_frame_idx is not None and frame_idx == target_frame_idx:
                break

    return seen_frames


# 動画出力（track）用の双方向（順・逆）完全自動伝播メソッド
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

                # 一時オブジェクトID（temp_id）から元のオブジェクトID（obj_id = temp_id // 1000）へマージして格納
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

        # get_frameは最優先冒盤で処理
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

        # ---- これより下は preview / track 用 of 処理 ----
        objects = cmd.get("objects", {})
        output_dir = cmd.get("output_dir", "")
        output_format = cmd.get("output_format", "GreenScreenMP4")
        text_prompt = cmd.get("text_prompt", "")
        text_mode = cmd.get("text_mode", "all")
        frame_index = cmd.get("frame_index", 0)

        # SAM3ImageClickWindow（静止画モード）からの直接形式（objectsキーがない構造）との上位互換性確保
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

        # テキストプロンプト処理 (静止画モード専用)
        if text_prompt.strip() and len(frames_pil) == 1:
            # 静止画用オブジェクト1に割り当てられているボックス/ポイントを抽出して連動させる
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

        # テキストプロンプトが入力されており、かつオブジェクトが空の場合のフォールバック
        if text_prompt.strip() and not objects:
            objects = {"1": {"points": [], "labels": [], "boxes": []}}

        if not text_prompt.strip() and not objects:
            print(f"ERROR:{request_id}:No tracking targets specified", flush=True)
            return True

        # キャッシュ連動の比較判定を objects に合わせる
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

            # 1つのオブジェクト内のボックス群を一時的に個別のオブジェクトID（base_id * 1000 + idx）に分割
            temp_queries = []
            for obj_id_str, obj_data in objects.items():
                obj_id = int(obj_id_str)
                pts = obj_data.get("points", [])
                lbls = obj_data.get("labels", [])
                bxs = obj_data.get("boxes", [])

                if bxs:
                    for idx, box in enumerate(bxs):
                        temp_id = obj_id * 1000 + idx
                        # ポイントは最初のボックスクエリにのみ紐付ける
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

            # 第1段階：全一時的クエリの【ボックス】プロンプトを一括マージ登録
            all_boxes_list = []
            all_box_obj_ids = []
            for q in temp_queries:
                if q["box"] is not None:
                    all_boxes_list.append(q["box"])
                    all_box_obj_ids.append(q["temp_id"])

            if all_boxes_list:
                nested_boxes = [all_boxes_list]  # 3次元ネスト
                call_add_inputs(
                    processor,
                    inference_session=session,
                    frame_idx=frame_index,
                    obj_ids=all_box_obj_ids,
                    input_boxes=nested_boxes,
                    clear_old_inputs=is_first,
                )
                is_first = False

            # 第2段階：全一時的クエリの【ポイント】プロンプトを一括マージ登録
            all_points_list = []
            all_labels_list = []
            all_point_obj_ids = []
            for q in temp_queries:
                if q["points"] is not None:
                    all_points_list.append(q["points"])
                    all_labels_list.append(q["labels"])
                    all_point_obj_ids.append(q["temp_id"])

            if all_points_list:
                nested_points = [all_points_list]  # 4次元ネスト
                nested_labels = [all_labels_list]  # 3次元ネスト
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

            # 第3段階：テキストプロンプトによる初期ベースマスクの生成と登録
            # 各一時オブジェクトごとにマスク推論を個別呼び出し
            if text_prompt.strip():
                for q in temp_queries:
                    q_pts = q["points"] if q["points"] else []
                    q_lbls = q["labels"] if q["labels"] else []
                    q_bxs = [q["box"]] if q["box"] else []
                  
                    init_mask = generate_initial_mask_with_text(
                        frames_pil[frame_index], text_prompt, q_pts, q_lbls, q_bxs, height, width, device, dtype, text_mode
                    )
                  
                    # マスクのみを綺麗に登録
                    call_add_inputs(
                        processor,
                        inference_session=session,
                        frame_idx=frame_index,
                        obj_ids=[q["temp_id"]],
                        input_masks=init_mask,
                        clear_old_inputs=is_first,
                    )
                    is_first = False

            # 基準フレームにおける一時オブジェクト初期推論
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

            # 基準フレームのマルチマスクの取り出し
            processed = processor.post_process_masks(
                [outputs.pred_masks],
                [[session.video_height, session.video_width]],
                binarize=False,
            )[0]

            out_obj_ids = to_int_list(getattr(outputs, "object_ids", None))
            if out_obj_ids is None:
                out_obj_ids = [q["temp_id"] for q in temp_queries]

            masks_norm = normalize_masks(processed, out_obj_ids)

            # 一時オブジェクトID（temp_id）から元のオブジェクトIDへマージして masks_by_frame に格納
            masks_for_frame = masks_by_frame.setdefault(frame_index, {})
            for oid in objects.keys():
                masks_for_frame[int(oid)] = np.zeros((height, width), dtype=np.uint8)

            for k, temp_id in enumerate(out_obj_ids):
                obj_id = temp_id // 1000
                mask_2d = (masks_norm[k] > 0.0).to(torch.uint8).cpu().numpy()
                masks_for_frame[obj_id] = masks_for_frame[obj_id] | mask_2d

        # ============ アクション別処理 ============

        if action == "preview":
            frame_pil = frames_pil[frame_index]
            frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            _, im_buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64_frame = base64.b64encode(im_buf).decode("utf-8")

            if len(frames_pil) == 1:
                # 静止画モードのプレビュー: 透過マスクをファイル保存してパスを返す
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
                # 基準フレームのプレビュー
                b64_mask = save_preview_mask_multi_object(masks_by_frame, frame_index, height, width)
            else:
                # シーク先が過去方向か未来方向かを自動判定し、その一方向のみピンポイント順/逆伝播
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

            # --- 単一画像 ---
            if len(frames_pil) == 1:
                mask = get_union_mask(masks_by_frame, _prompt_frame_idx, height, width)
                mask_uint8 = mask.astype(np.uint8)

                if output_format == "MaskPNG":
                    final_frame = cv2.cvtColor(mask_uint8 * 255, cv2.COLOR_GRAY2BGR)
                elif output_format in ("TransparentPNG", "AutoMaskPNG"):
                    frame_rgb = np.array(frames_pil[0])
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    b, g, r_ch = cv2.split(frame_bgr)
                    # アルファマスク外のピクセルを完全黒潰し
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
                    # 白黒（前景が真っ白255、背景が真っ黒0）のマスクフレームを書き出し
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
                    # アルファマスク外のピクセルを完全黒潰し（透過表示バグ解消）
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

        login(token=hf_token, add_to_git_credential=False)

        device = select_device(device_index)
        dtype = select_dtype(device)

        print(f"Device: {device}, Dtype: {dtype}, StartSec: {start_sec}, EndSec: {end_sec}", flush=True)

        model = Sam3TrackerVideoModel.from_pretrained(
            "facebook/sam3", torch_dtype=dtype
        ).to(device).eval()
        processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

        # ★ 入力ファイルが画像（静止画）かどうかを拡張子で判定し分岐
        is_image_file = video_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))

        frames_pil = []
        fps = 30.0
        width = 0
        height = 0

        if is_image_file:
            try:
                # 画像ファイルとして直接安全にロード
                orig_img = Image.open(video_path).convert("RGB")
                width, height = orig_img.size
                frames_pil.append(orig_img)
            except Exception as ex:
                print(f"ERROR: Failed to open image: {ex}", flush=True)
                sys.exit(1)
        else:
            # 動画ファイルとしてデコードロード
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"ERROR: Failed to open video: {video_path}", flush=True)
                sys.exit(1)

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps) if fps and fps > 0 else 30.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

            cap.release()

        if len(frames_pil) == 0:
            print("ERROR: No frames found", flush=True)
            sys.exit(1)

        print(f"Loaded {len(frames_pil)} frames ({width}x{height} @ {fps:.2f}fps)", flush=True)

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