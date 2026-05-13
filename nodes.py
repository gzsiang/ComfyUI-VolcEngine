"""
ComfyUI VolcEngine Video/Image Generation Node

Supports Seedance series: text-to-video + image-to-video (multi-image + reference audio)

API Reference: https://www.volcengine.com/docs/82379/1520757
Endpoint: POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks

Seedance 2.0 Multi-modal Reference:
  - Images (1~9): role=first_frame / last_frame / reference_image
  - Audio (1~3 clips): role=reference_audio (must have at least 1 reference image)
  - Text prompt

Output: Video frames (IMAGE) + Audio (AUDIO) + Video info (STRING)
"""

import os
import io
import time
import json
import re
import base64
import requests
import numpy as np
from PIL import Image
import folder_paths
import torch


# ─────────────────────────────────────────────
# Shared Resources: Preset Model Lists (Fallback)
# ─────────────────────────────────────────────

# Video generation models
FALLBACK_MODELS = [
    # Seedance 2.0 series
    "doubao-seedance-2-0-260128",
    # Seedance 1.5 series
    "doubao-seedance-1-5-pro-251215",
    # Seedance 1.0 series
    "doubao-seedance-1-0-pro-250528",
    "doubao-seedance-1-0-pro",
    "doubao-seedance-1-0-pro-fast",
    "doubao-seedance-1-0-lite",
]

# Image generation models
FALLBACK_IMAGE_MODELS = [
    # Seedream 5.0 series
    "doubao-seedream-5-0",
    # Seedream 4.5 series
    "doubao-seedream-4-5-251128",
    # Seedream 4.0 series
    "doubao-seedream-4-0-250828",
    # Seedream 3.x series
    "doubao-seedream-3-1-250312",
]

# Global cache: api_key → [model_id list]
MODEL_CACHE = {}


def fetch_available_models(api_key: str) -> list[str]:
    """Fetch available model list from VolcEngine API"""
    url = "https://ark.cn-beijing.volces.com/api/v3/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if not resp.ok:
            return []
        data = resp.json()
        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            if "seedance" in model_id.lower() or "seedream" in model_id.lower():
                models.append(model_id)
        return sorted(models)
    except Exception:
        return []


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def tensor_to_base64(image_tensor) -> str:
    """ComfyUI IMAGE tensor (1,H,W,C) → base64 JPEG string"""
    img_np = image_tensor[0].cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensors_to_base64_list(image_tensor) -> list[str]:
    """ComfyUI IMAGE tensor (B,H,W,C) → base64 JPEG list (one per image)"""
    results = []
    for i in range(image_tensor.shape[0]):
        img_np = image_tensor[i].cpu().numpy()
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        results.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return results


def audio_to_base64(audio_dict: dict) -> str:
    """ComfyUI AUDIO dict → base64 WAV string"""
    import wave

    waveform = audio_dict.get("waveform")
    sample_rate = audio_dict.get("sample_rate", 44100)

    if waveform is None:
        raise ValueError("Audio data is empty")

    audio_np = waveform.squeeze(0).cpu().numpy()  # [samples, channels]
    audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

    n_channels = audio_np.shape[1] if len(audio_np.shape) > 1 else 1

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(n_channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_np.tobytes())

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_video_from_url(url: str, expected_audio: bool = False):
    """
    Download video from URL and parse into frames (IMAGE tensor) and audio (AUDIO tensor)
    Returns: (frames_tensor, audio_dict, video_info_dict)

    Args:
        url: Video download URL
        expected_audio: Whether user expects audio (for error strategy)
    """
    import tempfile
    import subprocess

    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        import imageio
        reader = imageio.get_reader(tmp_path, "ffmpeg")
        meta = reader.get_meta_data()
        fps = meta.get("fps", 25.0)

        frames = []
        for frame in reader:
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            frames.append(frame_tensor)
        reader.close()

        if len(frames) == 0:
            raise ValueError("Unable to read video frames")

        frames_tensor = torch.stack(frames)  # [T, H, W, 3]

        # Extract audio
        audio_dict = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
                audio_path = audio_tmp.name

            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                audio_path
            ], check=True, capture_output=True)

            import wave
            with wave.open(audio_path, 'rb') as wav:
                n_channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_data = wav.readframes(n_frames)

                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_np = audio_np.reshape(-1, n_channels)
                audio_np = audio_np.astype(np.float32) / 32768.0
                # ComfyUI AUDIO format: [batch, channels, samples]
                audio_tensor = torch.from_numpy(audio_np).transpose(0, 1).unsqueeze(0)  # [1, channels, samples]

                audio_dict = {
                    "waveform": audio_tensor,
                    "sample_rate": sample_rate
                }

            os.unlink(audio_path)
        except subprocess.CalledProcessError as ffmpeg_err:
            stderr_str = ffmpeg_err.stderr.decode('utf-8', errors='replace') if ffmpeg_err.stderr else ""
            # Silent video is normal
            if 'no stream' not in stderr_str and 'Stream' not in stderr_str:
                if expected_audio:
                    if ffmpeg_err.returncode > 255:
                        print("[VolcEngine] Video has no audio, skipped")
                    else:
                        print(f"[VolcEngine] Audio extraction error: {stderr_str[:200]}")
            audio_dict = {
                "waveform": torch.zeros((1, 2, 1)),
                "sample_rate": 44100
            }

        except Exception as e:
            err_str = str(e)
            err_str = re.sub(r'exit status \d+', '', err_str).strip()
            if err_str:
                if expected_audio:
                    print(f"[VolcEngine] Audio extraction skipped: {err_str[:100]}")
            audio_dict = {
                "waveform": torch.zeros((1, 2, 1)),
                "sample_rate": 44100
            }

        video_info = {
            "fps": fps,
            "total_frames": len(frames),
            "width": frames[0].shape[1],
            "height": frames[0].shape[0],
            "duration": len(frames) / fps if fps > 0 else 0,
            "source_url": url,
        }

        return frames_tensor, audio_dict, video_info

    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# VolcEngine API Client
# ─────────────────────────────────────────────

class VolcEngineAPI:
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def create_task(self, payload: dict, api_key: str = None) -> str:
        """Create video generation task, return task_id"""
        url = f"{self.BASE_URL}/tasks"
        resp = requests.post(url, json=payload, headers=self.headers, timeout=60)
        if not resp.ok:
            error_text = resp.text
            friendly_hint = ""
            try:
                error_data = resp.json()
                error_code = ""
                if isinstance(error_data, dict):
                    err_obj = error_data.get("error", error_data)
                    error_code = err_obj.get("code", "") if isinstance(err_obj, dict) else ""
                # Content moderation errors
                if "SensitiveContent" in error_code or "PrivacyInformation" in error_code:
                    friendly_hint = "\n⚠️ Input image triggered content moderation. May contain faces or private info. Please change the reference image and retry."
                # 404 Model not found
                if resp.status_code == 404 and "NotFound" in str(error_data):
                    key = api_key or self.api_key
                    if key not in MODEL_CACHE:
                        print("[VolcEngine] Model not found, fetching available models...")
                        MODEL_CACHE[key] = fetch_available_models(key)
                    if MODEL_CACHE[key]:
                        print("\n" + "=" * 60)
                        print("[VolcEngine] Available models:")
                        for i, m in enumerate(MODEL_CACHE[key], 1):
                            print(f"  {i}. {m}")
                        print("=" * 60 + "\n")
                # 429 Rate limit
                if resp.status_code == 429:
                    friendly_hint = "\n⚠️ API rate limited. Please retry later or reduce concurrency."
                # 402 Insufficient balance
                if resp.status_code == 402:
                    friendly_hint = "\n⚠️ Insufficient account balance or expired plan. Please recharge."
                # 401 Auth failed
                if resp.status_code == 401:
                    friendly_hint = "\n⚠️ API Key is invalid or expired. Please check your key."
            except Exception:
                pass
            raise RuntimeError(f"Task creation failed [{resp.status_code}]: {error_text}{friendly_hint}")
        data = resp.json()
        task_id = data.get("id") or data.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id returned, response: {json.dumps(data, ensure_ascii=False)}")
        return task_id

    def poll_task(self, task_id: str, poll_interval: int = 10, max_wait: int = 600, model_name: str = "") -> dict:
        """Poll task status until success or timeout; retry on transient network errors"""
        import requests.exceptions
        url = f"{self.BASE_URL}/tasks/{task_id}"
        start = time.time()
        net_retry_count = 0
        max_net_retries = 5
        poll_count = 0
        prefix = model_name.strip() if model_name.strip() else "VolcEngine"
        print(f"[{prefix}] Polling task {task_id}...", flush=True)
        while time.time() - start < max_wait:
            try:
                resp = requests.get(url, headers=self.headers, timeout=30)
                net_retry_count = 0
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ReadTimeout) as e:
                net_retry_count += 1
                if net_retry_count > max_net_retries:
                    raise RuntimeError(f"Network error during polling (retried {max_net_retries} times): {e}")
                wait = min(poll_interval * net_retry_count, 60)
                elapsed = int(time.time() - start)
                print(f"\r[{prefix}] ⏳ Network error (retry #{net_retry_count}) | Waiting {elapsed//60}:{elapsed%60:02d} | Retrying...", end="", flush=True)
                time.sleep(wait)
                continue
            if not resp.ok:
                raise RuntimeError(f"Task query failed [{resp.status_code}]: {resp.text}")
            data = resp.json()
            status = data.get("status", "")
            if status == "succeeded":
                elapsed = int(time.time() - start)
                print(f"\r[{prefix}] ✅ Done | Polls {poll_count} | Elapsed {elapsed//60}:{elapsed%60:02d}    ")
                return data
            if status == "failed":
                error = data.get("error", {})
                raise RuntimeError(f"Task failed: {error.get('message', json.dumps(data, ensure_ascii=False))}")
            poll_count += 1
            elapsed = int(time.time() - start)
            print(f"\r[{prefix}] ⏳ Running | Poll #{poll_count} | Waiting {elapsed//60}:{elapsed%60:02d}", end="", flush=True)
            time.sleep(poll_interval)
        raise TimeoutError(f"Task timeout ({max_wait}s), task_id={task_id}")

    def extract_video_url(self, result: dict) -> str:
        """Extract video URL from task result"""
        choices = result.get("choices", [])
        if choices:
            for choice in choices:
                msg = choice.get("message", {})
                content = msg.get("content", "")
                if content and isinstance(content, str):
                    try:
                        c = json.loads(content)
                        url = c.get("url") or c.get("video_url")
                        if url:
                            return url
                    except json.JSONDecodeError:
                        pass
                url = msg.get("url") or msg.get("video_url")
                if url:
                    return url

        output = result.get("output", {})
        url = output.get("video_url") or output.get("url")
        if url:
            return url

        # New API: content is dict
        content = result.get("content")
        if isinstance(content, dict):
            url = content.get("video_url") or content.get("url")
            if url:
                return url
        # Legacy: content is list
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    url = item.get("url") or item.get("video_url")
                    if url:
                        return url

        raise RuntimeError(f"Unable to extract video URL: {json.dumps(result, ensure_ascii=False)[:500]}")


# ─────────────────────────────────────────────
# Build API Request Payload
# ─────────────────────────────────────────────

def build_payload(model, prompt="", image=None, audio=None, video_duration=5,
                  resolution="720p", aspect_ratio="adaptive", generate_audio=True,
                  camera_fixed=False, return_last_frame=False, service_tier="default",
                  seed=-1, watermark=True):
    """
    Build Seedance 2.0 multi-modal request payload

    Compatible with old/new API:
    - Seedance 2.0: content array + top-level params
    - Legacy: content array + prompt-embedded params
    """
    is_v2 = "2-0" in model

    if is_v2:
        # ── Seedance 2.0: content array + top-level params ──
        content_list = []

        # Text prompt
        if prompt.strip():
            content_list.append({"type": "text", "text": prompt.strip()})

        # Images: auto-detect role based on count
        if image is not None:
            num_images = image.shape[0]
            b64_list = tensors_to_base64_list(image)

            if num_images == 1:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_list[0]}"},
                    "role": "first_frame",
                })
            elif num_images == 2:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_list[0]}"},
                    "role": "first_frame",
                })
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_list[1]}"},
                    "role": "last_frame",
                })
            else:
                for b64 in b64_list:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        "role": "reference_image",
                    })

        # Reference audio (role=reference_audio, must have at least 1 image)
        if audio is not None:
            audio_b64 = audio_to_base64(audio)
            content_list.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                "role": "reference_audio",
            })

        payload = {
            "model": model.strip(),
            "content": content_list,
            "duration": video_duration,
            "resolution": resolution,
            "ratio": aspect_ratio,
            "generate_audio": generate_audio,
            "camera_fixed": camera_fixed,
            "return_last_frame": return_last_frame,
            "watermark": watermark,
        }
        # service_tier only supported for image-to-video
        if image is not None and service_tier != "default":
            payload["service_tier"] = service_tier
        if seed != -1:
            payload["seed"] = seed

    else:
        # ── Legacy Seedance 1.x: content array ──
        param_str = f" --duration {video_duration}"
        if seed != -1:
            param_str += f" --seed {seed}"
        param_str += " --watermark true" if watermark else " --watermark false"

        text_content = prompt.strip() if prompt.strip() else ""
        text_content += param_str

        content = [{"type": "text", "text": text_content}]

        if image is not None:
            num_images = image.shape[0]
            b64_list = tensors_to_base64_list(image)

            if num_images == 1:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_list[0]}"},
                    "role": "first_frame",
                })
            elif num_images == 2:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_list[0]}"},
                    "role": "first_frame",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_list[1]}"},
                    "role": "last_frame",
                })
            else:
                for b64 in b64_list:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        "role": "reference_image",
                    })

        payload = {"model": model.strip(), "content": content}

    return payload


# ─────────────────────────────────────────────
# Concurrent Generation Helper
# ─────────────────────────────────────────────

def _concurrent_run(num_tasks, concurrency, tasks, run_fn, log_prefix):
    """
    Execute tasks concurrently

    tasks: [(prompt, seed), ...] or [(image, prompt, seed), ...]
    run_fn: function(idx, task) -> (frames, audio, info)
    """
    import concurrent.futures

    all_frames = []
    all_audios = []
    all_infos = []
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(concurrency, num_tasks)) as executor:
        future_map = {}
        for idx, task in enumerate(tasks):
            future = executor.submit(run_fn, idx, task)
            future_map[future] = idx

        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                frames_tensor, audio_dict, video_info = future.result()
                all_frames.append((idx, frames_tensor))
                all_audios.append((idx, audio_dict))
                all_infos.append((idx, video_info))
            except Exception as e:
                print(f"[VolcEngine {log_prefix}] #{idx+1} failed: {e}")
                failed.append((idx, str(e)))

    if not all_frames:
        failed_msgs = [f"#Task{idx+1}: {err}" for idx, err in failed]
        error_detail = "\n".join(failed_msgs)
        raise RuntimeError(f"All {num_tasks} tasks failed\nDetails:\n{error_detail}")

    # Sort by original order
    all_frames.sort(key=lambda x: x[0])
    all_audios.sort(key=lambda x: x[0])
    all_infos.sort(key=lambda x: x[0])

    return all_frames, all_audios, all_infos, failed


def _combine_results(all_frames, all_audios):
    """Merge multiple results into one batch"""
    combined_frames = torch.cat([f for _, f in all_frames], dim=0)

    combined_waveforms = []
    max_sr = 44100
    for _, ad in all_audios:
        wf = ad.get("waveform", torch.zeros((1, 2, 1)))
        sr = ad.get("sample_rate", 44100)
        max_sr = max(max_sr, sr)
        combined_waveforms.append(wf)

    try:
        combined_audio = torch.cat(combined_waveforms, dim=2)
        combined_audio_dict = {
            "waveform": combined_audio,
            "sample_rate": max_sr
        }
    except Exception:
        combined_audio_dict = {
            "waveform": torch.zeros((1, 2, 1)),
            "sample_rate": 44100
        }

    return combined_frames, combined_audio_dict


# ─────────────────────────────────────────────
# Text-to-Video Node
# ─────────────────────────────────────────────

class VolcEngineTextToVideo:
    """VolcEngine Text-to-Video

    When concurrency > 1: roll mode (same prompt, different seeds)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "VolcEngine Ark API Key",
                }),
                "model": (FALLBACK_MODELS, {"default": FALLBACK_MODELS[0]}),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe the video content you want to generate",
                }),
                "concurrency": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                }),
                "video_duration": ("INT", {
                    "default": 5,
                    "min": 4,
                    "max": 15,
                    "step": 1,
                    "display": "number",
                }),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {"default": "16:9"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "service_tier": (["default", "flex"], {"default": "default"}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "poll_interval": ("INT", {"default": 10, "min": 3, "max": 30}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 1800}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("Video Frames", "Audio", "Video Info")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "VolcEngine"

    def execute(self, api_key, model, prompt, concurrency, video_duration, resolution, aspect_ratio, generate_audio, camera_fixed, return_last_frame, service_tier, watermark,
                seed=-1, poll_interval=10, max_wait=600):

        if not api_key.strip():
            raise ValueError("Please fill in VolcEngine Ark API Key")
        if not model.strip():
            raise ValueError("Please select a model")
        if not prompt.strip():
            raise ValueError("Please fill in prompt")

        api = VolcEngineAPI(api_key.strip())
        import random

        # Generate seed list
        base_seed = seed if seed != -1 else random.randint(0, 2147483647)
        seeds = [base_seed + i for i in range(concurrency)]

        log_prefix = "Text-to-Video"

        desc_parts = [f"model={model}", f"tasks={concurrency}"]
        if service_tier == "flex":
            desc_parts.append("offline")
        print(f"[VolcEngine {log_prefix}] {' | '.join(desc_parts)}")

        def _run_single(idx, task_seed):
            payload = build_payload(
                model=model, prompt=prompt, image=None, audio=None,
                video_duration=video_duration, resolution=resolution, aspect_ratio=aspect_ratio,
                generate_audio=generate_audio, camera_fixed=camera_fixed, return_last_frame=return_last_frame,
                service_tier="default", seed=task_seed, watermark=watermark
            )
            task_id = api.create_task(payload, api_key=api_key.strip())
            print(f"[VolcEngine {log_prefix}] #{idx+1}/{concurrency} seed={task_seed} | task_id={task_id}")

            result = api.poll_task(task_id, poll_interval=poll_interval, max_wait=max_wait, model_name=model)
            video_url = api.extract_video_url(result)

            print(f"[VolcEngine {log_prefix}] #{idx+1} fetching video and parsing frames...")
            frames_tensor, audio_dict, video_info = load_video_from_url(video_url, expected_audio=generate_audio)

            video_info["seed"] = task_seed
            video_info["task_index"] = idx + 1
            print(f"[VolcEngine {log_prefix}] #{idx+1} done: {video_info['total_frames']} frames, {video_info['duration']:.2f}s")

            return frames_tensor, audio_dict, video_info

        # Task list: (seed,)
        tasks = [(s,) for s in seeds]

        all_frames, all_audios, all_infos, failed = _concurrent_run(
            concurrency, concurrency, tasks, lambda idx, task: _run_single(idx, task[0]), log_prefix
        )

        combined_frames, combined_audio_dict = _combine_results(all_frames, all_audios)

        # Build info
        result_info = {
            "type": "text-to-video",
            "total_tasks": concurrency,
            "success": len(all_frames),
            "failed": len(failed),
            "base_seed": base_seed,
            "seeds": seeds,
            "videos": [info for _, info in all_infos],
        }
        if failed:
            result_info["failed_details"] = [{"task_index": idx + 1, "error": err} for idx, err in failed]

        info_str = json.dumps(result_info, ensure_ascii=False, indent=2)
        print(f"[VolcEngine {log_prefix}] Done: {len(all_frames)}/{concurrency} success, {len(failed)} failed")

        return (combined_frames, combined_audio_dict, info_str)


# ─────────────────────────────────────────────
# Image-to-Video Node
# ─────────────────────────────────────────────

class VolcEngineImageToVideo:
    """VolcEngine Image-to-Video (supports multi-image + reference audio)

    When concurrency > 1: roll mode (same config, different seeds)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "VolcEngine Ark API Key",
                }),
                "model": (FALLBACK_MODELS, {"default": FALLBACK_MODELS[0]}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe the motion (can be left empty)",
                }),
                "concurrency": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                }),
                "video_duration": ("INT", {
                    "default": 5,
                    "min": 4,
                    "max": 15,
                    "step": 1,
                    "display": "number",
                }),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "aspect_ratio": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {"default": "adaptive"}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "service_tier": (["default", "flex"], {"default": "default"}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "poll_interval": ("INT", {"default": 10, "min": 3, "max": 30}),
                "max_wait": ("INT", {"default": 600, "min": 60, "max": 1800}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("Video Frames", "Audio", "Video Info")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "VolcEngine"

    def execute(self, api_key, model, image, prompt, concurrency, video_duration, resolution, aspect_ratio, generate_audio, camera_fixed, return_last_frame, service_tier, watermark,
                reference_audio=None, seed=-1, poll_interval=10, max_wait=600):

        if not api_key.strip():
            raise ValueError("Please fill in VolcEngine Ark API Key")
        if not model.strip():
            raise ValueError("Please select a model")

        num_images = image.shape[0]
        if num_images > 9:
            raise ValueError(f"Maximum 9 images supported, currently {num_images}")

        api = VolcEngineAPI(api_key.strip())
        import random

        # Mode description
        if num_images == 1:
            mode_desc = "first-frame"
        elif num_images == 2:
            mode_desc = "first-last-frame"
        else:
            mode_desc = f"multi-reference ({num_images})"

        if reference_audio is not None:
            mode_desc += "+reference-audio"

        # Generate seed list
        base_seed = seed if seed != -1 else random.randint(0, 2147483647)
        seeds = [base_seed + i for i in range(concurrency)]

        log_prefix = "Image-to-Video"

        desc_parts = [f"model={model}", f"mode={mode_desc}", f"tasks={concurrency}", f"duration={video_duration}s"]
        if service_tier == "flex":
            desc_parts.append("offline")
        print(f"[VolcEngine {log_prefix}] {' | '.join(desc_parts)}")

        def _run_single(idx, task_seed):
            payload = build_payload(
                model=model, prompt=prompt, image=image, audio=reference_audio,
                video_duration=video_duration, resolution=resolution, aspect_ratio=aspect_ratio,
                generate_audio=generate_audio, camera_fixed=camera_fixed, return_last_frame=return_last_frame,
                service_tier=service_tier, seed=task_seed, watermark=watermark
            )
            task_id = api.create_task(payload, api_key=api_key.strip())
            print(f"[VolcEngine {log_prefix}] #{idx+1}/{concurrency} seed={task_seed} | task_id={task_id}")

            result = api.poll_task(task_id, poll_interval=poll_interval, max_wait=max_wait, model_name=model)
            video_url = api.extract_video_url(result)

            print(f"[VolcEngine {log_prefix}] #{idx+1} fetching video and parsing frames...")
            frames_tensor, audio_dict, video_info = load_video_from_url(video_url, expected_audio=generate_audio)

            video_info["seed"] = task_seed
            video_info["task_index"] = idx + 1
            print(f"[VolcEngine {log_prefix}] #{idx+1} done: {video_info['total_frames']} frames, {video_info['duration']:.2f}s")

            return frames_tensor, audio_dict, video_info

        # Task list: (seed,)
        tasks = [(s,) for s in seeds]

        all_frames, all_audios, all_infos, failed = _concurrent_run(
            concurrency, concurrency, tasks, lambda idx, task: _run_single(idx, task[0]), log_prefix
        )

        combined_frames, combined_audio_dict = _combine_results(all_frames, all_audios)

        # Build info
        result_info = {
            "type": "image-to-video",
            "mode": mode_desc,
            "total_tasks": concurrency,
            "success": len(all_frames),
            "failed": len(failed),
            "base_seed": base_seed,
            "seeds": seeds,
            "videos": [info for _, info in all_infos],
        }
        if failed:
            result_info["failed_details"] = [{"task_index": idx + 1, "error": err} for idx, err in failed]

        info_str = json.dumps(result_info, ensure_ascii=False, indent=2)
        print(f"[VolcEngine {log_prefix}] Done: {len(all_frames)}/{concurrency} success, {len(failed)} failed")

        return (combined_frames, combined_audio_dict, info_str)


# ─────────────────────────────────────────────
# Image Generation Node
# ─────────────────────────────────────────────

class VolcEngineImageGeneration:
    """VolcEngine Image Generation

    Supports:
    - Text-to-image: generate from prompt
    - Image-to-image: reference image + prompt
    - Concurrent generation: same config, different seeds

    API Endpoint: /api/v3/images/generations (synchronous)
    """

    IMAGE_API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "VolcEngine Ark API Key",
                }),
                "model": (FALLBACK_IMAGE_MODELS, {"default": FALLBACK_IMAGE_MODELS[0]}),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe the image content you want to generate",
                }),
                "concurrency": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                }),
                "image_size": (["1024x1024", "2048x2048", "3072x3072", "4096x4096", "1280x720", "1920x1080"], {"default": "1024x1024"}),
                "service_tier": (["default", "flex"], {"default": "default"}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("Image", "Generation Info")
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "VolcEngine"

    def execute(self, api_key, model, prompt, concurrency, image_size, service_tier, watermark,
                reference_image=None, seed=-1):

        if not api_key.strip():
            raise ValueError("Please fill in VolcEngine Ark API Key")
        if not model.strip():
            raise ValueError("Please select a model")
        if not prompt.strip() and reference_image is None:
            raise ValueError("Please fill in prompt or provide reference image")

        import random

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

        # Generate seed list
        base_seed = seed if seed != -1 else random.randint(0, 2147483647)
        seeds = [base_seed + i for i in range(concurrency)]

        log_prefix = "Image Generation"

        mode_desc = "image-to-image" if reference_image is not None else "text-to-image"
        print(f"[VolcEngine {log_prefix}] model={model} | mode={mode_desc} | tasks={concurrency} | size={image_size}")

        # Capture outer self for closure
        outer_self = self

        def _run_single(idx, task_seed):
            payload = {
                "model": model.strip(),
                "prompt": prompt.strip(),
                "size": image_size,
                "watermark": watermark,
                "response_format": "url",
            }
            if service_tier != "default":
                payload["service_tier"] = service_tier

            if task_seed != -1:
                payload["seed"] = task_seed

            # Reference image: convert to URL or base64
            if reference_image is not None:
                b64_list = tensors_to_base64_list(reference_image)
                if len(b64_list) == 1:
                    payload["image"] = f"data:image/jpeg;base64,{b64_list[0]}"
                else:
                    for i, b64 in enumerate(b64_list[:10]):
                        if i == 0:
                            payload["image"] = f"data:image/jpeg;base64,{b64}"
                        else:
                            payload[f"image{i+1}"] = f"data:image/jpeg;base64,{b64}"

            print(f"[VolcEngine {log_prefix}] #{idx+1}/{concurrency} seed={task_seed} | requesting...")

            # Call image generation API (synchronous)
            resp = requests.post(outer_self.IMAGE_API_URL, json=payload, headers=headers, timeout=120)
            if not resp.ok:
                error_hint = ""
                try:
                    error_data = resp.json()
                    error_code = ""
                    if isinstance(error_data, dict):
                        err_obj = error_data.get("error", error_data)
                        error_code = err_obj.get("code", "") if isinstance(err_obj, dict) else ""
                    if resp.status_code == 429:
                        error_hint = "\n⚠️ API rate limited. Please retry later or reduce concurrency."
                    if resp.status_code == 402:
                        error_hint = "\n⚠️ Insufficient account balance or expired plan. Please recharge."
                    if resp.status_code == 401:
                        error_hint = "\n⚠️ API Key is invalid or expired. Please check your key."
                except Exception:
                    pass
                raise RuntimeError(f"Image generation failed [{resp.status_code}]: {resp.text}{error_hint}")

            result = resp.json()

            # Extract image URL
            data = result.get("data", [])
            if not data:
                raise RuntimeError(f"No image data returned: {json.dumps(result, ensure_ascii=False)}")

            image_url = data[0].get("url") or data[0].get("b64_json")
            if not image_url:
                raise RuntimeError(f"No image URL found: {json.dumps(result, ensure_ascii=False)}")

            # Download image
            if image_url.startswith("http"):
                img_resp = requests.get(image_url, timeout=60)
                img_resp.raise_for_status()
                img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            else:
                # base64
                img_data = base64.b64decode(image_url)
                img = Image.open(io.BytesIO(img_data)).convert("RGB")

            # Convert to tensor
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, C]

            info = {
                "seed": task_seed,
                "task_index": idx + 1,
                "width": img.width,
                "height": img.height,
            }
            print(f"[VolcEngine {log_prefix}] #{idx+1} done: {img.width}x{img.height}")

            return img_tensor, info

        # Concurrent execution
        import concurrent.futures
        all_images = []
        all_infos = []
        failed = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {}
            for idx, task_seed in enumerate(seeds):
                future = executor.submit(_run_single, idx, task_seed)
                future_map[future] = idx

            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                try:
                    img_tensor, info = future.result()
                    all_images.append((idx, img_tensor))
                    all_infos.append((idx, info))
                except Exception as e:
                    print(f"[VolcEngine {log_prefix}] #{idx+1} failed: {e}")
                    failed.append((idx, str(e)))

        if not all_images:
            raise RuntimeError(f"All {concurrency} tasks failed")

        # Sort and merge
        all_images.sort(key=lambda x: x[0])
        all_infos.sort(key=lambda x: x[0])
        combined_images = torch.cat([img for _, img in all_images], dim=0)

        # Build info
        result_info = {
            "type": "image-generation",
            "mode": mode_desc,
            "model": model,
            "size": image_size,
            "total_tasks": concurrency,
            "success": len(all_images),
            "failed": len(failed),
            "base_seed": base_seed,
            "seeds": seeds,
            "images": [info for _, info in all_infos],
        }
        if failed:
            result_info["failed_details"] = [{"task_index": idx + 1, "error": err} for idx, err in failed]

        info_str = json.dumps(result_info, ensure_ascii=False, indent=2)
        print(f"[VolcEngine {log_prefix}] Done: {len(all_images)}/{concurrency} success, {len(failed)} failed")

        return (combined_images, info_str)


# ─────────────────────────────────────────────
# Registration Mappings
# ─────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "VolcEngineTextToVideo": VolcEngineTextToVideo,
    "VolcEngineImageToVideo": VolcEngineImageToVideo,
    "VolcEngineImageGeneration": VolcEngineImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VolcEngineTextToVideo": "🎬 VolcEngine Text to Video",
    "VolcEngineImageToVideo": "🎬 VolcEngine Image to Video",
    "VolcEngineImageGeneration": "🖼️ VolcEngine Image Generation",
}
