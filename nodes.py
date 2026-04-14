"""
ComfyUI 火山引擎 视频生成节点

支持 Seedance 系列，文生视频 + 图生视频（多图 + 参考音频）

API 参考：https://www.volcengine.com/docs/82379/1520757
接口：POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks

Seedance 2.0 多模态参考：
  - 图片 (1~9张): role=first_frame / last_frame / reference_image
  - 音频 (1~3段): role=reference_audio（必须至少包含1个参考图片）
  - 文本提示词

输出：视频帧(IMAGE) + 音频(AUDIO) + 视频信息(STRING)
"""

import os
import io
import time
import json
import base64
import requests
import numpy as np
from PIL import Image
import folder_paths
import torch


# ─────────────────────────────────────────────
# 公共资源：预置模型列表（兜底）
# ─────────────────────────────────────────────

FALLBACK_MODELS = [
    # Seedance 2.0 系列（从 API 返回确认的名称）
    "doubao-seedance-2-0-260128",
    # Seedance 1.5 系列
    "doubao-seedance-1-5-pro-251215",
    # Seedance 1.0 系列
    "doubao-seedance-1-0-pro-250528",
    "doubao-seedance-1-0-pro",
    "doubao-seedance-1-0-pro-fast",
    "doubao-seedance-1-0-lite",
    # Seedream 系列
    "doubao-seedream-5-0",
    "doubao-seedream-4-5-251128",
    "doubao-seedream-4-0-250828",
    "doubao-seedream-3-1-250312",
]

# 全局缓存：api_key → [model_id列表]
MODEL_CACHE = {}


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def tensor_to_base64(image_tensor) -> str:
    """ComfyUI IMAGE tensor (1,H,W,C) → base64 JPEG 字符串"""
    img_np = image_tensor[0].cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def tensors_to_base64_list(image_tensor) -> list[str]:
    """ComfyUI IMAGE tensor (B,H,W,C) → base64 JPEG 列表（每张图一个）"""
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
    """ComfyUI AUDIO dict → base64 WAV 字符串"""
    import wave

    waveform = audio_dict.get("waveform")
    sample_rate = audio_dict.get("sample_rate", 44100)

    if waveform is None:
        raise ValueError("音频数据为空")

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


def load_video_from_url(url: str):
    """
    从 URL 下载视频并解析为帧(IMAGE tensor)和音频(AUDIO tensor)
    返回: (frames_tensor, audio_dict, video_info_dict)
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
            raise ValueError("无法读取视频帧")

        frames_tensor = torch.stack(frames)  # [T, H, W, 3]

        # 提取音频
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
        except Exception as e:
            print(f"[火山引擎] 音频提取失败或视频无音频: {e}")
            # ComfyUI AUDIO format: [batch, channels, samples]
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
# 火山引擎 API 客户端
# ─────────────────────────────────────────────

class 火山引擎API:
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def create_task(self, payload: dict) -> str:
        """创建视频生成任务，返回 task_id"""
        url = f"{self.BASE_URL}/tasks"
        resp = requests.post(url, json=payload, headers=self.headers, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"创建任务失败 [{resp.status_code}]: {resp.text}")
        data = resp.json()
        task_id = data.get("id") or data.get("task_id")
        if not task_id:
            raise RuntimeError(f"未获取到 task_id，响应: {json.dumps(data, ensure_ascii=False)}")
        return task_id

    def poll_task(self, task_id: str, poll_interval: int = 10, max_wait: int = 600) -> dict:
        """轮询任务状态，直到成功或超时"""
        url = f"{self.BASE_URL}/tasks/{task_id}"
        start = time.time()
        while time.time() - start < max_wait:
            resp = requests.get(url, headers=self.headers, timeout=30)
            if not resp.ok:
                raise RuntimeError(f"查询任务失败 [{resp.status_code}]: {resp.text}")
            data = resp.json()
            status = data.get("status", "")
            if status == "succeeded":
                return data
            if status == "failed":
                error = data.get("error", {})
                raise RuntimeError(f"任务失败: {error.get('message', json.dumps(data, ensure_ascii=False))}")
            print(f"[火山引擎] 任务状态: {status}，{poll_interval}秒后重试...")
            time.sleep(poll_interval)
        raise TimeoutError(f"任务超时（{max_wait}秒），task_id={task_id}")

    def extract_video_url(self, result: dict) -> str:
        """从任务结果中提取视频 URL"""
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

        # 新版 API: content 是 dict，直接取 video_url
        content = result.get("content")
        if isinstance(content, dict):
            url = content.get("video_url") or content.get("url")
            if url:
                return url
        # 旧版兼容: content 是 list
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    url = item.get("url") or item.get("video_url")
                    if url:
                        return url

        raise RuntimeError(f"未能从结果中提取视频 URL: {json.dumps(result, ensure_ascii=False)[:500]}")


# ─────────────────────────────────────────────
# 构建 API 请求 payload
# ─────────────────────────────────────────────

def build_payload(模型, 提示词="", 图片=None, 音频=None, 视频时长=5,
                  分辨率="720p", 宽高比="adaptive", 生成音频=True,
                  固定摄像头=False, 返回尾帧=False, 服务等级="default",
                  随机种子=-1, 水印=True):
    """
    构建 Seedance 2.0 多模态请求 payload

    兼容新旧 API：
    - Seedance 2.0: 使用 input 数组 + 顶层参数
    - 旧版: 使用 content 数组 + 提示词内嵌参数
    """
    is_v2 = "2-0" in 模型

    if is_v2:
        # ── Seedance 2.0: content 数组 + 顶层参数 ──
        content_list = []

        # 文本提示词
        if 提示词.strip():
            content_list.append({"type": "text", "text": 提示词.strip()})

        # 图片：根据数量自动判断 role
        if 图片 is not None:
            num_images = 图片.shape[0]
            b64_list = tensors_to_base64_list(图片)

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

        # 参考音频 (role=reference_audio，必须至少有1张图片)
        if 音频 is not None:
            audio_b64 = audio_to_base64(音频)
            content_list.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
                "role": "reference_audio",
            })

        payload = {
            "model": 模型.strip(),
            "content": content_list,
            "duration": 视频时长,
            "resolution": 分辨率,
            "ratio": 宽高比,
            "generate_audio": 生成音频,
            "camera_fixed": 固定摄像头,
            "return_last_frame": 返回尾帧,
            "watermark": 水印,
        }
        # service_tier 只在图生视频时支持
        if 图片 is not None and 服务等级 != "default":
            payload["service_tier"] = 服务等级
        if 随机种子 != -1:
            payload["seed"] = 随机种子

    else:
        # ── 旧版 Seedance 1.x: content 数组 ──
        param_str = f" --duration {视频时长}"
        if 随机种子 != -1:
            param_str += f" --seed {随机种子}"
        param_str += " --watermark true" if 水印 else " --watermark false"

        text_content = 提示词.strip() if 提示词.strip() else ""
        text_content += param_str

        content = [{"type": "text", "text": text_content}]

        if 图片 is not None:
            num_images = 图片.shape[0]
            b64_list = tensors_to_base64_list(图片)

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

        payload = {"model": 模型.strip(), "content": content}

    return payload


# ─────────────────────────────────────────────
# 文生视频节点
# ─────────────────────────────────────────────

class 火山引擎文生视频:
    """火山引擎 文生视频（纯文本生成视频）→ 输出帧+音频+信息"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "火山引擎方舟 API Key",
                }),
                "模型": (FALLBACK_MODELS, {"default": FALLBACK_MODELS[0]}),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "描述你想生成的视频内容",
                }),
                "视频时长秒": ("INT", {
                    "default": 5,
                    "min": 4,
                    "max": 15,
                    "step": 1,
                    "display": "number",
                }),
                "分辨率": (["480p", "720p"], {"default": "720p"}),
                "宽高比": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {"default": "16:9"}),
                "生成音频": ("BOOLEAN", {"default": True}),
                "固定摄像头": ("BOOLEAN", {"default": False}),
                "返回尾帧": ("BOOLEAN", {"default": False}),
                "服务等级": (["default", "flex"], {"default": "default"}),
                "水印": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "轮询间隔": ("INT", {"default": 10, "min": 3, "max": 30}),
                "最大等待": ("INT", {"default": 600, "min": 60, "max": 1800}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("视频帧", "音频", "视频信息")
    OUTPUT_NODE = True
    FUNCTION = "生成"
    CATEGORY = "火山引擎"

    def 生成(self, api_key, 模型, 提示词, 视频时长秒, 分辨率, 宽高比, 生成音频, 固定摄像头, 返回尾帧, 服务等级, 水印,
             随机种子=-1, 轮询间隔=10, 最大等待=600):

        if not api_key.strip():
            raise ValueError("请填入火山引擎方舟 API Key")
        if not 模型.strip():
            raise ValueError("请选择模型")
        if not 提示词.strip():
            raise ValueError("请填入提示词")

        api = 火山引擎API(api_key.strip())

        payload = build_payload(
            模型=模型, 提示词=提示词, 图片=None, 音频=None,
            视频时长=视频时长秒, 分辨率=分辨率, 宽高比=宽高比,
            生成音频=生成音频, 固定摄像头=固定摄像头, 返回尾帧=返回尾帧,
            服务等级=服务等级, 随机种子=随机种子, 水印=水印
        )

        desc_parts = [f"模型={模型}", f"时长={视频时长秒}s", f"分辨率={分辨率}", f"宽高比={宽高比}", f"水印={水印}"]
        if 服务等级 == "flex":
            desc_parts.append("离线推理")
        print(f"[火山引擎 文生视频] {' | '.join(desc_parts)}")

        task_id = api.create_task(payload)
        print(f"[火山引擎 文生视频] 任务已创建 | task_id={task_id}")

        result = api.poll_task(task_id, poll_interval=轮询间隔, max_wait=最大等待)
        video_url = api.extract_video_url(result)

        print(f"[火山引擎 文生视频] 获取视频并解析帧...")
        frames_tensor, audio_dict, video_info = load_video_from_url(video_url)

        info_str = json.dumps(video_info, ensure_ascii=False, indent=2)
        print(f"[火山引擎 文生视频] 解析完成: {video_info['total_frames']}帧, {video_info['duration']:.2f}秒")

        return (frames_tensor, audio_dict, info_str)


# ─────────────────────────────────────────────
# 图生视频节点
# ─────────────────────────────────────────────

class 火山引擎图生视频:
    """火山引擎 图生视频（支持多图+参考音频）→ 输出帧+音频+信息"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "火山引擎方舟 API Key",
                }),
                "模型": (FALLBACK_MODELS, {"default": FALLBACK_MODELS[0]}),
                "图片": ("IMAGE",),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "描述运动方式（可留空）",
                }),
                "视频时长秒": ("INT", {
                    "default": 5,
                    "min": 4,
                    "max": 15,
                    "step": 1,
                    "display": "number",
                }),
                "分辨率": (["480p", "720p"], {"default": "720p"}),
                "宽高比": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {"default": "adaptive"}),
                "生成音频": ("BOOLEAN", {"default": True}),
                "固定摄像头": ("BOOLEAN", {"default": False}),
                "返回尾帧": ("BOOLEAN", {"default": False}),
                "服务等级": (["default", "flex"], {"default": "default"}),
                "水印": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "参考音频": ("AUDIO",),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "轮询间隔": ("INT", {"default": 10, "min": 3, "max": 30}),
                "最大等待": ("INT", {"default": 600, "min": 60, "max": 1800}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("视频帧", "音频", "视频信息")
    OUTPUT_NODE = True
    FUNCTION = "生成"
    CATEGORY = "火山引擎"

    def 生成(self, api_key, 模型, 图片, 提示词, 视频时长秒, 分辨率, 宽高比, 生成音频, 固定摄像头, 返回尾帧, 服务等级, 水印,
             参考音频=None, 随机种子=-1, 轮询间隔=10, 最大等待=600):

        if not api_key.strip():
            raise ValueError("请填入火山引擎方舟 API Key")
        if not 模型.strip():
            raise ValueError("请选择模型")

        num_images = 图片.shape[0]
        if num_images > 9:
            raise ValueError(f"最多支持 9 张图片，当前 {num_images} 张")

        api = 火山引擎API(api_key.strip())

        payload = build_payload(
            模型=模型, 提示词=提示词, 图片=图片, 音频=参考音频,
            视频时长=视频时长秒, 分辨率=分辨率, 宽高比=宽高比,
            生成音频=生成音频, 固定摄像头=固定摄像头, 返回尾帧=返回尾帧,
            服务等级=服务等级, 随机种子=随机种子, 水印=水印
        )

        # 描述模式
        if num_images == 1:
            mode_desc = "首帧模式"
        elif num_images == 2:
            mode_desc = "首尾帧模式"
        else:
            mode_desc = f"多参考图模式 ({num_images}张)"

        if 参考音频 is not None:
            mode_desc += "+参考音频"

        desc_parts = [f"模型={模型}", f"模式={mode_desc}", f"时长={视频时长秒}s", f"分辨率={分辨率}", f"宽高比={宽高比}", f"水印={水印}"]
        if 服务等级 == "flex":
            desc_parts.append("离线推理")
        print(f"[火山引擎 图生视频] {' | '.join(desc_parts)}")

        task_id = api.create_task(payload)
        print(f"[火山引擎 图生视频] 任务已创建 | task_id={task_id}")

        result = api.poll_task(task_id, poll_interval=轮询间隔, max_wait=最大等待)
        video_url = api.extract_video_url(result)

        print(f"[火山引擎 图生视频] 获取视频并解析帧...")
        frames_tensor, audio_dict, video_info = load_video_from_url(video_url)

        info_str = json.dumps(video_info, ensure_ascii=False, indent=2)
        print(f"[火山引擎 图生视频] 解析完成: {video_info['total_frames']}帧, {video_info['duration']:.2f}秒")

        return (frames_tensor, audio_dict, info_str)


# ─────────────────────────────────────────────
# 并发抽卡辅助函数
# ─────────────────────────────────────────────

def _concurrent_run(任务数, 并发数, seeds, run_fn, log_prefix):
    """并发执行抽卡任务，返回排序后的结果和失败列表"""
    import concurrent.futures

    all_frames = []
    all_audios = []
    all_infos = []
    failed = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(并发数, 任务数)) as executor:
        future_map = {}
        for idx, seed in enumerate(seeds):
            future = executor.submit(run_fn, idx, seed)
            future_map[future] = idx

        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            try:
                frames_tensor, audio_dict, video_info = future.result()
                all_frames.append((idx, frames_tensor))
                all_audios.append((idx, audio_dict))
                all_infos.append((idx, video_info))
            except Exception as e:
                print(f"[火山引擎 {log_prefix}] #{idx+1} 失败: {e}")
                failed.append((idx, str(e)))

    if not all_frames:
        raise RuntimeError(f"所有 {任务数} 次抽卡均失败")

    # 按原始顺序排列
    all_frames.sort(key=lambda x: x[0])
    all_audios.sort(key=lambda x: x[0])
    all_infos.sort(key=lambda x: x[0])

    return all_frames, all_audios, all_infos, failed


def _combine_results(all_frames, all_audios):
    """合并多组结果为一个批次"""
    # 拼接所有帧为一个批次
    combined_frames = torch.cat([f for _, f in all_frames], dim=0)

    # 拼接所有音频
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
# 通用抽卡节点（文生/图生 合一）
# ─────────────────────────────────────────────

class 火山引擎并发:
    """火山引擎 并发生成节点（接图片=图生并发，不接=文生并发）

    同一提示词并发 N 次，每次使用不同种子，提高出片效率。
    如需重复生成，直接在 ComfyUI 中复制节点连线即可。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "火山引擎方舟 API Key",
                }),
                "模型": (FALLBACK_MODELS, {"default": FALLBACK_MODELS[0]}),
                "提示词": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "描述你想生成的视频内容",
                }),
                "并发数": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                }),
                "视频时长秒": ("INT", {
                    "default": 5,
                    "min": 4,
                    "max": 15,
                    "step": 1,
                    "display": "number",
                }),
                "分辨率": (["480p", "720p"], {"default": "720p"}),
                "宽高比": (["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"], {"default": "16:9"}),
                "生成音频": ("BOOLEAN", {"default": True}),
                "固定摄像头": ("BOOLEAN", {"default": False}),
                "服务等级": (["default", "flex"], {"default": "default"}),
                "水印": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "图片": ("IMAGE",),
                "参考音频": ("AUDIO",),
                "随机种子": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "轮询间隔": ("INT", {"default": 10, "min": 3, "max": 30}),
                "最大等待": ("INT", {"default": 600, "min": 60, "max": 1800}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("视频帧", "音频", "视频信息")
    OUTPUT_NODE = True
    FUNCTION = "生成"
    CATEGORY = "火山引擎"

    def 生成(self, api_key, 模型, 提示词, 并发数, 视频时长秒, 分辨率, 宽高比, 生成音频, 固定摄像头, 服务等级, 水印,
             图片=None, 参考音频=None, 随机种子=-1, 轮询间隔=10, 最大等待=600):

        if not api_key.strip():
            raise ValueError("请填入火山引擎方舟 API Key")
        if not 模型.strip():
            raise ValueError("请选择模型")

        is_image_mode = 图片 is not None
        mode_prefix = "图生并发" if is_image_mode else "文生并发"

        if not 提示词.strip() and not is_image_mode:
            raise ValueError("请填入提示词")

        # 图片校验
        if is_image_mode:
            num_images = 图片.shape[0]
            if num_images > 9:
                raise ValueError(f"最多支持 9 张图片，当前 {num_images} 张")
            if num_images == 1:
                img_mode_desc = "首帧模式"
            elif num_images == 2:
                img_mode_desc = "首尾帧模式"
            else:
                img_mode_desc = f"多参考图模式 ({num_images}张)"
        else:
            img_mode_desc = None

        api = 火山引擎API(api_key.strip())
        import random

        task_count = 并发数
        base_seed = 随机种子 if 随机种子 != -1 else random.randint(0, 2147483647)
        seeds = [base_seed + i for i in range(task_count)]

        def _run_single(idx, seed):
            payload = build_payload(
                模型=模型, 提示词=提示词, 图片=图片, 音频=参考音频,
                视频时长=视频时长秒, 分辨率=分辨率, 宽高比=宽高比,
                生成音频=生成音频, 固定摄像头=固定摄像头, 返回尾帧=False,
                服务等级=服务等级 if is_image_mode else "default",
                随机种子=seed, 水印=水印
            )
            task_id = api.create_task(payload)
            print(f"[火山引擎 {mode_prefix}] #{idx+1}/{task_count} seed={seed} | task_id={task_id}")

            result = api.poll_task(task_id, poll_interval=轮询间隔, max_wait=最大等待)
            video_url = api.extract_video_url(result)

            print(f"[火山引擎 {mode_prefix}] #{idx+1} 获取视频并解析帧...")
            frames_tensor, audio_dict, video_info = load_video_from_url(video_url)

            video_info["seed"] = seed
            video_info["task_index"] = idx + 1
            print(f"[火山引擎 {mode_prefix}] #{idx+1} 完成: {video_info['total_frames']}帧, {video_info['duration']:.2f}秒")

            return frames_tensor, audio_dict, video_info

        print(f"[火山引擎 {mode_prefix}] 并发 {task_count} 次 | base_seed={base_seed}"
              + (f" | {img_mode_desc}" if img_mode_desc else ""))

        all_frames, all_audios, all_infos, failed = _concurrent_run(
            task_count, 并发数, seeds, _run_single, mode_prefix
        )

        combined_frames, combined_audio_dict = _combine_results(all_frames, all_audios)

        # 构建信息
        result_info = {
            "type": "图生视频" if is_image_mode else "文生视频",
            "concurrent": task_count,
            "success": len(all_frames),
            "failed": len(failed),
            "base_seed": base_seed,
            "seeds": seeds,
            "videos": [info for _, info in all_infos],
        }
        if img_mode_desc:
            result_info["image_mode"] = img_mode_desc
        if failed:
            result_info["failed_details"] = [{"task_index": idx + 1, "error": err} for idx, err in failed]

        info_str = json.dumps(result_info, ensure_ascii=False, indent=2)
        print(f"[火山引擎 {mode_prefix}] 完成: {len(all_frames)}/{task_count} 成功, {len(failed)} 失败")

        return (combined_frames, combined_audio_dict, info_str)


# ─────────────────────────────────────────────
# 注册映射
# ─────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "火山引擎文生视频": 火山引擎文生视频,
    "火山引擎图生视频": 火山引擎图生视频,
    "火山引擎并发": 火山引擎并发,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "火山引擎文生视频": "🎬 火山引擎 文生视频",
    "火山引擎图生视频": "🎬 火山引擎 图生视频",
    "火山引擎并发": "⚡ 火山引擎 并发",
}
