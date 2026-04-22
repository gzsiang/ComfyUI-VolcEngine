<h4 align="center">
  English | <a href="./README.zh.md">中文</a>
</h4>

<br />

# ComfyUI VolcEngine Node

🎬 **ComfyUI Custom Node for VolcEngine (火山引擎) Video & Image Generation**

Custom nodes for video/image generation via VolcEngine Ark API, supporting Seedance (video) and Seedream (image) model series.

English | [中文](README.zh.md)

---

## ✨ Features

### Video Generation
- **Text-to-Video**: Generate videos from text descriptions
- **Image-to-Video**: Supports single image (first frame), dual images (start/end frames), and multi-image (reference) modes
- **Reference Audio**: Pass audio as voice reference (Seedance 2.0)
- **Parallel Generation**: Generate multiple seeds with the same prompt (multi-roll mode)

### Image Generation
- **Text-to-Image**: Generate images from text descriptions, up to 4K resolution
- **Image-to-Image**: Generate new images from reference images + prompts
- **Multi-Reference**: Supports up to 10 reference images

### General
- **Multi-Model Support**: Seedance 2.0/1.5/1.0 (video), Seedream 5.0/4.5/4.0/3.1 (image)
- **Full Parameters**: Resolution, aspect ratio, audio generation, locked camera, service tier, etc.
- **Auto Model List**: Automatically fetches available models when a 404 error occurs

---

## 📦 Installation

### Method 1: Clone to custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gzsiang/ComfyUI-VolcEngine.git
```

### Method 2: Manual Install

1. Download this repository
2. Extract to `ComfyUI/custom_nodes/ComfyUI-VolcEngine/`
3. Restart ComfyUI

---

## 🔑 Getting an API Key

1. Visit [VolcEngine Console](https://console.volcengine.com/)
2. Go to the "Ark" service
3. Create an API Key
4. Enable model access:
   - Video generation: Seedance series
   - Image generation: Seedream series

---

## 🚀 Usage

### Nodes

| Node Name | Function | Inputs |
|-----------|----------|--------|
| 🎬 VolcEngine Text-to-Video | Text to video | API Key, model, prompt, parameters |
| 🎬 VolcEngine Image-to-Video | Image to video | API Key, model, image, prompt, parameters |
| 🖼️ VolcEngine Image Generation | Text/image to image | API Key, model, prompt, parameters |

### Image-to-Video Modes

Automatically determined by input image count:

| Image Count | Mode | Description |
|-------------|------|-------------|
| 1 image | First-frame mode | Uses image as video start frame |
| 2 images | Start/end frame mode | First image = start, second = end |
| 3-9 images | Multi-reference mode | Used as style/content reference |

### Reference Audio

- When audio is provided, the generated video will use the audio's voice
- **Note**: Must provide at least 1 image when using reference audio

---

## ⚙️ Parameters

### Video Generation Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| API Key | PASSWORD | - | - | VolcEngine Ark API Key (password input) |
| Model | Dropdown | - | doubao-seedance-2-0-260128 | Video generation model |
| Prompt | String | - | - | Video content description |
| Concurrency | Int | 1-5 | 1 | Number of parallel generations per prompt |
| Video Duration (s) | Int | 4-15 | 5 | Video length |
| Resolution | Dropdown | 480p/720p | 720p | Video resolution |
| Aspect Ratio | Dropdown | Various | 16:9 (text-to-video) / adaptive (image-to-video) | Video aspect ratio |
| Watermark | Boolean | - | True | Whether to add watermark |
| Random Seed | Int | -1~∞ | -1 | -1 for random |
| Poll Interval | Int | 3-30 | 10 | Task status check interval (seconds) |
| Max Wait | Int | 60-1800 | 600 | Maximum wait time (seconds) |

### Seedance 2.0 Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| Generate Audio | Boolean | True | Whether to generate video audio |
| Locked Camera | Boolean | False | Lock camera perspective |
| Return Last Frame | Boolean | False | Also return the last frame image |
| Service Tier | Dropdown | default | default (standard) / flex (offline, ~50% cheaper) |

### Image Generation Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| API Key | PASSWORD | - | - | VolcEngine Ark API Key (password input) |
| Model | Dropdown | - | doubao-seedream-4-5-251128 | Image generation model |
| Prompt | String | - | - | Image content description |
| Reference Image | Image | - | - | Optional, for image-to-image |
| Concurrency | Int | 1-5 | 1 | Number of parallel generations per prompt |
| Image Size | Dropdown | Various | 1024x1024 | Output image size |
| Service Tier | Dropdown | - | default | default (standard) / flex (offline, ~50% cheaper) |
| Watermark | Boolean | - | False | Whether to add watermark |
| Random Seed | Int | -1~∞ | -1 | -1 for random |

---

## 📋 Supported Models

### Video Generation - Seedance Series

#### Seedance 2.0 (Recommended)
- `doubao-seedance-2-0-260128` - Latest version, supports multi-modal references

#### Seedance 1.5
- `doubao-seedance-1-5-pro-251215`

#### Seedance 1.0
- `doubao-seedance-1-0-pro-250528`
- `doubao-seedance-1-0-pro`
- `doubao-seedance-1-0-pro-fast`
- `doubao-seedance-1-0-lite`

### Image Generation - Seedream Series

- `doubao-seedream-5-0`
- `doubao-seedream-4-5-251128`
- `doubao-seedream-4-0-250828`
- `doubao-seedream-3-1-250312`

---

## 📁 Workflow Examples

The repository includes example workflow files:

- `workflows/火山API测试.json` - Basic test workflow (text-to-video + image-to-video)

Import: ComfyUI interface → Load → select the json file

---

## 🔄 Output Format

### Video Nodes

| Port | Type | Description |
|------|------|-------------|
| Video Frames | IMAGE | Video frame sequence, can connect to VHS or other nodes for preview/save |
| Audio | AUDIO | Video audio, can connect to audio output nodes |
| Video Info | STRING | JSON metadata (fps, frame count, duration, etc.) |

### Image Nodes

| Port | Type | Description |
|------|------|-------------|
| Image | IMAGE | Generated image sequence |
| Generation Info | STRING | JSON metadata (size, seed, etc.) |

---

## ⚠️ Notes

1. **API Key Security**: Do not share your API Key when sharing workflows
2. **Image Limits**: Image-to-video supports up to 9 images, image generation supports up to 10 reference images
3. **Resolution Limits**: Seedance 2.0 max 720p, older versions max 1080p; Seedream max 4K
4. **Audio Reference**: Must provide an image to use reference audio
5. **Service Tier**: `flex` mode uses offline inference, ~50% cheaper but may have longer wait times
6. **Concurrency**: Max 5 concurrent requests. With >1, multiple videos/images are generated simultaneously — watch API costs

---

## 📄 License

MIT License

---

## 🔗 Related Links

- [VolcEngine Documentation](https://www.volcengine.com/docs/82379/1520757)
- [ComfyUI Official Repo](https://github.com/comfyanonymous/ComfyUI)
