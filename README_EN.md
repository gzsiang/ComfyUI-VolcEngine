# ComfyUI VolcEngine Generation Nodes

🎬 **ComfyUI Custom Node for VolcEngine Video & Image Generation**

Custom nodes for video/image generation based on VolcEngine Ark API, supporting Seedance (video) and Seedream (image) model series.

[中文文档](README.md) | English

---

## ✨ Features

### Video Generation
- **Text-to-Video**: Generate videos from pure text descriptions
- **Image-to-Video**: Support single image (first frame), dual images (first/last frames), and multiple images (reference) modes
- **Reference Audio**: Support audio input as character voice reference (Seedance 2.0)
- **Concurrent Generation**: Multiple seed concurrent generation for the same prompt (gacha mode)

### Image Generation
- **Text-to-Image**: Generate images from text descriptions, supporting up to 4K resolution
- **Image-to-Image**: Generate new images from reference images + prompts
- **Multiple Reference Images**: Support up to 10 reference images

### General Features
- **Multi-Model Support**: Seedance 2.0/1.5/1.0 (video), Seedream 5.0/4.5/4.0/3.1 (image)
- **Complete Parameters**: Resolution, aspect ratio, audio generation, fixed camera, service tier, etc.
- **Auto Model Detection**: Automatically fetch available model list when model 404 error occurs

---

## 📦 Installation

### Method 1: Clone to custom_nodes

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gzsiang/ComfyUI-VolcEngine.git
```

### Method 2: Manual Installation

1. Download this repository
2. Extract to `ComfyUI/custom_nodes/ComfyUI-VolcEngine/`
3. Restart ComfyUI

---

## 🔑 Get API Key

1. Visit [VolcEngine Console](https://console.volcengine.com/)
2. Enter "Ark" service
3. Create API Key
4. Enable model permissions:
   - Video Generation: Seedance series
   - Image Generation: Seedream series

---

## 🚀 Usage

### Node Description

| Node Name | Function | Inputs |
|---------|------|------|
| 🎬 VolcEngine Text-to-Video | Text to video generation | API Key, Model, Prompt, Parameters |
| 🎬 VolcEngine Image-to-Video | Image to video generation | API Key, Model, Image, Prompt, Parameters |
| 🖼️ VolcEngine Image Generation | Text/Image to image generation | API Key, Model, Prompt, Parameters |

### Image-to-Video Modes

Automatically determined by input image count:

| Image Count | Mode | Description |
|---------|------|------|
| 1 | First Frame Mode | Use this image as video starting frame |
| 2 | First/Last Frame Mode | First image as start, second as end |
| 3-9 | Multi-Reference Mode | Used as style/content reference |

### Reference Audio

- After inputting audio, generated video will use the audio's voice characteristics
- **Note**: When using reference audio, you must also input at least 1 image

---

## ⚙️ Parameter Description

### Video Generation Parameters

| Parameter | Type | Range | Default | Description |
|-----|------|------|-------|------|
| API Key | String | - | - | VolcEngine Ark API Key |
| Model | Dropdown | - | doubao-seedance-2-0-260128 | Video generation model |
| Prompt | String | - | - | Video content description |
| Concurrent Count | Int | 1-10 | 1 | Concurrent generation count for same prompt (gacha) |
| Video Duration (sec) | Int | 4-15 | 5 | Generated video duration |
| Resolution | Dropdown | 480p/720p | 720p | Video resolution |
| Aspect Ratio | Dropdown | Various | 16:9 (T2V) / adaptive (I2V) | Video aspect ratio |
| Watermark | Boolean | - | True | Whether to add watermark |
| Random Seed | Int | -1~∞ | -1 | -1 for random |
| Poll Interval | Int | 3-30 | 10 | Task status query interval (seconds) |
| Max Wait | Int | 60-1800 | 600 | Maximum wait time (seconds) |

### Seedance 2.0 Specific Parameters

| Parameter | Type | Default | Description |
|-----|------|-------|------|
| Generate Audio | Boolean | True | Whether to generate video audio |
| Fixed Camera | Boolean | False | Fixed camera perspective |
| Return Last Frame | Boolean | False | Also return the last frame image |
| Service Tier | Dropdown | default | default (standard) / flex (offline, ~50% cheaper) |

### Image Generation Parameters

| Parameter | Type | Range | Default | Description |
|-----|------|------|-------|------|
| API Key | String | - | - | VolcEngine Ark API Key |
| Model | Dropdown | - | doubao-seedream-4-5-251128 | Image generation model |
| Prompt | String | - | - | Image content description |
| Reference Image | Image | - | - | Optional, for image-to-image |
| Concurrent Count | Int | 1-10 | 1 | Concurrent generation count for same prompt |
| Image Size | Dropdown | Various | 1024x1024 | Output image size |
| Watermark | Boolean | - | False | Whether to add watermark |
| Random Seed | Int | -1~∞ | -1 | -1 for random |

---

## 📋 Supported Models

### Video Generation - Seedance Series

#### Seedance 2.0 (Recommended)
- `doubao-seedance-2-0-260128` - Latest version, supports multimodal reference

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

This repository includes example workflow files:

- `workflows/火山API测试.json` - Basic test workflow (Text-to-Video + Image-to-Video)

Import method: ComfyUI interface → Load → Select json file

---

## 🔄 Output Format

### Video Nodes

| Port | Type | Description |
|-----|------|------|
| Video Frames | IMAGE | Video frame sequence, can connect to VHS nodes for preview/saving |
| Audio | AUDIO | Video audio, can connect to audio output nodes |
| Video Info | STRING | JSON format video metadata (fps, frame count, duration, etc.) |

### Image Node

| Port | Type | Description |
|-----|------|------|
| Images | IMAGE | Generated image sequence |
| Generation Info | STRING | JSON format metadata (size, seed, etc.) |

---

## ⚠️ Notes

1. **API Key Security**: Do not leak your API Key when sharing workflows
2. **Image Limits**: Image-to-Video supports up to 9 images, Image Generation supports up to 10 reference images
3. **Resolution Limits**: Seedance 2.0 supports up to 720p, older versions up to 1080p; Seedream supports up to 4K
4. **Audio Reference**: Must provide at least 1 image to use reference audio feature
5. **Service Tier**: `flex` mode is offline inference, ~50% cheaper but may have longer wait times
6. **Concurrent Count**: When concurrent count > 1, multiple videos will be generated simultaneously, be aware of API call costs

---

## 📄 License

MIT License

---

## 🔗 Related Links

- [VolcEngine Documentation](https://www.volcengine.com/docs/82379/1520757)
- [ComfyUI Official Repository](https://github.com/comfyanonymous/ComfyUI)
