## 构建并分享你的模型

### 构建你的第一个应用

- 文本生成，通过 pipline 拉取模型到本地缓存

```Python
# 使用 transformers 库创建文本生成管道
from transformers import pipeline

# 创建默认的文本生成模型管道
model = pipeline("text-generation")


def predict(prompt):
    # 调用模型生成文本，返回生成的完整文本
    completion = model(prompt)[0]["generated_text"]
    return completion


# 测试预测功能
print(predict("My favorite programming language is"))

# 使用 Gradio 创建 Web 界面
import gradio as gr

# 创建并启动交互界面：输入为文本，输出为文本
gr.Interface(fn=predict, inputs="text", outputs="text").launch()

```

---

## 了解接口类

- 音频反转，演示接收不同类型的输入和返回不同类型的输出

```Python
import numpy as np
import gradio as gr


def reverse_audio(audio):
    """将音频数据反转播放"""
    sr, data = audio  # sr: 采样率，data: 音频数据数组
    reversed_audio = (sr, np.flipud(data))  # 翻转数组顺序，实现音频倒放
    return reversed_audio


# 麦克风输入组件，返回 numpy 格式的音频数据
mic = gr.Audio(sources="microphone", type="numpy", label="Speak here...")
# 创建接口：输入麦克风录音，输出反转后的音频
gr.Interface(reverse_audio, mic, "audio").launch()
```

- 音频生成，演示多输入

```Python
import numpy as np
import gradio as gr

# 12个音符名称，用于下拉选择
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def generate_tone(note, octave, duration):
    """生成指定音符、八度和持续时间的音频信号"""
    sr = 48000  # 采样率：每秒48000个样本点
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)  # A4频率440Hz，计算与A4的距离
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)  # 根据十二平均律计算目标频率
    duration = int(duration)  # 持续时间（秒）
    audio = np.linspace(0, duration, duration * sr)  # 生成时间轴数组
    audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)  # 生成正弦波，振幅20000
    return (sr, audio)  # 返回采样率和音频数据


gr.Interface(
    generate_tone,
    [
        gr.Dropdown(notes, type="index"),  # 音符下拉框，返回索引值（0-11）
        gr.Slider(minimum=4, maximum=6, step=1),  # 八度滑块（4-6）
        gr.Number(value=1, label="Duration in seconds"),  # 持续时间输入框
    ],
    "audio",  # 输出类型：音频播放器
).launch()
```

- 音频识别

```Python
from transformers import pipeline
import gradio as gr

# 加载预训练的自动语音识别模型（默认使用 Whisper）
model = pipeline("automatic-speech-recognition")


def transcribe_audio(mic=None, file=None):
    """将音频转录为文本"""
    # 优先使用麦克风录音，否则使用上传的文件
    if mic is not None:
        audio = mic
    elif file is not None:
        audio = file
    else:
        return "You must either provide a mic recording or a file"
    # 使用模型进行语音识别，返回转录文本
    transcription = model(audio)["text"]
    return transcription


gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(sources="microphone", type="filepath"),  # 麦克风录音输入
        gr.Audio(sources="upload", type="filepath"),  # 上传音频文件输入
    ],
    outputs="text",  # 输出：文本框显示转录结果
).launch()
```

---

## 与他人分享演示

- Gradio 演示可以通过两种方式进行分享：使用 临时的共享链接 或者 在 Spaces 上永久托管。

---

## 与 huggingface hub整合

```Python
import gradio as gr
from transformers import pipeline

title = "GPT-2 Text Generation"
description = "Gradio Demo for GPT-2 text generation. Enter your text and the model will continue it. GPT-J-6B is no longer available via Hugging Face Inference API, so we use GPT-2 as an alternative."
article = "<p style='text-align: center'><a href='https://github.com/kingoflolz/mesh-transformer-jax' target='_blank'>Original GPT-J-6B Reference</a></p>"

# 加载本地 GPT-2 模型（替代不再可用的 GPT-J-6B）
generator = pipeline("text-generation", model="gpt2")


def generate_text(input_text):
    """使用 GPT-2 模型生成文本"""
    result = generator(input_text, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]


gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Textbox(lines=10, label="Generated Text"),
    title=title,
    description=description,
    article=article,
).launch()
```

---

## 高级界面功能

- 简单聊天框功能

```Python
import random

import gradio as gr


def chat(message, history):
    """简单聊天函数，根据问题类型返回随机响应"""
    history = history or []
    if message.startswith("How many"):
        response = str(random.randint(1, 10))
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    # 新版 Gradio chatbot 要求消息格式为字典列表
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, history


iface = gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
)
iface.launch()
```

- 图像识别 模型解释

```Python
import requests
import tensorflow as tf

import gradio as gr

inception_net = tf.keras.applications.MobileNetV2()  # 加载模型

# 下载 ImageNet 的可读标签
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")


def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()
    return {labels[i]: float(prediction[i]) for i in range(1000)}


image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)

title = "Gradio Image Classifiction + Interpretation Example"
gr.Interface(
    fn=classify_image, inputs=image, outputs=label, interpretation="default", title=title
).launch()

```

---

## gradio blocks 简介

---