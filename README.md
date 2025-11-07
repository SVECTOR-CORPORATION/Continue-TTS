<p align="center">
  <img alt="Continue-TTS" src="https://github.com/SVECTOR-CORPORATION/Continue-TTS/blob/main/continue-tts-image-banner.jpg?raw=true" width="800">
</p>

# Continue-TTS

### Text-to-Speech Model Based on Continue-1-OSS

<div align="left" style="line-height: 1;">
  <a href="https://spec-chat.tech" target="_blank" style="margin: 2px;">
    <img alt="SVECTOR" src="https://img.shields.io/badge/💬%20Spec%20Chat-Spec%20Chat-blue?style=plastic" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
  <a href="https://huggingface.co/SVECTOR-CORPORATION" target="_blank" style="margin: 2px;">
    <img alt="SVECTOR" src="https://img.shields.io/badge/🤗%20Hugging%20Face-SVECTOR-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
  <a href="https://huggingface.co/SVECTOR-CORPORATION/Continue-TTS/blob/main/LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue?color=1e88e5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  
  <a href="https://github.com/SVECTOR-CORPORATION/Continue-TTS" target="_blank" style="margin: 2px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Continue--TTS-181717?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## Introduction

We are thrilled to introduce **Continue-TTS**, a fine-tuned text-to-speech model based on the **Continue-1-OSS** architecture, developed by SVECTOR. This model is specifically trained for high-quality speech synthesis and delivers exceptional voice generation capabilities.

**Continue-TTS** is engineered to provide:

- **Natural Speech:** Human-like intonation, emotion, and rhythm that rivals commercial solutions
- **8 Unique Voices:** Diverse voice options with distinct personalities and characteristics
- **Real-time Generation:** Low-latency streaming for interactive applications (~200ms)
- **Emotional Expression:** Built-in support for laughter, sighs, gasps, and other natural emotions
- **Open Source:** Fully accessible under Apache 2.0 license for research and commercial use

This model is based on the **Continue-1-OSS** architecture and combines the power of large language models with neural audio codecs to generate exceptionally natural speech from text.

<audio controls src="https://ik.imagekit.io/svector/efd3e807-49a4-463b-af6d-4069acf7ff3a.wav"></audio>

```
The sun was setting behind the mountains, painting the sky with soft shades of orange and violet.
She stood there quietly, breathing in the moment. <sigh>
Sometimes, the smallest moments are the ones that change everything.
```

<audio controls src="https://ik.imagekit.io/svector/c99ff697-291a-4fb7-940a-56b523b9f286.wav?updatedAt=1762362454065"></audio>

```
<sigh>  
Not every journey is loud.  
Some begin quietly… inside.  
But once they begin, they never stop.  
We continue.
```

### Model Specifications

- **Base Architecture:** Continue-1-OSS
- **Type:** Text-to-Speech (TTS) Model
- **Parameters:** 3 Billion
- **Audio Codec:** SNAC (24kHz)
- **Context Length:** 131,072 tokens
- **Vocabulary:** 156,940 tokens (including 28,672 audio tokens)
- **License:** Apache 2.0
- **Voices:** 8 (Nova, Aurora, Stellar, Atlas, Orion, Luna, Phoenix, Ember)

## Requirements

To use Continue-TTS, install the required dependencies:

```bash
pip install transformers torch
pip install snac  # Audio codec
pip install vllm==0.7.3  # For fast inference (optional but recommended)
```

## Quickstart

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "SVECTOR-CORPORATION/Continue-TTS"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Prepare text with voice
text = "Hello! I am Continue-TTS, a text-to-speech model based on Continue-1-OSS."
voice = "nova"  # Choose: nova, aurora, stellar, atlas, orion, luna, phoenix, ember

# Format prompt (TTS format)
adapted_prompt = f"{voice}: {text}"
prompt_tokens = tokenizer(adapted_prompt, return_tensors="pt")
start_token = torch.tensor([[128259]], dtype=torch.int64)
end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)

# Generate audio tokens
outputs = model.generate(
    input_ids.to(model.device),
    max_new_tokens=1200,
    temperature=0.6,
    top_p=0.8,
    repetition_penalty=1.3,
    eos_token_id=49158,  # TTS stop token
    do_sample=True
)

# Decode tokens (audio codes can be decoded using SNAC decoder)
generated_tokens = tokenizer.decode(outputs[0], skip_special_tokens=False)
```

### Using Continue-TTS Package (Recommended)

For easier usage with audio generation, use the Continue-TTS package:

```bash
pip install continue-speech
```

```python
from continue_tts import Continue1Model
import wave

# Initialize model
model = Continue1Model(model_name="SVECTOR-CORPORATION/Continue-TTS", max_model_len=2048)

# Generate speech
text = "Welcome to Continue-TTS! This model is built on Continue-1-OSS."
audio_chunks = model.generate_speech(prompt=text, voice="nova")

# Save to file
with wave.open("output.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    for chunk in audio_chunks:
        wf.writeframes(chunk)
```

## Available Voices

Continue-TTS includes 8 professionally designed voices:

| Voice | Gender | Description |
|-------|--------|-------------|
| **nova** | Female | Conversational and natural, perfect for general use |
| **aurora** | Female | Warm and friendly, excellent for storytelling |
| **stellar** | Female | Energetic and bright, great for upbeat content |
| **atlas** | Male | Deep and authoritative, ideal for narration |
| **orion** | Male | Friendly and casual, perfect for conversational content |
| **luna** | Female | Soft and gentle, excellent for calm narration |
| **phoenix** | Male | Dynamic and expressive, great for engaging content |
| **ember** | Female | Warm and engaging, perfect for emotional expression |

## Advanced Features

### Emotion Tags

Add natural emotions to your speech:

```python
text = "This is incredible! <laugh> I can't believe how natural it sounds. <gasp>"
```

**Supported emotions:**
- `<laugh>` - Natural laughter
- `<chuckle>` - Light laugh
- `<sigh>` - Expressive sigh
- `<gasp>` - Surprised gasp
- `<cough>` - Cough sound
- `<yawn>` - Yawn
- `<groan>` - Groan
- `<sniffle>` - Sniffle

### Custom Generation Parameters

Fine-tune generation quality:

```python
audio = model.generate_speech(
    prompt="Your text here",
    voice="nova",
    temperature=0.6,        # Lower = more consistent, Higher = more varied
    top_p=0.8,             # Nucleus sampling threshold
    max_tokens=1200,       # Maximum audio length
    repetition_penalty=1.3 # Prevent token repetition
)
```

## Use Cases

Continue-TTS excels at:

- **Audiobook Narration:** Natural storytelling with emotional expression
- **Virtual Assistants:** Conversational AI with personality
- **Accessibility:** Text-to-speech for visually impaired users
- **Content Creation:** Voiceovers for videos, podcasts, and presentations
- **Gaming:** Dynamic character voices and dialogue
- **Education:** Interactive learning materials with voice
- **Customer Service:** Natural-sounding automated responses

## Performance

- **Quality:** State-of-the-art natural speech synthesis
- **Latency:** ~200ms for streaming generation (GPU)
- **Speed:** Real-time on GPU, slower on CPU
- **Memory:** ~7GB GPU RAM (FP16), ~14GB (FP32)
- **Sample Rate:** 24kHz (high quality audio)

## Model Architecture

Continue-TTS is built on the Continue-1-OSS and combines:
- **Base Model:** Continue-1-OSS (LLaMA-based, 3.3B parameters)
- **Audio Codec:** SNAC multi-scale neural audio codec
- **Token Structure:** 7 audio tokens per frame (hierarchical encoding)
- **Training:** Fine-tuned on few hours of diverse speech data

The model generates audio tokens autoregressively, which are then decoded into waveforms using the SNAC neural codec.

## Training

Continue-TTS was fine-tuned on the Continue-1-OSS using:
- High-quality speech datasets covering diverse accents and styles
- Multi-speaker recordings for voice diversity
- Emotional speech data for expressive synthesis
- Conversational and narrative content

Training utilized:
- Continue-1-OSS as base
- Custom tokenizer with 28,672 audio tokens
- Multi-stage training (pretraining + fine-tuning)
- Optimized for naturalness and emotion

## Limitations

As with any TTS model, Continue-TTS has certain limitations:

- **Pronunciation:** May struggle with unusual names, technical terms, or non-English words
- **Consistency:** Long-form generation may have minor quality variations
- **Accents:** Primarily trained on specific accent patterns
- **Compute:** Requires GPU for real-time generation (CPU is slower)
- **Language:** Currently optimized for English

## Ethical Considerations

SVECTOR is committed to responsible AI development. Users should:

- **Transparency:** Disclose when audio is AI-generated
- **Consent:** Do not clone voices without explicit permission
- **Verification:** Implement safeguards against deepfakes and misinformation
- **Attribution:** Credit the model when used in public projects
- **Responsible Use:** Avoid generating harmful, deceptive, or illegal content

## License

This model is released under the **Apache License 2.0**. See the [LICENSE](https://huggingface.co/SVECTOR-CORPORATION/Continue-TTS/blob/main/LICENSE) file for complete details.

## Acknowledgments

Continue-1-OSS builds upon advances in neural speech synthesis, large language models, and neural audio codecs. We thank the open-source community for their contributions to these foundational technologies.

---

<p align="center">
    <i>Developed by <a href="https://www.svector.co.in">SVECTOR</a></i>
</p>
