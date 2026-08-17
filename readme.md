# Bhashini Models — Unified Inference Stack

This repository provides a unified local inference setup for:

* **ASR** — Automatic Speech Recognition
* **NMT** — Neural Machine Translation
* **TTS** — Text-to-Speech (Flite-based)

It includes automated environment setup, model loading, and individual inference execution.

---

#  Installation & Setup

Follow the steps below to set up the environment and dependencies.

---

## 1️⃣ Run Installer Script

This script will:

* Create Python virtual environment (3.10)
* Build & install Flite
* Download default + Indic voices

```bash
bash install.sh
```

---

## 2️⃣ Activate Virtual Environment

```bash
source venv/bin/activate
```

## 3️⃣ Install Python Dependencies

Install required libraries manually:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

* torch
* onnxruntime
* librosa
* numpy
* ctranslate2
* sentencepiece

---

## 4️⃣ Run Inference

Execute the main runner:

```bash
python main.py
```

---

# Inference Usage

`main.py` allows running:

* ASR
* NMT
* TTS
* Speech-to-Speech pipeline

Uncomment the required block inside `main.py`.

---

## ASR Example

```python
run_asr("samples/sample_hi.wav", "hi")
```

Output:

```
ASR Output: नमस्ते आपका स्वागत है
```

---

## NMT Example

```python
run_nmt(
    text="भारत एक महान देश है",
    src_lang="hi",
    tgt_lang="en"
)
```

Output:

```
India is a great country
```

---

## TTS Example

```python
run_tts(
    text="Hello, welcome to Bhashini.",
    language="en"
)
```

Audio file saved as:

```
tts_out.wav
```

---

#  Supported Languages

## ASR

* Hindi (hi)
* Tamil (ta)

## NMT

* English ↔ Indic
* Indic ↔ Indic

## TTS

* Hindi
* Tamil
* Telugu
* Marathi
* Gujarati
* Bengali
* Punjabi
* Kannada
* English

(Voice auto-selected per language)

---