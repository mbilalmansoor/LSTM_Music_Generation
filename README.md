# 🎛️ AI Music Orchestrator 

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/bilalmansoor/LSTM_Music_Generation)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-FF5000?style=flat)](https://gradio.app/)

An interactive web application that leverages a stateful Deep Learning LSTM neural network to generate, compile, and orchestrate unique musical compositions simultaneously. 

🔗 **Live Demo:** [Deploy on Hugging Face Spaces](https://huggingface.co/spaces/bilalmansoor/LSTM_Music_Generation)

---

## 🎓 Course Project Context

This project was developed as an official academic assignment for the PGD **Deep Learning Course** at **NED University of Engineering and Technology**.

* **Supervised By:** [Sir Sajid Majeed](https://github.com/SajidMajeed92)
* **Project Team:**
  * [Bilal Mansoor](https://github.com/mbilalmansoor)
  * [Aqsa Mansoor](https://github.com/AqsaMansoor32)

---

## 🚀 Features

* **Multi-Track Orchestration:** Generate up to 10 unique tracks in a single batch loop utilizing individualized structural seeds.
* **Dynamic Variable Control:** Modify target track lengths (5s to 180s) and scale predictive text tokens on the fly.
* **Creativity (Temperature) Tuning:** Scalable entropy parameter (set to `2.0` by default) to flatten distribution matrices for experimental melodies.
* **Real-time Synthesis Sandbox:** Seamless internal compilation pipeline converting raw AI text tokens into physical MIDI arrays and synthesized audio streams.

---

## 🛠️ Tech Stack & Utilities

The platform combines robust deep learning frameworks with low-level Linux audio rendering engines:

* **Core AI Engine:** ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) (Keras Sequential API, Stateful LSTM Layers)
* **Numerical Processing:** ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) / ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
* **Interactive Interface:** ![Gradio](https://img.shields.io/badge/Gradio-FF5000?style=flat) UI framework with yielding block controls.
* **MIDI Compiler:** `abc2midi` (Translates structural ABC notation strings to binary MIDI format).
* **Audio Synthesizer:** `TiMidity++` (Renders MIDI streams into high-fidelity `.wav` soundboards).

---

## 🧠 Technical Approach

The core generative architecture functions through a character-based predictive pipeline mapped across individual musical sequences:

### 1. Vectorization Matrix
The training data uses standard **ABC Notation**, representing pitches, dynamic durations, bars, and metadata headers as unique text sequences. The input characters are vectorized using a hardcoded token index lookup:
$$\text{Character Index Mapping} \xrightarrow{} \mathbf{X} \in \mathbb{R}^{B \times T}$$

### 2. Stateful LSTM Recurrent Layers
Unlike traditional stateless structures, the model sets `stateful=True`. Hidden cell memory matrices carry over context from sequence to sequence. States are flushed layer-by-layer prior to initiating a new generation tracking loop to avoid cross-track structural interference.

### 3. Temperature Scaling Math
To govern randomness, predictions are scaled using a Temperature parameter ($T$) before categorical sample distribution:
$$\mathbf{P}_{\text{adjusted}} = \frac{\mathbf{Predictions}}{T}$$
Setting $T = 2.0$ flattens the probability distribution. This lowers the weight of predictable characters, promoting wider variation and highly creative musical leaps.

### 4. Sandbox Synthesis Loop
* The text tokens are filtered via RegEx boundaries into structural code headers.
* The system initializes temporary background file streams to run a sandboxed subprocess sequence: `ABC -> MIDI -> WAV`.
* Audio waveforms are dynamic-buffered (tiled or trimmed) to perfectly match user-specified durations.

---

## 💾 Local Environment Setup

### Prerequisites
Ensure your underlying system has the necessary Linux audio conversion utilities installed:
```bash
sudo apt-get update
sudo apt-get install abc2midi timidity
