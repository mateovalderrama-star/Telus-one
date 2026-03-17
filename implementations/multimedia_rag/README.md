# Multimedia RAG + VLM Reference Implementation

Multimedia Retrieval-Augmented Generation (RAG) pipeline for video understanding using:

* **ImageBind** for cross-modal embedding (audio, video, text)
* **PyTorchVideo** for video feature handling
* **Qwen Omni** stack for multimodal QA
* SONIC-O1 VQA benchmark

The implementation supports:

* 🔎 **Retrieval mode (RAG)** – segment-level audio/video embedding + top-k retrieval
* ❓ **Video QA mode** – multimodal inference over retrieved segments

Two isolated environments are required due to incompatible Torch stacks.

---

## Project Structure

```text
video_rag/
│
├── checkpoints/          # ImageBind weights
├── data/                 # VQA JSON + media files
├── .venv-rag/            # Retrieval environment
├── .venv-qa/             # QA environment
└── pyproject.toml
```

---

## 1. Download Model Checkpoint

```bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
cd ..
```

This is required for embedding video/audio segments.

---

## 2. Download Dataset

### Create data directory

```bash
mkdir -p data
```

### VQA JSON Files (Place in `data/`)

Download:

* [https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/02_Job_Interviews.json](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/02_Job_Interviews.json)
* [https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/04_Customer_Service_Interactions.json](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/04_Customer_Service_Interactions.json)
* [https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/01_Patient-Doctor_Consultations.json](https://huggingface.co/datasets/vector-institute/sonic-o1/blob/main/vqa/task2_mcq/01_Patient-Doctor_Consultations.json)

These define the multiple-choice video QA tasks.

### Video / Audio / Caption Files

Download from:

```
<GOOGLE_DRIVE_LINK_PLACEHOLDER>
```

Extract contents into:

```bash
data/
```

The expected structure includes:

* video
* audio
* caption

---

## 3. Environment Setup

Two independent environments are required due to package conflicts between the RAG (retrieval) and QA (inference) pipelines. Follow the instructions below to set up each environment.

---

```bash
# If ffmpeg is not installed, install it for video/audio processing
sudo apt update
sudo apt install ffmpeg
```

### A. Ref5 – Video RAG (Retrieval)

Used for:

* Segment embedding
* Cross-modal similarity search
* Top-k retrieval

From the **root of the repository**:

```bash
uv sync --group multimedia-rag
source .venv/bin/activate
```

---

### B. Ref5 – Video QA (Inference)

Used for:

* Qwen Omni multimodal reasoning
* Answer generation over retrieved segments

From the **root of the repository**:

```bash
uv sync --group multimedia-rag
source .venv/bin/activate
```

> **Note:** The two groups are mutually exclusive — switch between them by re-running `uv sync --group <name>` as needed.

---

## 4. Running Notebooks

```bash
jupyter lab
```

Choose:

* **Ref5 (Video RAG)** → retrieval pipeline
* **Ref5 (Video QA)** → multimodal QA evaluation

---

## Implementation Overview

## Retrieval Pipeline

1. Segment video/audio
2. Encode segments using ImageBind
3. Encode question text
4. Compute cosine similarity
5. Retrieve top-k segments
6. Save retrieval metadata

## QA Pipeline

1. Load retrieval outputs
2. Load associated media segments
3. Pass multimodal inputs to Qwen Omni
4. Generate MCQ answer
5. Evaluate accuracy

## References and Resources

1. **[MAGNET: A Multi-agent Framework for Finding Audio-Visual Needles by Reasoning over Multi-Video Haystacks](https://arxiv.org/abs/2506.07016)**
    A comprehensive framework for temporal, causal, and multi-hop retrieval across long video haystacks.

2. **[SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding](https://arxiv.org/abs/2601.21666)**
    A benchmark for evaluating multimodal LLMs on real-world audio-video understanding tasks.

3. **[VisRAG: Vision-based Retrieval-Augmented Generation on Multi-modality Documents](https://arxiv.org/abs/2410.10594)**
    A study on retrieval-augmented generation leveraging vision-based multi-modal documents.

4. **[Videos Dataset for LLMs RAG](https://huggingface.co/datasets/elmoghany/Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text)**
    A dataset designed for retrieval-augmented generation tasks requiring audio, video, and text.

5. **[SONIC-O1 Dataset](https://huggingface.co/datasets/vector-institute/sonic-o1)**
    A dataset for audio-visual question answering tasks across various domains.
