# AI Guide

A beginner-friendly guide to the core Artificial Intelligence concepts used
today — from classic machine learning to modern generative AI, LLMs, and
AI agents.

## Table of Contents

1. [Foundations](#1-foundations)
2. [Machine Learning](#2-machine-learning)
3. [Deep Learning](#3-deep-learning)
4. [Natural Language Processing (NLP)](#4-natural-language-processing-nlp)
5. [Large Language Models (LLMs)](#5-large-language-models-llms)
6. [Generative AI](#6-generative-ai)
7. [Computer Vision](#7-computer-vision)
8. [Retrieval-Augmented Generation (RAG)](#8-retrieval-augmented-generation-rag)
9. [Prompt Engineering](#9-prompt-engineering)
10. [Fine-Tuning & Model Adaptation](#10-fine-tuning--model-adaptation)
11. [AI Agents](#11-ai-agents)
12. [Model Context Protocol (MCP) & Tool Use](#12-model-context-protocol-mcp--tool-use)
13. [Evaluation & Benchmarking](#13-evaluation--benchmarking)
14. [Responsible AI & Safety](#14-responsible-ai--safety)
15. [MLOps & Deployment](#15-mlops--deployment)
16. [Glossary of Common Terms](#16-glossary-of-common-terms)
17. [Further Learning](#17-further-learning)

---

## 1. Foundations

**Artificial Intelligence (AI)** is the broad field of building machines
that perform tasks normally requiring human intelligence — reasoning,
learning, perception, and language.

- **AI** → the overall field.
- **Machine Learning (ML)** → a subset of AI where systems learn patterns
  from data instead of following hard-coded rules.
- **Deep Learning (DL)** → a subset of ML using multi-layered neural
  networks to learn complex patterns.
- **Generative AI (GenAI)** → a subset of DL focused on generating new
  content (text, images, audio, code) rather than just making predictions.

```
AI  ⊃  Machine Learning  ⊃  Deep Learning  ⊃  Generative AI
```

---

## 2. Machine Learning

Core paradigms used to train models:

| Type | Description | Example |
|---|---|---|
| **Supervised Learning** | Learns from labeled input-output pairs | Spam detection, price prediction |
| **Unsupervised Learning** | Finds patterns in unlabeled data | Customer segmentation, clustering |
| **Semi-Supervised Learning** | Mix of small labeled + large unlabeled data | Medical image tagging |
| **Reinforcement Learning (RL)** | Learns by trial and error via rewards/penalties | Game-playing agents, robotics |
| **Self-Supervised Learning** | Generates its own labels from raw data | Pretraining LLMs on raw text |

**Key concepts:**
- **Features** — input variables used to make predictions.
- **Labels** — the correct output a model is trained to predict.
- **Training / Validation / Test sets** — data splits used to train a
  model, tune it, and evaluate its final performance.
- **Overfitting** — model memorizes training data but fails to generalize.
- **Underfitting** — model is too simple to capture the underlying pattern.
- **Bias-Variance Tradeoff** — balancing a model's simplicity (bias)
  against its sensitivity to training data noise (variance).
- **Gradient Descent** — optimization algorithm that adjusts model
  parameters to minimize error (loss).
- **Loss Function** — measures how far predictions are from the truth.

---

## 3. Deep Learning

Deep learning uses **neural networks** — layers of interconnected nodes
("neurons") that transform input data through weighted connections.

- **Neural Network** — layers of neurons (input → hidden → output) that
  learn representations of data.
- **Backpropagation** — the algorithm used to compute how much each
  weight contributed to the error, so it can be updated.
- **Activation Function** — introduces non-linearity (e.g., ReLU,
  Sigmoid, Softmax) so networks can learn complex patterns.
- **CNN (Convolutional Neural Network)** — specializes in grid-like data
  such as images.
- **RNN / LSTM (Recurrent Networks)** — historically used for sequential
  data like text and time series (largely replaced by Transformers).
- **Transformer** — the architecture (2017, "Attention Is All You Need")
  behind virtually all modern LLMs; processes sequences in parallel using
  **attention** instead of recurrence.
- **Attention Mechanism** — lets a model weigh the importance of
  different parts of the input when producing an output.
- **Embeddings** — dense numeric vectors that represent words, sentences,
  images, or other data in a way that captures meaning/similarity.

---

## 4. Natural Language Processing (NLP)

NLP is the field concerned with enabling computers to understand and
generate human language.

- **Tokenization** — splitting text into smaller units (tokens: words,
  subwords, or characters) that a model can process.
- **Word Embeddings** — vector representations of words (e.g., Word2Vec,
  GloVe) capturing semantic meaning.
- **Named Entity Recognition (NER)** — identifying names, places,
  organizations, etc. in text.
- **Sentiment Analysis** — determining the emotional tone of text.
- **Machine Translation** — translating text between languages.
- **Text Summarization** — condensing text while preserving key meaning.

---

## 5. Large Language Models (LLMs)

LLMs (e.g., Claude, GPT, Gemini, Llama) are large Transformer-based
models trained on massive text (and increasingly multimodal) datasets.

- **Pretraining** — training a model on huge amounts of raw data (often
  self-supervised, e.g., "predict the next token") to learn general
  language patterns.
- **Context Window** — the maximum amount of text (measured in tokens) a
  model can consider at once.
- **Tokens** — the basic units LLMs process; pricing and limits are
  usually measured in tokens, not words or characters.
- **Temperature** — a sampling parameter controlling randomness/creativity
  of outputs (low = more deterministic, high = more varied).
- **Hallucination** — when a model generates plausible-sounding but false
  or fabricated information.
- **Multimodal Models** — models that can process/generate more than one
  data type (text, images, audio, video) — e.g., Claude's vision input.
- **Chain-of-Thought (CoT)** — prompting/reasoning technique where a
  model works through intermediate steps before answering, improving
  accuracy on complex tasks.
- **Extended/Reasoning Models** — models with a dedicated "thinking"
  phase before producing a final answer (e.g., Claude's extended
  thinking mode).

---

## 6. Generative AI

GenAI models create new content rather than just classifying or
predicting existing data.

- **Text Generation** — LLMs producing prose, code, summaries, etc.
- **Image Generation** — diffusion models (e.g., Stable Diffusion,
  Midjourney, DALL·E) that generate images from text prompts.
- **Diffusion Models** — generate content by gradually removing noise
  from random data, guided by a prompt.
- **GANs (Generative Adversarial Networks)** — two networks (generator
  vs. discriminator) compete, improving generated output quality.
- **Audio/Voice Generation** — text-to-speech and voice cloning models.
- **Code Generation** — models trained to write, explain, and debug code
  (e.g., Claude Code, GitHub Copilot).

---

## 7. Computer Vision

- **Image Classification** — assigning a label to an image.
- **Object Detection** — locating and classifying multiple objects in
  an image.
- **Image Segmentation** — labeling each pixel of an image by object.
- **OCR (Optical Character Recognition)** — extracting text from images.
- **Vision Transformers (ViT)** — applying transformer architecture to
  image patches instead of convolutions.

---

## 8. Retrieval-Augmented Generation (RAG)

RAG combines an LLM with an external knowledge source so it can answer
using up-to-date or private information instead of relying only on what
it learned during training.

**How it works:**
1. Convert documents into **embeddings** and store them in a
   **vector database** (e.g., Pinecone, Weaviate, Chroma, pgvector).
2. On a query, retrieve the most relevant chunks (**semantic search**).
3. Feed the retrieved context + the user's question into the LLM to
   generate a grounded answer.

- **Vector Database** — stores embeddings and enables fast similarity
  search.
- **Chunking** — splitting documents into smaller pieces for embedding
  and retrieval.
- **Semantic Search** — finding results by meaning/similarity rather
  than exact keyword match.
- **Grounding** — anchoring model outputs to retrieved/verifiable data
  to reduce hallucination.

---

## 9. Prompt Engineering

The practice of crafting inputs to get better, more reliable outputs
from an LLM.

- **Zero-shot prompting** — asking the model to perform a task with no
  examples.
- **Few-shot prompting** — providing a few examples in the prompt to
  guide the model's output format/behavior.
- **System Prompt** — instructions that set the model's role, tone, and
  constraints for a conversation.
- **Chain-of-Thought prompting** — asking the model to "think step by
  step" to improve reasoning accuracy.
- **Prompt Injection** — a security risk where malicious text embedded
  in input tries to override the model's instructions.

---

## 10. Fine-Tuning & Model Adaptation

Ways to specialize a general-purpose model for a specific task or domain:

- **Fine-Tuning** — further training a pretrained model on a smaller,
  task-specific dataset.
- **Instruction Tuning** — fine-tuning a model to better follow
  natural-language instructions.
- **RLHF (Reinforcement Learning from Human Feedback)** — using human
  preference data to align model behavior with what people find helpful
  and safe.
- **LoRA / PEFT (Parameter-Efficient Fine-Tuning)** — techniques that
  fine-tune a small number of additional parameters instead of the
  entire model, saving compute.
- **Distillation** — training a smaller "student" model to mimic a
  larger "teacher" model's outputs.

---

## 11. AI Agents

Agents are AI systems that can plan, use tools, and take multi-step
actions autonomously to achieve a goal — rather than just responding to
a single prompt.

- **Agentic Loop** — the cycle of an agent reasoning, choosing a tool,
  observing the result, and deciding the next step, repeated until the
  task is done.
- **Tool Use / Function Calling** — an LLM's ability to call external
  functions/APIs (e.g., search the web, run code, query a database).
- **Planning** — breaking a complex goal into smaller, ordered steps.
- **Memory** — short-term (conversation context) vs. long-term
  (persisted knowledge across sessions) state an agent can draw on.
- **Multi-Agent Systems** — multiple specialized agents collaborating
  (e.g., a "researcher" agent and a "writer" agent working together).
- **Orchestration** — coordinating multiple tools/agents/steps to
  complete a larger workflow.

---

## 12. Model Context Protocol (MCP) & Tool Use

**MCP** is an open standard for connecting AI models to external tools,
data sources, and services in a consistent way — so an assistant can,
for example, read files, query a database, or call a GitHub API without
each integration being custom-built.

- **MCP Server** — exposes tools/data/resources that an AI client can use.
- **MCP Client** — the AI application (e.g., Claude Code) that connects
  to MCP servers to extend its capabilities.
- **Tool Schema** — a structured definition (name, description,
  parameters) describing what a tool does so a model can call it
  correctly.

---

## 13. Evaluation & Benchmarking

- **Benchmark** — a standardized test set used to compare model
  performance (e.g., MMLU, HumanEval, GSM8K).
- **Accuracy / Precision / Recall / F1 Score** — common metrics for
  classification performance.
- **Perplexity** — measures how well a language model predicts text
  (lower is better).
- **Human Evaluation** — using human judges to rate output quality,
  helpfulness, or safety.
- **LLM-as-a-Judge** — using one LLM to evaluate the outputs of another.

---

## 14. Responsible AI & Safety

- **Alignment** — ensuring an AI system's goals and behavior match human
  values and intentions.
- **Bias** — systematic unfairness in model outputs, often inherited
  from training data.
- **Explainability / Interpretability** — understanding why a model
  produced a given output.
- **Guardrails** — rules, filters, or systems that constrain model
  behavior to prevent harmful or unwanted outputs.
- **Red Teaming** — deliberately probing a model for weaknesses,
  vulnerabilities, or harmful behaviors before deployment.
- **Data Privacy** — protecting personal or sensitive data used in
  training or inference.

---

## 15. MLOps & Deployment

- **Inference** — running a trained model to make predictions/generate
  outputs (as opposed to training it).
- **Latency** — the time it takes a model to respond.
- **Throughput** — how many requests a system can handle per unit time.
- **Quantization** — reducing a model's numerical precision (e.g.,
  32-bit → 8-bit) to shrink size and speed up inference with minimal
  accuracy loss.
- **Model Serving** — infrastructure that hosts a model and exposes it
  via an API.
- **Caching (e.g., prompt caching)** — reusing previously computed
  results to reduce cost/latency for repeated context.

---

## 16. Glossary of Common Terms

| Term | Meaning |
|---|---|
| **API** | Interface that lets applications communicate with a model/service |
| **Corpus** | A large collection of text used for training/analysis |
| **Dataset** | A structured collection of data used to train or evaluate a model |
| **Epoch** | One full pass through the training dataset |
| **GPU/TPU** | Specialized hardware used to train and run AI models efficiently |
| **Inference Cost** | The compute/monetary cost of running a model to get a result |
| **Latent Space** | The compressed, abstract representation a model learns internally |
| **Parameters** | The learned weights of a model; roughly indicates model size/capacity |
| **Pipeline** | A sequence of connected data-processing/model steps |
| **Zero-day / Jailbreak** | Techniques used to bypass a model's safety restrictions |

---

## 17. Further Learning

- [Anthropic — Claude Documentation](https://docs.anthropic.com)
- [Google — Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Hugging Face — NLP Course](https://huggingface.co/course)
- [3Blue1Brown — Neural Networks (YouTube series)](https://www.youtube.com/c/3blue1brown)
- ["Attention Is All You Need" (Transformer paper, 2017)](https://arxiv.org/abs/1706.03762)

---

*This guide is a living document — contributions and corrections are
welcome via pull request.*
