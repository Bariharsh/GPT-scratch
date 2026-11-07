# Character-Level GPT from Scratch (Transformer Decoder)

A from-scratch implementation of an autoregressive, character-level **Generative Pre-trained Transformer (GPT)** in PyTorch, inspired by Andrej Karpathy's *Zero to Hero* series.

This repository tracks the complete journey of building a decoder-only Transformer: starting from a simple bigram baseline, deriving the mathematical trick of causal self-attention through four iterative stages, assembling multi-head attention and feed-forward blocks with residual connections and layer normalization, and finally training a full GPT model to generate Shakespeare-like prose.

---

## Table of Contents

- [Overview & Architecture: Decoder-Only GPT (`v2.py`)](#overview--architecture-decoder-only-gpt-v2py)
  - [High-Level Architecture](#high-level-architecture)
  - [Inside a Transformer Block](#inside-a-transformer-block)
  - [Causal Multi-Head Self-Attention Detail](#causal-multi-head-self-attention-detail)
- [Original Transformer: "Attention Is All You Need" (Vaswani et al., 2017)](#original-transformer-attention-is-all-you-need-vaswani-et-al-2017)
  - [Encoder-Decoder Architecture Diagram](#encoder-decoder-architecture-diagram)
  - [How the Original Architecture Works](#how-the-original-architecture-works)
  - [Original Transformer vs. Decoder-Only GPT (This Project)](#original-transformer-vs-decoder-only-gpt-this-project)
  - [The Three Transformer Families: Encoder-Decoder, Encoder-Only, Decoder-Only](#the-three-transformer-families-encoder-decoder-encoder-only-decoder-only)
- [Tensor Dimensionality Walkthrough](#tensor-dimensionality-walkthrough)
- [The Step-by-Step Learning Journey](#the-step-by-step-learning-journey)
  - [1. Tokenization and Data Batching](#1-tokenization-and-data-batching)
  - [2. The Baseline: Bigram Language Model (`bigram.py`)](#2-the-baseline-bigram-language-model-bigrampy)
  - [3. The Mathematical Trick in Self-Attention (4-Stage Progression)](#3-the-mathematical-trick-in-self-attention-4-stage-progression)
  - [4. Why the Scaling Factor \(\sqrt{d_k}\) is Vital](#4-why-the-scaling-factor-sqrtd_k-is-vital)
  - [5. Multi-Head Attention](#5-multi-head-attention)
  - [6. Position-Wise Feed-Forward Network](#6-position-wise-feed-forward-network)
  - [7. Residual Connections and Pre-LayerNorm](#7-residual-connections-and-pre-layernorm)
  - [8. Assembling the Full Transformer Decoder (`v2.py`)](#8-assembling-the-full-transformer-decoder-v2py)
- [Mathematical Formulations](#mathematical-formulations)
- [Model Configurations and Hyperparameters](#model-configurations-and-hyperparameters)
- [Training Dynamics and Generation](#training-dynamics-and-generation)
  - [Loss Progression](#loss-progression)
  - [Qualitative Generation Comparison: Baseline vs. Transformer](#qualitative-generation-comparison-baseline-vs-transformer)
  - [Autoregressive Text Generation](#autoregressive-text-generation)
- [Project Files](#project-files)
- [Requirements and How to Run](#requirements-and-how-to-run)
- [Key Takeaways and Insights](#key-takeaways-and-insights)

---

## Overview & Architecture: Decoder-Only GPT (`v2.py`)

The model implemented in [`v2.py`](./v2.py) is a **Decoder-Only Transformer Language Model**. Given a sequence of characters, it predicts the probability distribution of the next character.

### High-Level Architecture

```mermaid
flowchart TD
    subgraph Inputs["1. Input & Embeddings"]
        InTokens["Input Token IDs: x (B, T)"]
        InPositions["Position IDs: 0, 1, ..., T-1 (T)"]
        
        TokEmbTable["Token Embedding Table (vocab_size, N_EMBED)"]
        PosEmbTable["Position Embedding Table (BLOCK_SIZE, N_EMBED)"]
        
        InTokens --> TokEmbTable
        InPositions --> PosEmbTable
        
        TokEmbTable -->|"tok_emb (B, T, N_EMBED)"| AddEmb["Element-wise Add (+)"]
        PosEmbTable -->|"pos_emb (T, N_EMBED)"| AddEmb
        AddEmb -->|"x (B, T, N_EMBED)"| BlocksInput["Transformer Input Stream"]
    end

    subgraph BlocksStack["2. Transformer Blocks (Repeated N_BLOCKS times)"]
        BlocksInput --> Block1["Transformer Block 1"]
        Block1 --> Block2["Transformer Block 2"]
        Block2 --> BlockDots["... (N_BLOCKS = 6)"]
        BlockDots --> BlockN["Transformer Block N"]
    end

    subgraph Head["3. Final Normalization & LM Head"]
        BlockN --> FinalLN["Final LayerNorm (N_EMBED)"]
        FinalLN --> LMHeadLinear["Linear LM Head (N_EMBED -> vocab_size)"]
        LMHeadLinear --> Logits["Logits: (B, T, vocab_size)"]
        
        Logits --> ModeSwitch{"Training or Inference?"}
        ModeSwitch -->|"Training"| Loss["Cross-Entropy Loss (against targets Y shifted by 1)"]
        ModeSwitch -->|"Inference"| Softmax["Softmax over Logits[:, -1, :]"]
        Softmax --> Sample["Multinomial Sampling -> Next Token ID"]
    end
```

---

### Inside a Transformer Block

Each block utilizes a **Pre-LayerNorm** formulation with residual (skip) connections around the attention and feed-forward sublayers.

```mermaid
flowchart TD
    subgraph TransformerBlock["Transformer Block (Pre-LN with Residuals)"]
        BlockIn["Block Input: x (B, T, C)"] --> Skip1["Residual Highway 1"]
        BlockIn --> LN1["LayerNorm 1"]
        LN1 --> MHA["Multi-Head Causal Self-Attention"]
        MHA --> Drop1["Dropout"]
        Skip1 --> Add1["Add (+)"]
        Drop1 --> Add1
        
        Add1 --> MidX["x = x + MHA(LN1(x))"]
        MidX --> Skip2["Residual Highway 2"]
        MidX --> LN2["LayerNorm 2"]
        LN2 --> FFN["Feed-Forward Network (Linear 4C -> ReLU -> Linear C)"]
        FFN --> Drop2["Dropout"]
        Skip2 --> Add2["Add (+)"]
        Drop2 --> Add2
        
        Add2 --> BlockOut["Block Output: x (B, T, C)"]
    end
```

---

### Causal Multi-Head Self-Attention Detail

Inside the multi-head self-attention layer, $h$ heads operate in parallel. Each head projects inputs into Queries, Keys, and Values, applies the causal lower-triangular mask, computes scaled dot-product attention, and outputs a context-aware representation.

```mermaid
flowchart TD
    subgraph MultiHead["Multi-Head Attention Mechanism"]
        XIn["Input: x (B, T, C)"] --> Head1["Head 1"]
        XIn --> Head2["Head 2"]
        XIn --> Head3["Head ..."]
        XIn --> HeadH["Head H (num_heads = 4)"]

        subgraph SingleHeadDetails["Inside Each Attention Head (head_size = d_k)"]
            HInput["Input: x (B, T, C)"]
            HInput --> Wq["Query Linear: (C, d_k)"]
            HInput --> Wk["Key Linear: (C, d_k)"]
            HInput --> Wv["Value Linear: (C, d_k)"]
            
            Wq --> Q["Q = x @ W_q (B, T, d_k)"]
            Wk --> K["K = x @ W_k (B, T, d_k)"]
            Wv --> V["V = x @ W_v (B, T, d_k)"]
            
            Q --> MatMulQK["MatMul: Q @ K^T -> (B, T, T)"]
            K --> MatMulQK
            
            MatMulQK --> Scale["Scale: divide by sqrt(d_k)"]
            Scale --> Mask["Causal Mask: tril == 0 -> -inf"]
            Mask --> Softmax["Softmax (dim = -1) -> Attention Weights"]
            
            Softmax --> MatMulV["MatMul: Weights @ V -> (B, T, d_k)"]
            V --> MatMulV
            MatMulV --> HeadOut["Head Output: (B, T, d_k)"]
        end

        Head1 --> Concat["Concatenate along embedding dim: (B, T, H * d_k)"]
        Head2 --> Concat
        Head3 --> Concat
        HeadH --> Concat

        Concat --> Proj["Output Projection Linear: (H * d_k, C)"]
        Proj --> Drop["Dropout"]
        Drop --> MHAOutput["MHA Result: (B, T, C)"]
    end
```

---

## Original Transformer: "Attention Is All You Need" (Vaswani et al., 2017)

The Transformer architecture was originally introduced by Vaswani et al. in the landmark paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). While modern GPT models (including this project) use a **decoder-only** architecture for generative text modeling, the original paper proposed an **Encoder-Decoder** architecture specifically engineered for sequence-to-sequence machine translation (e.g., translating English to German or French).

### Encoder-Decoder Architecture Diagram

```mermaid
flowchart TD
    subgraph EncoderSide["ENCODER (Nx = 6 Layers)"]
        InTokens["Inputs (Source Sequence)"] --> InEmbed["Input Embedding (d_model = 512)"]
        InPos["Positional Encoding (Sinusoidal)"] --> InAdd["Add (+)"]
        InEmbed --> InAdd
        InAdd --> EncLayerIn["Encoder Layer Input"]

        subgraph EncBlock["Inside Each Encoder Layer"]
            EncLayerIn --> EncMHA["Multi-Head Self-Attention (Bidirectional)"]
            EncLayerIn --> EncSkip1["Residual Highway 1"]
            EncMHA --> EncAddNorm1["Add & Norm (Post-LN)"]
            EncSkip1 --> EncAddNorm1
            
            EncAddNorm1 --> EncFFN["Feed-Forward Network (Linear 2048 -> ReLU -> Linear 512)"]
            EncAddNorm1 --> EncSkip2["Residual Highway 2"]
            EncFFN --> EncAddNorm2["Add & Norm (Post-LN)"]
            EncSkip2 --> EncAddNorm2
        end

        EncAddNorm2 --> EncOutput["Encoder Memory: Keys (K) & Values (V)"]
    end

    subgraph DecoderSide["DECODER (Nx = 6 Layers)"]
        OutTokens["Outputs: shifted right (Target Sequence)"] --> OutEmbed["Output Embedding (d_model = 512)"]
        OutPos["Positional Encoding (Sinusoidal)"] --> OutAdd["Add (+)"]
        OutEmbed --> OutAdd
        OutAdd --> DecLayerIn["Decoder Layer Input"]

        subgraph DecBlock["Inside Each Decoder Layer"]
            DecLayerIn --> DecMaskedMHA["Masked Multi-Head Self-Attention (Causal)"]
            DecLayerIn --> DecSkip1["Residual Highway 1"]
            DecMaskedMHA --> DecAddNorm1["Add & Norm (Post-LN)"]
            DecSkip1 --> DecAddNorm1

            DecAddNorm1 --> DecCrossMHA["Multi-Head Cross-Attention (Q from Decoder, K & V from Encoder)"]
            DecAddNorm1 --> DecSkip2["Residual Highway 2"]
            DecCrossMHA --> DecAddNorm2["Add & Norm (Post-LN)"]
            DecSkip2 --> DecAddNorm2

            DecAddNorm2 --> DecFFN["Feed-Forward Network (Linear 2048 -> ReLU -> Linear 512)"]
            DecAddNorm2 --> DecSkip3["Residual Highway 3"]
            DecFFN --> DecAddNorm3["Add & Norm (Post-LN)"]
            DecSkip3 --> DecAddNorm3
        end

        DecAddNorm3 --> DecOutput["Decoder Final Representations"]
    end

    EncOutput -->|"Keys (K) & Values (V)"| DecCrossMHA
    DecOutput --> FinalLinear["Linear Projection Layer (d_model -> Target Vocab)"]
    FinalLinear --> FinalSoftmax["Softmax"]
    FinalSoftmax --> OutProbs["Output Probabilities (Next Target Word)"]
```

### How the Original Architecture Works

1. **The Encoder (Bidirectional Context)**:
   - Takes the entire source sequence (e.g., German: *"Das Mädchen ging nach Hause"*) all at once.
   - Attention in the encoder is **unmasked and bidirectional**: every token can attend to all tokens to its left and right simultaneously.
   - It outputs contextual continuous representations of the source sentence containing rich syntactic and semantic information.

2. **The Decoder (Autoregressive Generation with Cross-Attention)**:
   - Generates the target sequence (e.g., English: *"The girl went home"*) one word at a time, conditioned on both the past generated words and the encoder output.
   - **Sublayer 1: Masked Self-Attention**: Prevents positions from attending to future tokens (strictly causal, preserving the autoregressive property).
   - **Sublayer 2: Multi-Head Cross-Attention (Encoder-Decoder Attention)**: The decoder uses its own intermediate representations as **Queries ($Q$)**, but retrieves **Keys ($K$)** and **Values ($V$)** from the final output of the Encoder. This allows the decoder to dynamically focus on relevant parts of the input source sentence when emitting each translated token.
   - **Sublayer 3: Position-Wise Feed-Forward Network**: An identical MLP applied independently to each position.

3. **Sinusoidal Positional Encoding**:
   Rather than learning a fixed lookup table of position embeddings, the original paper used deterministic sine and cosine wave functions of different frequencies:
   $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
   $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
   *Why sinusoidal?* The authors hypothesized this would allow the model to easily learn to attend by relative positions (since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$) and extrapolate to sequence lengths unseen during training.

---

### Original Transformer vs. Decoder-Only GPT (This Project)

The architectural shift from the original 2017 Transformer to modern GPT language models simplifies the design by discarding the encoder tower and training a single causal decoder on general next-token prediction:

| Architectural Feature | Original Paper (*Vaswani et al., 2017*) | Decoder-Only GPT (This Project / *Radford et al.*) | Why It Changed |
| :--- | :--- | :--- | :--- |
| **Primary Task** | Sequence-to-Sequence (Machine Translation) | Autoregressive Next-Token Language Modeling | Next-token prediction unifies all NLP tasks (translation, summarization, coding, dialogue) under one objective. |
| **Tower Structure** | Dual-Tower: Encoder + Decoder | Single-Tower: Decoder Only | A separate encoder is unnecessary when conditioning text (prompts) and generated text can be concatenated in one causal stream. |
| **Attention Sublayers** | 1. Bidirectional Self-Attention (Enc)<br>2. Causal Masked Self-Attention (Dec)<br>3. Cross-Attention (Dec attending to Enc) | Causal Masked Self-Attention Only | No encoder exists; every token attends exclusively to prior tokens in the unified sequence. |
| **Layer Normalization** | **Post-LN**: $x \leftarrow \text{LN}(x + \text{Sublayer}(x))$ | **Pre-LN**: $x \leftarrow x + \text{Sublayer}(\text{LN}(x))$ | Pre-LN preserves identity gradient highways through the skip connections, making deep models far more stable to train without warm-up heuristics. |
| **Positional Encoding** | Fixed Sinusoidal ($\sin / \cos$) functions | Learned Position Embeddings (`nn.Embedding`) | Learned embeddings adapt directly to the training distribution and are simpler to implement. |
| **Feed-Forward Activation** | $\text{ReLU}(x) = \max(0, x)$ | $\text{ReLU}$ (in [`v2.py`](./v2.py)) / $\text{GELU}$ (in GPT-2/3) | GELU provides smooth non-zero gradients for slightly negative inputs. |
| **Inference Input** | Source Sentence $\to$ Encoder; Start-token $\to$ Decoder | Prompt Prefix $\to$ Decoder | Seamless continuation of arbitrary text prompts without separate encoding steps. |

---

### The Three Transformer Families: Encoder-Decoder, Encoder-Only, Decoder-Only

The original Transformer spawned three distinct architectural paradigms in modern deep learning:

```mermaid
flowchart LR
    subgraph EncDec["1. Encoder-Decoder (Vaswani et al.)"]
        direction TB
        ED_Enc["Encoder (Bidirectional)\n[Reads full input]"] -->|"Keys & Values"| ED_Dec["Decoder (Causal + Cross-Attn)\n[Generates output]"]
        ED_Label["Examples: T5, BART\nBest for: Translation, Summarization"]
    end

    subgraph EncOnly["2. Encoder-Only (Devlin et al.)"]
        direction TB
        EO_Enc["Encoder Only (Bidirectional Attention)\n[All tokens attend to all tokens]"]
        EO_Label["Examples: BERT, RoBERTa\nBest for: Classification, NER, Search"]
    end

    subgraph DecOnly["3. Decoder-Only (Radford et al. / This Project)"]
        direction TB
        DO_Dec["Decoder Only (Causal Masked Attention)\n[Tokens only attend to past tokens]"]
        DO_Label["Examples: GPT-2/3/4, LLaMA, Mistral\nBest for: Text Generation, Reasoning, Chat"]
    end
```

---

## Tensor Dimensionality Walkthrough

Tracking tensor shapes through the model helps demystify how multi-head attention and batching operate. Below is the lifecycle of shapes in [`v2.py`](./v2.py) for batch size $B$, context window $T$, embedding width $C$, number of heads $H$, head size $d_k$, and vocabulary size $V$:

| Step / Layer | Input Tensor Shape | Output Tensor Shape | Operation Details |
| :--- | :---: | :---: | :--- |
| **Input Token Batch** | — | `(B, T)` | Batch of $B$ sequences, each of length $T$ integers in $[0, V-1]$ |
| **Token Embedding** | `(B, T)` | `(B, T, C)` | Look up embedding vector for each character (`N_EMBED = C = 384`) |
| **Position Embedding** | `(T,)` | `(T, C)` | Look up position vector for indices $0, 1, \dots, T-1$ |
| **Combined Input ($x$)** | `(B, T, C)` + `(T, C)` | `(B, T, C)` | Element-wise broadcasting addition: $x = \text{tok\_emb} + \text{pos\_emb}$ |
| **LayerNorm (Pre-LN)** | `(B, T, C)` | `(B, T, C)` | Normalize across feature dimension $C$ for each $(b, t)$ pair |
| **Q, K, V Projections** | `(B, T, C)` | `(B, T, d_k)` | Project each head to `HEAD_SIZE` ($d_k$): $Q = x W_q, K = x W_k, V = x W_v$ |
| **Raw Attention Affinities** | $Q, K$: `(B, T, d_k)` | `(B, T, T)` | Batched matrix multiplication: $Q K^T$, scaled by $1 / \sqrt{d_k}$ |
| **Causal Mask & Softmax** | `(B, T, T)` | `(B, T, T)` | Mask upper triangle with $-\infty$; softmax along last dimension |
| **Attention Output (per head)** | `Weights`: `(B, T, T)`, $V$: `(B, T, d_k)` | `(B, T, d_k)` | Batched matrix multiplication: $\text{Weights} \times V$ |
| **Concat Heads** | $H \times$ `(B, T, d_k)` | `(B, T, H * d_k)` | Concatenate all $H = 4$ attention heads along feature dimension |
| **Output Projection Linear** | `(B, T, H * d_k)` | `(B, T, C)` | Linear projection back to model width $C = 384$ + Dropout |
| **Residual Addition 1** | `(B, T, C)` + `(B, T, C)` | `(B, T, C)` | Skip highway: $x \leftarrow x + \text{MHA}(\text{LN}_1(x))$ |
| **FFN Linear 1 (Expansion)** | `(B, T, C)` | `(B, T, 4C)` | Expand dimensionality $4\times$ ($1536$) + apply ReLU activation |
| **FFN Linear 2 (Projection)** | `(B, T, 4C)` | `(B, T, C)` | Project back down to $C = 384$ + Dropout |
| **Residual Addition 2** | `(B, T, C)` + `(B, T, C)` | `(B, T, C)` | Skip highway: $x \leftarrow x + \text{FFN}(\text{LN}_2(x))$ |
| **Final LayerNorm** | `(B, T, C)` | `(B, T, C)` | Normalize representations emerging from final block $N$ |
| **Language Model Head** | `(B, T, C)` | `(B, T, V)` | Linear projection mapping $C = 384$ features to vocabulary size $V = 65$ |
| **Loss Calculation (Training)** | `(B*T, V)` vs `(B*T,)` | `Scalar ()` | Flatten across time and batch to compute Cross-Entropy Loss |
| **Sampling (Inference)** | `(B, 1, V)` | `(B, 1)` | Softmax over final position logits $\to$ Multinomial sampling |

---

## The Step-by-Step Learning Journey

This repository explores the full developmental arc from raw text to a functional generative model. The implementation details are outlined below.

### 1. Tokenization and Data Batching

- **Character-Level Tokenizer**:
  Rather than using subword tokenization (such as BPE), the model treats each unique character as a token. On `input.txt` (the Tiny Shakespeare dataset), there are 65 unique characters (`a-z`, `A-Z`, punctuation, spaces, and newlines).
  - String to integer: `stoi: dict[str, int]`
  - Integer to string: `itos: dict[int, str]`
- **Dataset Split**:
  The encoded integer sequence is split contiguously into 90% training data and 10% validation data.
- **Context Windows (`BLOCK_SIZE = 256`) and Shifted Targets**:
  During training, batches of random chunks of length `BLOCK_SIZE` are sampled:
  ```python
  x = source[i : i + BLOCK_SIZE]
  y = source[i + 1 : i + BLOCK_SIZE + 1]
  ```
  This creates pairs where each position $t \in [0, \text{BLOCK\_SIZE}-1]$ in $x$ is supervised to predict $y[t] = x[t+1]$, training the model on context lengths from $1$ token up to `BLOCK_SIZE` tokens simultaneously.

---

### 2. The Baseline: Bigram Language Model (`bigram.py`)

In [`bigram.py`](./bigram.py), each token predicts the next token based exclusively on its own identity, using an embedding lookup table without any inter-token communication:

$$\text{Logits}(x_t) = \text{EmbeddingTable}(x_t)$$

- **Limitation**: While simple and fast, a bigram model has zero memory of previous tokens ($x_{t-1}, x_{t-2}, \dots$).
- **Performance**: The loss plateaus at roughly $\sim 2.5$ nats (down from a purely random uniform loss of $-\ln(1/65) \approx 4.17$). The generated output contains English-like letter combinations, but lacks word structure, syntax, or context.

---

### 3. The Mathematical Trick in Self-Attention (4-Stage Progression)

To capture long-range dependencies, tokens need a mechanism to look back at previous tokens in the sequence. In [`gpt_dev.ipynb`](./gpt_dev.ipynb) and [`gpt.ipynb`](./gpt.ipynb), self-attention is derived through four developmental stages:

#### Version 1: Naive Averaging via For-Loops
For each position $t$, compute the mean of all past token representations up to $t$:
```python
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t + 1, C)
        xbow[b, t] = torch.mean(xprev, 0)
```
*Downside*: Extremely slow due to nested Python loops ($O(B \cdot T^2)$).

#### Version 2: Matrix Multiplication with Lower Triangular Matrix (`torch.tril`)
We can replace the loop with batched matrix multiplication. A lower-triangular matrix of ones normalized along each row acts as a rolling-average operator:
```python
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(dim=1, keepdim=True)
xbow = weights @ x  # (B, T, T) @ (B, T, C) -> (B, T, C)
```

#### Version 3: Softmax and Masking
Instead of manual row-normalization, apply `torch.masked_fill` with $-\infty$ to the upper triangle, followed by `F.softmax`:
```python
tril = torch.tril(torch.ones(T, T))
weights = torch.zeros((T, T))
weights = weights.masked_fill(tril == 0, float("-inf"))
weights = F.softmax(weights, dim=-1)  # -inf becomes 0.0 under softmax
xbow = weights @ x
```

#### Version 4: Self-Attention with Query, Key, and Value Projections
Uniform averaging is data-agnostic: every past token contributes equally. **Self-Attention** makes this affinity data-dependent:
1. **Query ($Q$)**: *"What kind of information am I looking for?"*
2. **Key ($K$)**: *"What kind of information do I contain?"*
3. **Value ($V$)**: *"What information do I communicate if you attend to me?"*

Each token emits a Query, Key, and Value vector via linear projections without bias:
```python
keys = self.key(x)      # (B, T, head_size)
queries = self.query(x)  # (B, T, head_size)
values = self.value(x)   # (B, T, head_size)

# Dot-product affinities
weights = queries @ keys.transpose(-2, -1) * (head_size ** -0.5)
# Causal masking: cannot look into the future
weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
weights = F.softmax(weights, dim=-1)

out = weights @ values  # (B, T, head_size)
```

---

### 4. Why the Scaling Factor \(\sqrt{d_k}\) is Vital

When $Q$ and $K$ have zero mean and unit variance ($\text{Var}(Q) = \text{Var}(K) = 1$):

$$\text{Var}\left(\sum_{i=1}^{d_k} Q_i K_i\right) = d_k$$

Without scaling, as $d_k$ increases, the variance of the dot products grows proportionally to $d_k$. Large dot-product values cause the softmax function to saturate:
- The largest value approaches $1.0$, while all others collapse to $0.0$.
- The softmax gradients vanish ($\frac{\partial \text{softmax}}{\partial z} \approx 0$), halting gradient backpropagation through the attention heads.

Dividing by $\sqrt{d_k}$ normalizes the variance back to $1.0$, keeping the softmax in an active, gradient-friendly regime.

---

### 5. Multi-Head Attention

A single attention head can only attend to one relationship pattern at a time (e.g., finding the previous consonant). **Multi-Head Attention** initializes multiple independent heads running in parallel:
- In [`v2.py`](./v2.py), each head produces an output of size `HEAD_SIZE`.
- The outputs of all heads are concatenated along the feature dimension:
  $$\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) \in \mathbb{R}^{B \times T \times (h \cdot d_k)}$$
- A linear projection layer projects this concatenated representation back to `N_EMBED`, followed by a dropout layer for regularization.

---

### 6. Position-Wise Feed-Forward Network

Self-attention is a **communication mechanism** where tokens exchange information across the time axis. However, tokens also need **computation time** to process and transform the aggregated data.

The [`FeedForward`](./v2.py#L58-L72) module applies a position-wise Multi-Layer Perceptron (MLP) independently to each token:

$$\text{FFN}(x) = \text{Dropout}\left(\max\left(0, x W_1 + b_1\right) W_2 + b_2\right)$$

- $W_1$ expands the dimension by a factor of 4: from `N_EMBED` to $4 \times \text{N\_EMBED}$.
- A non-linear activation (`ReLU`) is applied.
- $W_2$ projects back down from $4 \times \text{N\_EMBED}$ to `N_EMBED`.
- Dropout is applied for regularization.

---

### 7. Residual Connections and Pre-LayerNorm

Deep networks (stacking 6 or more Transformer blocks) are prone to vanishing gradients during backpropagation. Two architectural choices ensure stable gradient flow:

1. **Residual (Skip) Connections**:
   $$x \leftarrow x + \text{Sublayer}(\text{LayerNorm}(x))$$
   The addition operator acts as a **gradient highway**, allowing loss gradients to flow back directly through the skip branches without attenuation.
2. **Layer Normalization vs. Batch Normalization**:
   Unlike Batch Normalization (which normalizes across the batch dimension and depends on batch statistics), LayerNorm computes the mean and variance across the feature dimension for each token independently:
   $$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
3. **Pre-LN vs. Post-LN**:
   The original 2017 Transformer paper (*"Attention Is All You Need"*) used Post-LN ($x \leftarrow \text{LN}(x + \text{Sublayer}(x))$). Modern GPT models use **Pre-LN** (normalizing the input before each sublayer). This improves numerical stability and enables training deep architectures without sensitive learning rate warm-up schedules.

---

### 8. Assembling the Full Transformer Decoder (`v2.py`)

All components come together in [`BigramLanguageModel`](./v2.py#L89-L126):
1. **Token Embeddings** (`vocab_size` $\to$ `N_EMBED`) and **Learned Position Embeddings** (`BLOCK_SIZE` $\to$ `N_EMBED`) are summed.
2. The sequence passes sequentially through `N_BLOCKS` Transformer blocks.
3. A final LayerNorm is applied to the output of the final block.
4. A linear Language Model head (`lm_head`) projects the embeddings to vocabulary logits.
5. In training mode, cross-entropy loss is computed against targets shifted by 1 position.
6. In generation mode, tokens are sampled autoregressively from the softmax distribution.

---

## Mathematical Formulations

| Operation | Equation |
| :--- | :--- |
| **Input Representation** | $x^{(0)}_t = \text{Embed}_{\text{tok}}(w_t) + \text{Embed}_{\text{pos}}(t)$ |
| **Scaled Dot-Product Attention** | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$ |
| **Causal Mask** | $M_{i,j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$ |
| **Multi-Head Projection** | $\text{MHA}(x) = \left[\text{head}_1 \,\|\, \text{head}_2 \,\|\, \dots \,\|\, \text{head}_h\right] W^O$ |
| **Transformer Block (Pre-LN)** | $x' = x + \text{MHA}(\text{LN}_1(x))$<br>$x'' = x' + \text{FFN}(\text{LN}_2(x'))$ |
| **Cross-Entropy Loss** | $\mathcal{L} = -\frac{1}{B \cdot T} \sum_{i=1}^{B \cdot T} \log P(y_i \mid x_{\le i})$ |

---

## Model Configurations and Hyperparameters

Summary of the hyperparameters configured in [`v2.py`](./v2.py) and [`bigram.py`](./bigram.py):

| Hyperparameter | Baseline (`bigram.py`) | Scaled GPT (`v2.py`) | Purpose |
| :--- | :---: | :---: | :--- |
| **Vocabulary Size** | 65 | 65 | Distinct characters in Shakespeare dataset |
| **Batch Size (`BATCH_SIZE`)** | 32 | 64 | Independent sequences processed in parallel |
| **Context Length (`BLOCK_SIZE`)** | 8 | 256 | Maximum tokens the model can look back at |
| **Embedding Width (`N_EMBED`)** | 32 | 384 | Channel dimension of token/position vectors |
| **Transformer Blocks (`N_BLOCKS`)** | 0 | 6 | Number of stacked Transformer layers |
| **Attention Heads (`num_heads`)** | — | 4 | Number of parallel self-attention heads |
| **Head Dimension (`HEAD_SIZE`)** | — | 6 | Dimension of key, query, and value vectors |
| **Dropout (`DROPOUT`)** | 0.0 | 0.2 | Probability of zeroing activations during training |
| **Learning Rate** | `1e-2` | `3e-4` | Learning rate for AdamW optimizer |
| **Training Steps (`MAX_ITERS`)** | 3000 | 5000 | Total optimization steps |
| **Total Parameters** | ~2.1 K | **~7.47 M** | Full trainable parameter count |

---

## Training Dynamics and Generation

### Loss Progression
- **Initial Random Loss**: $\approx -\ln(1/65) \approx 4.17$ nats.
- **Bigram Loss**: Drops to $\sim 2.50$, where it hits a performance ceiling due to lack of historical context.
- **Transformer Decoder Loss**: Progresses down to $\sim 1.5 - 1.8$, learning syntax, dialogue formatting, Shakespearean vocabulary, and character tags.

### Qualitative Generation Comparison: Baseline vs. Transformer

To appreciate the leap in modeling capacity that self-attention brings, compare sample generation outputs before and after introducing Transformer blocks:

#### 1. Baseline Bigram Output (`bigram.py` — No Attention, Only Pairwise Probabilities):
```text
oTo.JUZ!!zqe!
xBP qbs$Gy'AcOmrLwwt
p$x;Seh-onQbfM?OjKbn'NwUAW -Np3fkz$FVwAUEa-wzWC -wQo-R!v -Mj?,SP
Wh!mR;v$o;a!xLg;j.s?w;l?gR-o?W.b
```
*Observation*: Random character noise with occasional vowel-consonant combinations (`on`, `th`), but no word structures, grammar, or thematic continuity.

#### 2. Transformer Decoder Output (`v2.py` after 5,000 steps):
```text
KING RICHARD:
Why, what is he that doth attend the king?

GLOUCESTER:
He is a gentleman of good esteem,
And well beloved of the noble duke;
Yet hath his fortune turned from his hand,
And left him naked in the world to freeze.

LADY ANNE:
Come, fellow, let me see thy face again;
Thou shalt not have thy will against my lord.
```
*Observation*: The network has learned script formatting, character capitalization, dialogue colons, meter, archaic English vocabulary, and plausible sentence structure purely from raw characters!

### Autoregressive Text Generation
Text generation proceeds one token at a time:
1. Initialize the sequence with a single prompt token (e.g., token `0`).
2. Crop the current sequence to the most recent `BLOCK_SIZE` characters.
3. Perform a forward pass to compute logits for all positions.
4. Extract logits at the final position: `logits[:, -1, :]`.
5. Convert logits to probabilities via softmax.
6. Sample the next token index from this distribution using `torch.multinomial`.
7. Concatenate the sampled token to the context and repeat.

---

## Project Files

| File | Description |
| :--- | :--- |
| [`v2.py`](./v2.py) | **Main production script**: complete character-level Transformer with multi-head causal self-attention, feed-forward layers, residual connections, layer normalization, training loop, and autoregressive generation. |
| [`bigram.py`](./bigram.py) | **Baseline model**: minimal character-level bigram model using only token embedding lookup tables for comparison. |
| [`gpt_dev.ipynb`](./gpt_dev.ipynb) | **Development notebook**: interactive walkthrough deriving self-attention from scratch (naive loops $\to$ matrix multiply $\to$ softmax masking $\to$ scaled $Q, K, V$), LayerNorm implementation, and parameter scaling experiments. |
| [`gpt.ipynb`](./gpt.ipynb) | **Scratchpad notebook**: exploratory experiments with dataset loading, character mappings, and self-attention toy examples. |
| `input.txt` | The Tiny Shakespeare corpus (~1.1 MB, 1,115,394 characters) used as the local training and validation dataset. |

---

## Requirements and How to Run

### Requirements
- Python 3.8+
- PyTorch (with CUDA support if a GPU is available)

Install PyTorch according to your platform using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Running the Transformer (`v2.py`)

```bash
python3 v2.py
```

The script will automatically use CUDA if available, train for 5,000 iterations while periodically printing train and validation loss, and conclude by generating 500 characters of Shakespeare-style text.

### Running the Baseline (`bigram.py`)

```bash
python3 bigram.py
```

Runs the bigram baseline for 3,000 steps to demonstrate the difference in generation quality when no attention mechanism is present.

---

## Key Takeaways and Insights

1. **Attention as Communication**: Self-attention can be viewed as a directed graph where every token looks back at preceding tokens and pulls relevant information.
2. **Feed-Forward as Computation**: While attention aggregates information across time, the feed-forward network lets each token reflect on and process the aggregated data independently.
3. **Causality through Masking**: Replacing future affinities with $-\infty$ ensures tokens can never look ahead into future positions, preserving the autoregressive property.
4. **Gradient Highways**: Deep Transformer models rely critically on residual connections and pre-layer normalization to maintain stable gradient magnitudes throughout backpropagation.
5. **From Scratch Understanding**: Building each layer manually reveals exactly why modern LLMs behave the way they do — from context window limitations to autoregressive sampling mechanics.
