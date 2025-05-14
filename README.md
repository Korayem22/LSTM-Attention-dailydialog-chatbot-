#  DailyDialog Chatbot (Seq2Seq with Attention)

This project implements a sequence-to-sequence (Seq2Seq) chatbot using PyTorch, trained on the [DailyDialog dataset](https://huggingface.co/datasets/daily_dialog). The architecture combines an LSTM encoder-decoder with Bahdanau attention, pretrained GloVe embeddings, scheduled teacher forcing, and beam search decoding.

> This model is a **proof of concept**. While it demonstrates the structure of modern chatbots, it does not yet achieve fluent human-level interaction.

---

##  Features

- Bi-directional LSTM Encoder (2 layers)
- Unidirectional LSTM Decoder with Attention
- Bahdanau (additive) attention mechanism
- Pretrained **GloVe embeddings (100D)** (trainable)
- Beam Search decoding (instead of greedy)
- Scheduled teacher forcing decay
- Label smoothing for generalization
- Dropout to reduce repetition and overfitting

---

## Architecture

```text
[SOURCE SENTENCE]
     │
 [Embedding Layer (GloVe)]
     │
[BiLSTM Encoder] <── Attention ──┐
     │                           │
[Hidden State Projection]        │
     │                           ▼
[Unidirectional LSTM Decoder] ──► Beam Search Output
