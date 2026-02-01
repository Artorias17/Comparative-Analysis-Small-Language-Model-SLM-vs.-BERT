# Comparative Analysis: SLM vs BERT

**Course**: Deep Learning in Human Language Technology

**Corpus:** IMDB Dataset

## Overview

This project compares three approaches to sentiment analysis on movie reviews:
1. Fine-tuning bidirectional model (BERT)
2. Few-shot prompting with a generative model (SmolLM)
3. Instruction Fine-tuning generative model (SmolLM)


## Models Used

- **Bidirectional Model:** [`google-bert/bert-base-cased`](https://huggingface.co/google-bert/bert-base-cased)
- **Generative Model:** [`HuggingFaceTB/SmolLM-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)

## Dataset

**IMDB Dataset** - A large movie review dataset for sentiment classification
- 25,000 highly polar training reviews
- 25,000 test reviews
- Binary labels: negative (0) or positive (1)
- Random baseline: 50%

### Sampling Strategy

- **Training set:** 5,000 samples (stratified)
- **Validation set:** 1,000 samples (stratified)
- **Test set:** 1,000 samples (stratified)
- Equal distribution of positive and negative examples in all sets

## Installation

```bash
# Install dependencies
uv sync
```

## Methodology

### 1. BERT Fine-tuning

**Hyperparameters:**
- Learning rate: 5e-6
- Batch size: 16 (train), 32 (eval)
- Max steps: 1000
- Optimizer: AdamW (fused)
- Precision: bf16

**Results:**
- Test accuracy: **87.2%**

### 2. SmolLM Few-shot Prompting

**Approach:**
- 4-shot prompting with custom instruction
- Zero padding, direct text generation
- Temperature: 0.1
- Max new tokens: 1

**Instruction:**
```
You are a sentiment classifier. Analyze movie reviews and classify them as either "positive" or "negative". 
Only reply with "positive" or "negative". Do not include any additional texts. Do not use any capital characters.
```

**Results:**
- Test accuracy: **69.4%**
- Format errors: 26 (2.60%)

### 3. SmolLM Fine-tuning

**Hyperparameters:**
- Learning rate: 5e-5
- Batch size: 4 (with gradient accumulation steps: 4)
- Max steps: 1000
- Optimizer: AdamW (fused)
- Precision: bf16

**Special considerations:**
- Instruction fine-tuning with prompt masking
- Only evaluate loss on the output token
- Custom accuracy computation for causal language models

**Results:**
- Test accuracy: **79.8%**
- Format errors: 0 (0.00%)

## Key Findings

### Model Agreement Analysis

Among 1,000 test examples:
- **All three models correct:** 544 (54.4%)
- **All three models wrong:** 40 (4.0%)
- **Exactly one model correct:** 118 (11.8%)
- **Exactly two models correct:** 298 (29.8%)

### Error Analysis

**SmolLM Few-shot:**
- Format errors: 26 (invalid output format)
- Semantic errors: 298 (wrong label prediction)

**SmolLM Fine-tuned:**
- Format errors: 0 (perfect instruction following)
- Semantic errors: 202 (wrong label prediction)

### Key Insights

1. **BERT performed best** with 87.2% accuracy, demonstrating the effectiveness of bidirectional models for classification tasks.

2. **Fine-tuning improved SmolLM significantly** (69.4% → 79.8%), with perfect instruction following (no format errors).

3. **Few-shot prompting challenges:** The model sometimes generated multiple tokens or used variations like "neutral," despite clear instructions.

4. **Instruction fine-tuning complexity:** Required:
   - Custom dataset formatting with instruction-review-response structure
   - Manual label masking to focus only on output tokens
   - Modified accuracy computation to handle causal LM predictions
   - Shifted prediction alignment (prediction for token i is at position i-1)

## Contributions

**Abhishek Roy:** Notebook setup, dataset preprocessing, training pipeline creation

**Shaktiman Choudhury:** Related works research, training execution, results evaluation, error analysis

## References

- **Maas et al. (2011):** [Learning Word Vectors for Sentiment Analysis](http://www.aclweb.org/anthology/P11-1015). *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.*

- **Zhang et al. (2024):** [Large Language Models for Sentiment Analysis: A Survey and Comparative Study](https://arxiv.org/abs/2409.02370). *arXiv preprint arXiv:2409.02370*
