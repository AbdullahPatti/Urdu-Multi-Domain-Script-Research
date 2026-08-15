# Domain Robustness of Multilingual NLP Models Across Urdu and Roman Urdu Scripts 📊🔍

## Overview

This repository presents the first systematic study of **domain robustness** in Urdu and Roman Urdu NLP models. The research investigates how well NLP models maintain their accuracy when moved across different data sources within the same language, script, and task—a critical measure for real-world deployment.

> **Domain robustness**: The ability of an NLP model to keep its accuracy when moved from one data source to another within the same language and task.

## 🎯 Research Contributions

### 1. Comprehensive Benchmark
- **32 datasets** across **8 tasks**, **2 scripts** (Nastaliq & Roman Urdu), and **5 models**
- Built under a single, consistent protocol based on Calderon et al.'s SS/ST framework
- Addresses the gap in multilingual NLP research for Urdu languages

### 2. Task-Dependent Fragmentation Evidence
- **Sentiment Analysis**: 90-97% relative robustness (highly robust)
- **Fake News Detection**: 48-54% relative robustness (highly fragile)
- **Hate Speech Detection**: 61-76% relative robustness (moderately fragile)

### 3. Training Paradigm Comparison
- **Fine-tuned Transformers**: Highest in-domain scores (70-99%) but worst cross-domain generalization
- **Few-shot LLMs**: Lower in-domain scores (38-79%) but better transfer robustness (73-89%)

### 4. Script-Independent Analysis
- Nastaliq and Roman Urdu evaluated as separate tracks
- Accounts for different task sets and domain counts between scripts
- Prevents pooling of unrelated tasks for misleading comparisons

### 5. Actionable Recommendations
- Task-specific guidance for Urdu NLP practitioners
- Honest discussion of benchmark limitations
- Concrete insights for building robust Urdu NLP systems

## 📋 Quick Stats

| Script | Domains | Tasks | Total Datasets |
|--------|---------|-------|----------------|
| Nastaliq | 18 | 4 | 32% of benchmark |
| Roman Urdu | 14 | 4 | 18% of benchmark |
| **Total** | **32** | **8** | **100%** |

### Models Evaluated
- **Fine-tuned**: XLM-RoBERTa (250M params), mBERT (110M params)
- **Few-shot LLMs**: Llama3.1-8B, Mistral-7B, Qwen2.5-7B

## 🔍 Key Findings

### Domain Robustness Hierarchy

| Robustness Level | Tasks | Relative Robustness | Characteristics |
|------------------|-------|---------------------|----------------|
| **High** (90-97%) | Sentiment Analysis | 90-97% | Coarse semantic signals stable across domains |
| **Moderate** (67-83%) | QA, Cyber Abuse, Product Reviews, Twitter Opinions | 67-83% | Platform conventions transfer reasonably well |
| **Low** (48-76%) | Fake News, Hate Speech | 48-76% | Domain-specific surface patterns dominate |

### Fine-tuned vs. Few-shot LLMs

| Aspect | Fine-tuned Transformers | Few-shot LLMs |
|--------|-------------------------|--------------|
| In-domain (SS) | 70-99% | 38-79% |
| Cross-domain (ST) | 48-76% | 68-90% |
| Key Pattern | High SS, low ST | Lower SS, higher ST |

## 🛠️ Methodology

### Evaluation Protocol
- **SS (Same-Source)**: In-domain accuracy (training/evaluation on same domain)
- **ST (Source-to-Target)**: Cross-domain accuracy (training on domain A, testing on domain B)
- **Degradation**: SS - ST (absolute accuracy loss)
- **Relative Robustness**: ST/SS × 100% (percentage of in-domain performance retained)

### Dataset Construction
- **18 domains** (Nastaliq script): Sentiment (5), Hate Speech (5), Fake News (4), QA (4)
- **14 domains** (Roman Urdu script): Sentiment (4), Cyber Abuse (4), Product Reviews (3), Twitter Opinions (4)
- Constructed synthetic domains with LLM assistance (manually reviewed by Urdu speakers)
- Mixed sources: Kaggle, Hugging Face, GitHub, and existing published corpora

### Model Training & Evaluation
- **Fine-tuned Transformers**: 3 epochs, learning rate 2×10⁻⁵, early stopping
- **Few-shot LLMs**: 5 labeled examples, temperature 0.7, max 20 tokens
- Hardware: Intel Core i7 (13th Gen), 32GB RAM, NVIDIA RTX A4000

## 📊 Results by Task & Script

### Nastaliq Script Results

| Task | Best Model (SS/ST) | Relative Robustness | Key Insight |
|------|-------------------|---------------------|-------------|
| Sentiment Analysis | XLM-R (0.733/0.713) | 97.3% | Vocabulary remarkably stable across domains |
| Hate Speech | XLM-R (0.885/0.570) | 64.4% | Different domains target different groups/vocabularies |
| Fake News | XLM-R (0.890/0.430) | 48.3% | Severe domain-specific overfitting |
| Question Answering | XLM-R (0.765/0.518) | 67.7% | Structural differences affect cross-domain performance |

### Roman Urdu Script Results

| Task | Best Model (SS/ST) | Relative Robustness | Key Insight |
|------|-------------------|---------------------|-------------|
| Sentiment Analysis | XLM-R (0.754/0.685) | 90.8% | Robust despite spelling inconsistencies |
| Cyber Abuse | XLM-R (0.997/0.817) | 82.0% | Platform-specific vocabulary differences |
| Product Reviews | XLM-R (0.865/0.712) | 82.3% | Domain-specific phrasing conventions |
| Twitter Opinions | XLM-R (0.832/0.689) | 82.8% | Mixed content affects cross-domain transfer |

## 🎭 Roman vs. Nastaliq Analysis

### Script Characteristics
- **Nastaliq**: Right-to-left, Perso-Arabic script, edited sources (news, literature)
- **Roman Urdu**: Left-to-right, Latin script, user-generated, inconsistently spelled

### Cross-Domain Behavior
- **Sentiment Analysis**: Comparable robustness (~90% for both scripts)
- **Other Tasks**: Different task sets limit direct script comparison
- **Shared Challenges**: Domain-specific vocabulary and platform conventions

## 💡 Practical Implications

### For Researchers
1. **Task-aware evaluation**: Not all NLP tasks transfer equally well
2. **Model selection trade-offs**: Fine-tuning vs. few-shot for different use cases
3. **Script considerations**: Nastaliq vs. Roman Urdu require separate analysis

### For Practitioners
1. **Domain diversity in training**: Critical for tasks like fake news and hate speech
2. **Hybrid approaches**: Consider few-shot LLMs for cross-domain robustness
3. **Task-specific strategies**: Different mitigation strategies per robustness tier

## 📈 Comparison with English Benchmark

### English (Calderon et al.)
- **Sentiment Analysis**: ~85-90% relative robustness
- **Urdu Sentiment**: 90-97% relative robustness
- **Urdu Sentiment**: **Competitive** with English

### Concerning Gaps
- **Urdu Fake News**: ~30 percentage points behind English
- **Urdu Hate Speech**: ~15-20 percentage points behind English
- **Root cause**: Domain-specific overfitting and limited multilingual training data

## 🚀 Limitations & Future Work

### Current Benchmark Limitations
1. **LLM-generated domains**: Synthetic text may not match natural domains
2. **Static evaluation**: No fine-tuning on target domains
3. **Limited language coverage**: Only two Urdu scripts, multiple dialects unexplored

### Future Directions
1. **Real-world deployment**: Field testing on live social media platforms
2. **Dynamic domain adaptation**: Techniques for continual learning
3. **Dialectal variation**: Expanding to Urdu dialects (e.g., Dakhani, Saraiki)
4. **Code-switching**: Evaluating models on mixed-script content

## 🏆 Key Take-Aways

1. **Task matters more than model**: Some NLP tasks inherently resist domain shift
2. **Fine-tuning has trade-offs**: High performance comes with domain specificity
3. **Few-shot LLMs offer robustness**: Transferability without catastrophic forgetting
4. **Urdu NLP is maturing**: Sentiment analysis reaches English-level robustness
5. **Fake news and hate speech need work**: 30+ point gaps require targeted research

## 📚 Citation

```bibtex
@inproceedings{haroon2024domain,
  title={Domain Robustness of Multilingual NLP Models Across Urdu and Roman Urdu Scripts},
  author={Haroon, Muhammad Abdullah and Bashir, Maryam},
  booktitle={ACL},
  year={2024},
  organization={FAST-NUCES Lahore}
}
```

## 🔗 Related Links

- **Paper**: `urdu_domain_robustness_paper_v2.pdf`
- **Results**: `results/` directory with performance metrics
- **Figures**: `results/figures/` with visualization outputs
- **Datasets**: Documentation in dataset construction methodology

## 🤝 Contributing

This is a research artifact for domain robustness evaluation. Contributions are welcome for:
- Dataset construction (new domains or tasks)
- Model implementation (additional architectures)
- Analysis (new metrics or comparisons)
- Documentation (clarifications and extensions)

## 🚀 Getting Started

This repository contains the complete research package for Urdu domain robustness evaluation. For technical questions about the methodology, model implementations, or dataset usage, please refer to the code comments and methodology documentation throughout the repository.

The study establishes a foundational benchmark for Urdu NLP research and provides concrete guidance for both researchers and practitioners working with multilingual, low-resource languages.