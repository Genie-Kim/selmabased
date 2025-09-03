# Long-Text-to-Image Generation Analysis with Davidsonian Scene Graph

[![Technical Report](https://img.shields.io/badge/Technical%20Report-Project%20Page-blue)](https://genie-kim.github.io/projects/LongT2IEval/) [![Dataset](https://img.shields.io/badge/Dataset-Docci--Llama3--70B-green)](https://github.com/Genie-Kim/Longprompt_T2I_Analysis/blob/main/Docci-Llama3-70B-DescriptionSummary.jsonlines)

## Overview

This repository contains research on addressing the critical 77-token limitation in CLIP-based text-to-image diffusion models. While current models excel at generating images from short prompts, they fail when processing complex scene descriptions that exceed ~3-4 sentences due to CLIP's architectural constraints.

## Research Contribution

Through comprehensive evaluation on the Docci dataset (15K human-annotated long descriptions), we identify **two fundamental bottlenecks** in long-text processing:

1. **Text Encoder Limitation**: Despite CLIP's 77-token capacity, its effective encoding length is only ~20-30 tokens
2. **Cross-Attention Degradation**: U-Net's attention mechanism progressively fails as condition length increases

Our analysis evaluates multiple approaches including:
- Extended positional embeddings (LongCLIP)
- Chunk-based concatenation strategies
- LLM-based summarization
- End-to-end learning with perceiver modules (ELLA)

**Key Finding**: ELLA's perceiver-based architecture achieves superior performance by learning to compress long prompts end-to-end, successfully addressing both encoding and attention limitations inherent in CLIP-based models.

## Dataset Resources

### Docci-Llama3-70B Summary Dataset
We provide a preprocessed version of the Docci dataset with Llama3-70B generated summaries:
- **Download**: [Docci-Llama3-70B-DescriptionSummary.jsonlines](https://github.com/Genie-Kim/Longprompt_T2I_Analysis/blob/main/Docci-Llama3-70B-DescriptionSummary.jsonlines)
- **Format**: JSONL with original descriptions and compressed summaries
- **Usage**: Evaluation of summarization-based approaches for long-text handling

## Installation

For environment setup and dependencies, please refer to the [SELMA repository](https://github.com/jialuli-luka/SELMA) installation instructions, which provides a compatible framework for text-to-image evaluation.

## Evaluation Framework

We extend TIFA (Text-to-Image Faithfulness Assessment) with Davidsonian Scene Graph (DSG) for comprehensive evaluation across multiple dimensions:
- Entity recognition
- Attribute preservation  
- Spatial relationships
- Action detection
- Global scene understanding

## Key Insights

1. **Simple extensions fail**: Positional embedding interpolation (LongCLIP) and concatenation methods cannot overcome fundamental architectural limitations

2. **Dual bottleneck problem**: Both text encoding quality and cross-attention capacity degrade with length, requiring joint optimization

3. **End-to-end learning succeeds**: ELLA demonstrates that learning compression and alignment jointly provides the most effective solution

4. **Computational challenges remain**: Despite superior performance, ELLA requires 56 GPU-days (8×A100) for training, necessitating research into more efficient alternatives

## Future Directions

- Efficient training strategies through knowledge distillation
- Multi-concept decomposition with hierarchical processing
- Lightweight perceiver architectures
- Hybrid approaches combining summarization with minimal fine-tuning

## Citation

```bibtex
@article{kim2024longtext,
  title={Analysis of Long-Text-to-Image Generation with Davidsonian Scene Graph Evaluation},
  author={Kim, Jin},
  year={2024},
  institution={Yonsei University, Digital Image Media Lab}
}
```

## Related Work

- [ELLA: Equip Diffusion Models with LLM for Enhanced Semantic Alignment](https://github.com/TencentQQGYLab/ELLA)
- [Long-CLIP: Unlocking the Long-Text Capability of CLIP](https://github.com/beichenzbc/Long-CLIP)
- [SELMA: Learning and Merging Skill-Specific Text-to-Image Experts](https://github.com/jialuli-luka/SELMA)

## Contact

For questions and discussions, please open an issue or contact: kimjin928@yonsei.ac.kr
