# SIFThinker: Spatially-Aware Image Focus for Visual Reasoning
<div align="center">
<h1>SIFThinker: Spatially-Aware Image Focus for Visual Reasoning</h1>

**The official code has been migrated to [Bytedance-repo](https://github.com/bytedance/SIFThinker). Please refer to the official repository for further details.**

Zhangquan Chen, Ruihui Zhao, Chuwei Luo, Mingze Sun, Xinlei Yu, 

Yangyang Kang, Ruqi Huang
</div>

## Instruction
Current multimodal large language models (MLLMs) still face significant challenges in complex visual tasks (e.g., spatial understanding, fine-grained perception). Prior methods have tried to incorporate visual reasoning, however, they fail to leverage attention correction with spatial cues to iteratively refine their focus on prompt-relevant regions. In this paper, we introduce SIFThinker, a spatially-aware “think-with-images” framework that mimics human visual perception. Specifically, SIFThinker enables attention correcting and image region focusing by interleaving depth-enhanced bounding boxes and natural language. Our contributions are twofold: First, we introduce a reverse-expansion-forward-inference strategy that facilitates the generation of interleaved image-text chains of thought for process-level supervision, which in turn leads to the construction of the SIF-50K dataset. Besides, we propose GRPO-SIF, a reinforced training paradigm that integrates depth-informed visual grounding into a unified reasoning pipeline, teaching the model to dynamically correct and focus on prompt-relevant regions. Extensive experiments demonstrate that SIFThinker outperforms state-of-the-art methods in spatial understanding and fine-grained visual perception, while maintaining strong general capabilities, highlighting the effectiveness of our method.

## Bibtex
If you find SIFThinker helpful for your work, please cite

```
@article{chen2025sifthinker,
  title={SIFThinker: Spatially-Aware Image Focus for Visual Reasoning},
  author={Chen, Zhangquan and Zhao, Ruihui and Luo, Chuwei and Sun, Mingze and Yu, Xinlei and Kang, Yangyang and Huang, Ruqi},
  journal={arXiv preprint arXiv:2508.06259},
  year={2025}
}
```
