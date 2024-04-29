# Awesome-Quantization-Papers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo contains a comprehensive paper list of **Model Quantization** for efficient deep learning on AI conferences/journals/arXiv. As a highlight, we categorize the papers in terms of model structures and application scenarios, and label the quantization methods with keywords. <br>

This repo is being actively updated, and contributions in any form to make this list more comprehensive are welcome. Special thanks to collaborator [Zhikai Li](https://github.com/zkkli), and all researchers who have contributed to this repo! <br> 

If you find this repo useful, please consider **★STARing** and feel free to share it with others! <br>

**[Update: Apr, 2024]** Add new papers from AAAI-24. <br>
**[Update: Nov, 2023]** Add new papers from NeurIPS-23. <br>
**[Update: Oct, 2023]** Add new papers from ICCV-23. <br>
**[Update: Jul, 2023]** Add new papers from AAAI-23 and ICML-23. <br>
**[Update: Jun, 2023]** Add new arXiv papers uploaded in May 2023, especially the hot LLM quantization field. <br>
**[Update: Jun, 2023]** Reborn this repo! New style, better experience! <br>

---
## Overview

- [Awesome-Quantization-Papers ](#awesome-quantization-papers-)
  - [Overview](#overview)
  - [Survey](#survey)
  - [Transformer-based Models](#transformer-based-models)
    - [Vision Transformers](#vision-transformers)
    - [Language Transformers](#language-transformers)
  - [Convolutional Neural Networks](#convolutional-neural-networks)
    - [Visual Generation](#visual-generation)
    - [Image Classification](#image-classification)
    - [Other Tasks](#other-tasks)
      - [Object Detection](#object-detection)
      - [Super Resolution](#super-resolution)
      - [Point Cloud](#point-cloud)
  - [References](#references)

**Keywords**: **`PTQ`**: post-training quantization | **`Non-uniform`**: non-uniform quantization | **`MP`**: mixed-precision quantization | **`Extreme`**: binary or ternary quantization

---


## Survey
- "A Survey of Quantization Methods for Efficient Neural Network Inference", Book Chapter: Low-Power Computer Vision, 2021. [[paper](https://arxiv.org/abs/2103.13630)]
- "Full Stack Optimization of Transformer Inference: a Survey", arXiv, 2023. [[paper](https://arxiv.org/abs/2302.14017)]
- "A White Paper on Neural Network Quantization", arXiv, 2021. [[paper](https://arxiv.org/abs/2106.08295)]
- "Binary Neural Networks: A Survey", PR, 2020. [[Paper](https://arxiv.org/abs/2004.03333)] [**`Extreme`**]


## Transformer-based Models
### Vision Transformers
- "Bi-ViT: Pushing the Limit of Vision Transformer Quantization", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28109)] [**`Extreme`**]
- "AQ-DETR: Low-Bit Quantized Detection Transformer with Auxiliary Queries", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29487)]
- "LRP-QViT: Mixed-Precision Vision Transformer Quantization via Layer-wise Relevance Propagation", arXiv, 2023. [[paper](http://arxiv.org/abs/2401.11243)] [**`PTQ`**] [**`MP`**] 
- "MPTQ-ViT: Mixed-Precision Post-Training Quantization for Vision Transformer", arXiv, 2023. [[paper](http://arxiv.org/abs/2401.14895)] [**`PTQ`**] [**`MP`**] 
- "I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_I-ViT_Integer-only_Quantization_for_Efficient_Vision_Transformer_Inference_ICCV_2023_paper.pdf)] [[code](https://github.com/zkkli/I-ViT)]
- "RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_RepQ-ViT_Scale_Reparameterization_for_Post-Training_Quantization_of_Vision_Transformers_ICCV_2023_paper.pdf)] [[code](https://github.com/zkkli/RepQ-ViT)] [**`PTQ`**]
- "QD-BEV: Quantization-aware View-guided Distillation for Multi-view 3D Object Detection", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_QD-BEV__Quantization-aware_View-guided_Distillation_for_Multi-view_3D_Object_Detection_ICCV_2023_paper.pdf)]
- "BiViT: Extremely Compressed Binary Vision Transformers", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/He_BiViT_Extremely_Compressed_Binary_Vision_Transformers_ICCV_2023_paper.pdf)] [**`Extreme`**]
- "Jumping through Local Minima: Quantization in the Loss Landscape of Vision Transformers", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Frumkin_Jumping_through_Local_Minima_Quantization_in_the_Loss_Landscape_of_ICCV_2023_paper.pdf)]
- "PackQViT: Faster Sub-8-bit Vision Transformers via Full and Packed Quantization on the Mobile", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/71880)]
- "Oscillation-free Quantization for Low-bit Vision Transformers", ICML, 2023. [[paper](https://openreview.net/forum?id=DihXH24AdY)] [[code](https://github.com/nbasyl/OFQ)]
- "PSAQ-ViT V2: Towards Accurate and General Data-Free Quantization for Vision Transformers", TNNLS, 2023. [[paper](https://arxiv.org/abs/2209.05687)] 
- "Variation-aware Vision Transformer Quantization", arXiv, 2023. [[paper](http://arxiv.org/abs/2307.00331)]
- "NoisyQuant: Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers", CVPR, 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_NoisyQuant_Noisy_Bias-Enhanced_Post-Training_Activation_Quantization_for_Vision_Transformers_CVPR_2023_paper.pdf)]  [**`PTQ`**]
- "Boost Vision Transformer with GPU-Friendly Sparsity and Quantization", CVPR, 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Boost_Vision_Transformer_With_GPU-Friendly_Sparsity_and_Quantization_CVPR_2023_paper.pdf)] 
- "Q-DETR: An Efficient Low-Bit Quantized Detection Transformer", CVPR, 2023. [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Xu_Q-DETR_An_Efficient_Low-Bit_Quantized_Detection_Transformer_CVPR_2023_paper.html)]
- "Output Sensitivity-Aware DETR Quantization", 2023. [[paper](https://practical-dl.github.io/2023/extended_abstract/4/CameraReady/4.pdf)]
- "Q-HyViT: Post-Training Quantization for Hybrid Vision Transformer with Bridge Block Reconstruction", arXiv, 2023. [[paper](https://arxiv.org/abs/2303.12557)]  [**`PTQ`**]
- "Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer", NeurIPS, 2022. [[paper](https://openreview.net/forum?id=fU-m9kQe0ke)] [[code](https://github.com/yanjingli0202/q-vit)]
- "Patch Similarity Aware Data-Free Quantization for Vision Transformers", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710154.pdf)] [[code](https://github.com/zkkli/psaq-vit)]  [**`PTQ`**]
- "PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720190.pdf)] [[code](https://github.com/hahnyuan/ptq4vit)]  [**`PTQ`**]
- "FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer", IJCAI, 2022. [[paper](https://arxiv.org/abs/2111.13824)]  [[code](https://github.com/megvii-research/FQ-ViT)]  [**`PTQ`**]
- "Q-ViT: Fully Differentiable Quantization for Vision Transformer", arXiv, 2022. [[paper](https://arxiv.org/pdf/2201.07703.pdf)]
- "Post-Training Quantization for Vision Transformer", NeurIPS, 2021. [[paper](https://openreview.net/forum?id=9TX5OsKJvm)]  [**`PTQ`**]


[[Back to Overview](#overview)]

### Language Transformers
- "OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29237)]
- "Norm Tweaking: High-Performance Low-Bit Quantization of Large Language Models", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29815)]
- "Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29860)]
- "Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29908)]  [**`PTQ`**]
- "What Makes Quantization for Large Language Model Hard? An Empirical Study from the Lens of Perturbation", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29765)]
- "EasyQuant: An Efficient Data-free Quantization Algorithm for LLMs", arXiv, 2024. [[paper](http://arxiv.org/abs/2403.02775)]
- "IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact", arXiv, 2024. [[paper](http://arxiv.org/abs/2403.01241)]
- "FlattenQuant: Breaking Through the Inference Compute-bound for Large Language Models with Per-tensor Quantization", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.17985)]
- "A Comprehensive Evaluation of Quantization Strategies for Large Language Models", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.16775)]
- "GPTVQ: The Blessing of Dimensionality for LLM Quantization", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.15319)]
- "APTQ: Attention-aware Post-Training Mixed-Precision Quantization for Large Language Models", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.14866)]
- "EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.10787)]
- "RepQuant: Towards Accurate Post-Training Quantization of Large Transformer Models via Scale Reparameterization", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.05628)]
- "Accurate LoRA-Finetuning Quantization of LLMs via Information Retention", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.05445)]
- "BiLLM: Pushing the Limit of Post-Training Quantization for LLMs", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.04291)]
- "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", arXiv, 2023. [[paper](http://arxiv.org/abs/2401.18079)]
- "Extreme Compression of Large Language Models via Additive Quantization", arXiv, 2023. [[paper](http://arxiv.org/abs/2401.06118)]
- "ZeroQuant(4+2): Redefining LLMs Quantization with a New FP6-Centric Strategy for Diverse Generative Tasks", arXiv, 2023. [[paper](http://arxiv.org/abs/2312.08583)] [**`PTQ`**]
- "CBQ: Cross-Block Quantization for Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2312.07950)] [**`PTQ`**]
- "FP8-BERT: Post-Training Quantization for Transformer", arXiv, 2023. [[paper](http://arxiv.org/abs/2312.05725)] [**`PTQ`**]
- "Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge", arXiv, 2023. [[paper](http://arxiv.org/abs/2312.05693)]
- "SmoothQuant+: Accurate and Efficient 4-bit Post-Training WeightQuantization for LLM", arXiv, 2023. [[paper](http://arxiv.org/abs/2312.03788)] [**`PTQ`**]
- "A Speed Odyssey for Deployable Quantization of LLMs", arXiv, 2023. [[paper](http://arxiv.org/abs/2311.09550)]
- "AFPQ: Asymmetric Floating Point Quantization for LLMs", arXiv, 2023. [[paper](http://arxiv.org/abs/2311.01792)]
- "Enabling Fast 2-bit LLM on GPUs: Memory Alignment, Sparse Outlier, and Asynchronous Dequantization", arXiv, 2023. [[paper](http://arxiv.org/abs/2311.16442)]
- "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/71815)] [[code](https://github.com/artidoro/qlora)]
- "QuIP: 2-Bit Quantization of Large Language Models With Guarantees", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/69982)] [[code](https://github.com/jerry-chee/QuIP)] [**`PTQ`**]
- "Memory-Efficient Fine-Tuning of Compressed Large Language Models via sub-4-bit Integer Quantization", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/72931)]
- "SqueezeLLM: Dense-and-Sparse Quantization", arXiv, 2023. [[paper](https://arxiv.org/abs/2306.07629)] [**`PTQ`**] [**`Non-uniform`**]
- "QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources", arXiv, 2023. [[paper](https://arxiv.org/abs/2310.07147)]
- "PB-LLM: Partially Binarized Large Language Models", arXiv, 2023. [[paper](https://arxiv.org/abs/2310.00034)] [**`Extreme`**]
- "QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2310.16795)]
- "Atom: Low-bit Quantization for Efficient and Accurate LLM Serving", arXiv, 2023. [[paper](http://arxiv.org/abs/2310.19102)]
- "ZeroQuant-HERO: Hardware-Enhanced Robust Optimized Post-Training Quantization Framework for W8A8 Transformers", arXiv, 2023. [[paper](http://arxiv.org/abs/2310.17723)]
- "LLM-FP4: 4-Bit Floating-Point Quantized Transformers", arXiv, 2023. [[paper](https://arxiv.org/abs/2310.16836)]
- "TEQ: Trainable Equivalent Transformation for Quantization of LLMs", arXiv, 2023. [[paper](http://arxiv.org/abs/2310.10944)]
- "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2310.08659)]
- "QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2310.08041)] [**`PTQ`**]
- "Rethinking Channel Dimensions to Isolate Outliers for Low-bit Weight Quantization of Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.15531)]
- "QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.14717)]
- "Efficient Post-training Quantization with FP8 Formats", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.14592)]
- "Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantization", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.13575)]
- "Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.05516)]
- "Norm Tweaking: High-performance Low-bit Quantization of Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.02784)]
- "Understanding the Impact of Post-Training Quantization on Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.05210)]
- "QuantEase: Optimization-based Quantization for Language Models -- An Efficient and Intuitive Algorithm", arXiv, 2023. [[paper](http://arxiv.org/abs/2309.01885)]
- "FPTQ: Fine-grained Post-Training Quantization for Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2308.15987)]
- "OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2308.13137)]
- "FineQuant: Unlocking Efficiency with Fine-Grained Weight-Only Quantization for LLMs", arXiv, 2023. [[paper](http://arxiv.org/abs/2308.09723)] [**`PTQ`**]
- "Gradient-Based Post-Training Quantization: Challenging the Status Quo", arXiv, 2023. [[paper](http://arxiv.org/abs/2308.07662)] [**`PTQ`**]
- "NUPES : Non-Uniform Post-Training Quantization via Power Exponent Search", arXiv, 2023. [[paper](http://arxiv.org/abs/2308.05600)] [**`Non-uniform`**] 
- "ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats", arXiv, 2023. [[paper](http://arxiv.org/abs/2307.09782)]
- "Self-Distilled Quantization: Achieving High Compression Rates in Transformer-Based Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2307.05972)]
- "Do Emergent Abilities Exist in Quantized Large Language Models: An Empirical Study", arXiv, 2023. [[paper](https://arxiv.org/abs/2307.08072)]
- "INT2.1: Towards Fine-Tunable Quantized Large Language Models with Error Correction through Low-Rank Adaptation", arXiv, 2023. [[paper](https://arxiv.org/abs/2306.08162)]
- "QIGen: Generating Efficient Kernels for Quantized Inference on Large Language Models", arXiv, 2023. [[paper](https://arxiv.org/abs/2307.03738)] [[code](https://github.com/IST-DASLab/QIGen)]
- "OWQ: Lessons learned from activation outliers for weight quantization in large language models", arXiv, 2023. [[paper](http://arxiv.org/abs/2306.02272)] [**`PTQ`**]
- "PreQuant: A Task-agnostic Quantization Approach for Pre-trained Language Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2306.00014)]
- "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression", arXiv, 2023. [[paper](https://arxiv.org/abs/2306.03078)] [**`PTQ`**]
- "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", arXiv, 2023. [[paper](https://arxiv.org/abs/2306.00978)] [**`PTQ`**]
- "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models", arXiv, 2023. [[paper](https://arxiv.org/abs/2305.17888)]
- "Outlier Suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling", arXiv, 2023. [[paper](https://arxiv.org/abs/2304.09145)] [**`PTQ`**]
- "RPTQ: Reorder-based Post-training Quantization for Large Language Models", arXiv, 2023. [[paper](https://arxiv.org/abs/2304.01089)] [[code](https://github.com/hahnyuan/rptq4llm)] [**`PTQ`**]
- "The case for 4-bit precision: k-bit Inference Scaling Laws", ICML, 2023. [[paper](https://openreview.net/forum?id=i8tGb1ab1j)]
- "Quantized Distributed Training of Large Models with Convergence Guarantees", ICML, 2023. [[paper](https://openreview.net/forum?id=Nqp8A5IDzq)]
- "Understanding Int4 Quantization for Language Models: Latency Speedup, Composability, and Failure Cases", ICML, 2023. [[paper](https://openreview.net/forum?id=q1WGm3hItW)]
- "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML, 2023. [[paper](https://arxiv.org/abs/2211.10438)] [[code](https://github.com/mit-han-lab/smoothquant)] [**`PTQ`**]
- "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR, 2023. [[papar](https://arxiv.org/abs/2210.17323)]  [[code](https://github.com/IST-DASLab/gptq)] [**`PTQ`**]
- "LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models", arXiv, 2022. [[paper](https://arxiv.org/abs/2206.09557)] 
- "BiBERT: Accurate Fully Binarized BERT", ICLR, 2022. [[paper](https://openreview.net/forum?id=5xEgrl_5FAJ)] [[code](https://github.com/htqin/BiBERT)] [**`Extreme`**]
- "BiT: Robustly Binarized Multi-distilled Transformer", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=55032)] [[code](https://github.com/facebookresearch/bit)] [**`Extreme`**]
- "Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models", NeurIPS, 2022. [[paper]](https://arxiv.org/abs/2209.13325) [[code](https://github.com/wimh966/outlier_suppression)] [**`PTQ`**]
- "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", NeurIPS, 2022. [[paper](https://arxiv.org/abs/2208.07339)] [[code](https://github.com/timdettmers/bitsandbytes)]
- "Towards Efficient Post-training Quantization of Pre-trained Language Models", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=53407)] [**`PTQ`**]
- "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=54407)] [[code](https://github.com/microsoft/DeepSpeed)] [**`PTQ`**]
- "Compression of Generative Pre-trained Language Models via Quantization", ACL, 2022. [[paper](https://aclanthology.org/2022.acl-long.331)] 
- "I-BERT: Integer-only BERT Quantization", ICML, 2021. [[paper](https://proceedings.mlr.press/v139/kim21d.html)] [[code](https://github.com/kssteven418/I-BERT)]
- "BinaryBERT: Pushing the Limit of BERT Quantization", ACL, 2021. [[paper](https://arxiv.org/abs/2012.15701)] [[code](https://github.com/huawei-noah/Pretrained-Language-Model)] [**`Extreme`**]
- "On the Distribution, Sparsity, and Inference-time Quantization of Attention Values in Transformers", ACL, 2021. [[paper](https://aclanthology.org/2021.findings-acl.363)]
- "Understanding and Overcoming the Challenges of Efficient Transformer Quantization", EMNLP, 2021. [[paper](https://arxiv.org/abs/2109.12948)] [[code](https://github.com/qualcomm-ai-research/transformer-quantization)]
- "KDLSQ-BERT: A Quantized Bert Combining Knowledge Distillation with Learned Step Size Quantization", arXiv, 2021. [[paper](https://arxiv.org/abs/2101.05938)]
- "TernaryBERT: Distillation-aware Ultra-low Bit BERT", EMNLP, 2020. [[paper](https://arxiv.org/abs/2009.12812)] [[code](https://github.com/huawei-noah/Pretrained-Language-Model)] [**`Extreme`**]
- "Extremely Low Bit Transformer Quantization for On-Device Neural Machine Translation", EMNLP, 2020. [[paper](https://aclanthology.org/2020.findings-emnlp.433/)]
- "GOBO: Quantizing Attention-Based NLP Models for Low Latency and Energy Efficient Inference", MICRO, 2020. [[paper](https://arxiv.org/abs/2005.03842)]
- "Towards Fully 8-bit Integer Inference for the Transformer Model", IJCAI, 2020. [[paper](https://www.ijcai.org/Proceedings/2020/0520.pdf)] 
- "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT", AAAI, 2020. [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/6409/6265)]
- "Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model", ICML, 2019. [[paper](https://arxiv.org/abs/1906.00532)]
- "Q8BERT: Quantized 8Bit BERT", EMC2 Workshop, 2019. [[paper](https://www.emc2-ai.org/assets/docs/neurips-19/emc2-neurips19-paper-31.pdf)] 

[[Back to Overview](#overview)]

## Convolutional Neural Networks
### Visual Generation
- "QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning", arXiv, 2024. [[paper](http://arxiv.org/abs/2402.03666)]
- "Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2401.04585)]
- "Efficient Quantization Strategies for Latent Diffusion Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2312.05431)] [**`PTQ`**]
- "TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2311.16503)]
- "Post-training Quantization with Progressive Calibration and Activation Relaxing for Text-to-Image Diffusion Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2311.06322)]
- "Effective Quantization for Diffusion Models on CPUs", arXiv, 2023. [[paper](http://arxiv.org/abs/2311.16133)]
- "PTQD: Accurate Post-Training Quantization for Diffusion Models", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/71314)] [**`PTQ`**]
- "Q-DM: An Efficient Low-bit Quantized Diffusion Model", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/70279)]
- "Temporal Dynamic Quantization for Diffusion Models", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/72396)]
- "EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2310.03270)]
- "Q-diffusion: Quantizing Diffusion Models", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)] [[code](https://github.com/Xiuyu-Li/q-diffusion)] [**`PTQ`**]
- "Towards Accurate Data-free Quantization for Diffusion Models", arXiv, 2023. [[paper](http://arxiv.org/abs/2305.18723)] [**`PTQ`**]
- "Post-training Quantization on Diffusion Models", CVPR, 2023. [[paper](http://openaccess.thecvf.com/content/CVPR2023/html/Shang_Post-Training_Quantization_on_Diffusion_Models_CVPR_2023_paper.html)] [[code](https://https//github.com/42Shawn/PTQ4DM)] [**`PTQ`**]

### Image Classification
- "Post-Training Quantization with Low-precision Minifloats and Integers on FPGAs", arXiv, 2023. [[paper](https://arxiv.org/abs/2311.12359)] [**`PTQ`**]
- "Make RepVGG Greater Again: A Quantization-Aware Approach", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29045)]
- "MetaMix: Meta-State Precision Searcher for Mixed-Precision Activation Quantization", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29212)] [**`MP`**] 
- "Robustness-Guided Image Synthesis for Data-Free Quantization", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/28972)]
- "PTMQ: Post-training Multi-Bit Quantization of Neural Networks", AAAI, 2024. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/29553)] [**`PTQ`**]
- "Trainable Fixed-Point Quantization for Deep Learning Acceleration on FPGAs", arXiv, 2023. [[paper](http://arxiv.org/abs/2401.17544)]
- "StableQ: Enhancing Data-Scarce Quantization with Text-to-Image Data", arXiv, 2023. [[paper](http://arxiv.org/abs/2312.05272)]
- "Understanding Neural Network Binarization with Forward and Backward Proximal Quantizers", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/71526)] [**`Extreme`**]
- "TexQ: Zero-shot Network Quantization with Texture Feature Distribution Calibration", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/70325)]
- "Overcoming Forgetting Catastrophe in Quantization-Aware Training", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Overcoming_Forgetting_Catastrophe_in_Quantization-Aware_Training_ICCV_2023_paper.pdf)]
- "Causal-DFQ: Causality Guided Data-Free Network Quantization", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Shang_Causal-DFQ_Causality_Guided_Data-Free_Network_Quantization_ICCV_2023_paper.pdf)] [[code](https://github.com/42Shawn/Causal-DFQ)]
- "DenseShift: Towards Accurate and Efficient Low-Bit Power-of-Two Quantization", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_DenseShift_Towards_Accurate_and_Efficient_Low-Bit_Power-of-Two_Quantization_ICCV_2023_paper.pdf)]
- "EQ-Net: Elastic Quantization Neural Networks", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_EQ-Net_Elastic_Quantization_Neural_Networks_ICCV_2023_paper.pdf)] [[code](https://github.com/xuke225/EQ-Net)]
- "A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Colbert_A2Q_Accumulator-Aware_Quantization_with_Guaranteed_Overflow_Avoidance_ICCV_2023_paper.pdf)]
- "EMQ: Evolving Training-free Proxies for Automated Mixed Precision Quantization", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Dong_EMQ_Evolving_Training-free_Proxies_for_Automated_Mixed_Precision_Quantization_ICCV_2023_paper.pdf)] [**`MP`**]
- "Unified Data-Free Compression: Pruning and Quantization without Fine-Tuning", ICCV, 2023. [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Bai_Unified_Data-Free_Compression_Pruning_and_Quantization_without_Fine-Tuning_ICCV_2023_paper.pdf)] [**`PTQ`**]
- "Few-bit Backward: Quantized Gradients of Activation Functions for Memory Footprint Reduction", ICML, 2023. [[paper](https://openreview.net/forum?id=m2S96Qf2R3)] [[code](https://github.com/SkoltechAI/fewbit)]
- "FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization", ICML, 2023. [[paper](https://openreview.net/forum?id=EPnzNJTYsb)] [**`PTQ`**]
- "Data-Free Quantization via Mixed-Precision Compensation without Fine-Tuning", PR, 2023. [[paper](http://arxiv.org/abs/2307.00498)]
- "OMPQ: Orthogonal Mixed Precision Quantization", AAAI, 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26084)] [**`MP`**]
- "Rethinking Data-Free Quantization as a Zero-Sum Game", AAAI, 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26136)]
- "Quantized Feature Distillation for Network Quantization", AAAI, 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26354)]
- "Resilient Binary Neural Network", AAAI, 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26261)] [**`Extreme`**]
- "Fast and Accurate Binary Neural Networks Based on Depth-Width Reshaping", AAAI, 2023. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26268)] [**`Extreme`**]
- "Efficient Quantization-aware Training with Adaptive Coreset Selection", arXiv, 2023. [[paper](http://arxiv.org/abs/2306.07215)] 
- "One-Shot Model for Mixed-Precision Quantization", CVPR, 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Koryakovskiy_One-Shot_Model_for_Mixed-Precision_Quantization_CVPR_2023_paper.pdf)] [**`MP`**]
- "Adaptive Data-Free Quantization", CVPR, 2023. [[paper](https://arxiv.org/abs/2303.06869)] 
- "Bit-shrinking: Limiting Instantaneous Sharpness for Improving Post-training Quantization", CVPR, 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Bit-Shrinking_Limiting_Instantaneous_Sharpness_for_Improving_Post-Training_Quantization_CVPR_2023_paper.pdf)] [**`PTQ`**]
- "Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective", CVPR, 2023. [[paper](https://arxiv.org/pdf/2303.11906.pdf)] [[code](https://github.com/bytedance/mrecg)] [**`PTQ`**]
- "GENIE: Show Me the Data for Quantization", CVPR, 2023. [[paper](https://arxiv.org/abs/2212.04780)] [[code](https://github.com/SamsungLabs/Genie)] [**`PTQ`**]
- "Bayesian asymmetric quantized neural networks", PR, 2023. [[paper](https://www.sciencedirect.com/science/article/pii/S0031320323001632)]
- "Distribution-sensitive Information Retention for Accurate Binary Neural Network", IJCV, 2023. [[paper](https://arxiv.org/abs/2109.12338)] [**`Extreme`**]
- "SDQ: Stochastic Differentiable Quantization with Mixed Precision", ICML, 2022. [[paper](https://proceedings.mlr.press/v162/huang22h.html)] [**`MP`**]
- "Finding the Task-Optimal Low-Bit Sub-Distribution in Deep Neural Networks", ICML, 2022. [[paper](https://proceedings.mlr.press/v162/dong22a.html)] [[code](https://github.com/RunpeiDong/DGMS)]
- "GACT: Activation Compressed Training for Generic Network Architectures", ICML, 2022. [[paper](https://proceedings.mlr.press/v162/liu22v.html)] [[code](https://github.com/LiuXiaoxuanPKU/GACT-ICML)]
- "Overcoming Oscillations in Quantization-Aware Training", ICML, 2022. [[paper](https://proceedings.mlr.press/v162/nagel22a/nagel22a.pdf)] [[code](https://github.com/qualcomm-ai-research/oscillations-qat)]
- "Nonuniform-to-Uniform Quantization: Towards Accurate Quantization via Generalized Straight-Through Estimation", CVPR, 2022. [[paper](https://arxiv.org/abs/2111.14826)] [[code](https://github.com/liuzechun/Nonuniform-to-Uniform-Quantization)] [**`Non-uniform`**]
- "Learnable Lookup Table for Neural Network Quantization", CVPR, 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.pdf)] [[code](https://github.com/The-Learning-And-Vision-Atelier-LAVA/LLT)] [**`Non-uniform`**]
- "Mr.BiQ: Post-Training Non-Uniform Quantization based on Minimizing the Reconstruction Error", CVPR, 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Jeon_Mr.BiQ_Post-Training_Non-Uniform_Quantization_Based_on_Minimizing_the_Reconstruction_Error_CVPR_2022_paper.pdf)] [**`PTQ`**] [**`Non-uniform`**]
- "Data-Free Network Compression via Parametric Non-uniform Mixed Precision Quantization", CVPR, 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chikin_Data-Free_Network_Compression_via_Parametric_Non-Uniform_Mixed_Precision_Quantization_CVPR_2022_paper.pdf)] [**`Non-uniform`**] [**`MP`**]
- "IntraQ: Learning Synthetic Images With Intra-Class Heterogeneity for Zero-Shot Network Quantization", CVPR, 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_IntraQ_Learning_Synthetic_Images_With_Intra-Class_Heterogeneity_for_Zero-Shot_Network_CVPR_2022_paper.html)] [[code](https://github.com/zysxmu/IntraQ)]
- "Instance-Aware Dynamic Neural Network Quantization", CVPR, 2022. [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Instance-Aware_Dynamic_Neural_Network_Quantization_CVPR_2022_paper.pdf)]
- "Leveraging Inter-Layer Dependency for Post-Training Quantization", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=54389)]  [**`PTQ`**]
- "Theoretically Better and Numerically Faster Distributed Optimization with Smoothness-Aware Quantization Techniques", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=53476)]
- "Entropy-Driven Mixed-Precision Quantization for Deep Network Design",  NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=54104)] [**`MP`**]
- "Redistribution of Weights and Activations for AdderNet Quantization", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=54812)]
- "FP8 Quantization: The Power of the Exponent", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=53073)] [[code](https://github.com/qualcomm-ai-research/fp8-quantization)]
- "Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=53412)] [[code](https://github.com/ist-daslab/obc)]  [**`PTQ`**]
- "ClimbQ: Class Imbalanced Quantization Enabling Robustness on Efficient Inferences", NeurIPS, 2022. [[paper](https://nips.cc/Conferences/2022/Schedule?showEvent=55162)]
- "Non-Uniform Step Size Quantization for Accurate Post-Training Quantization", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710657.pdf)]  [**`PTQ`**] [**`Non-uniform`**]
- "Towards Accurate Network Quantization with Equivalent Smooth Regularizer", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710726.pdf)] 
- "BASQ: Branch-wise Activation-clipping Search Quantization for Sub-4-bit Neural Networks", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720017.pdf)] [[code](https://github.com/HanByulKim/BASQ)]
- "RDO-Q: Extremely Fine-Grained Channel-Wise Quantization via Rate-Distortion Optimization", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720156.pdf)]
- "Mixed-Precision Neural Network Quantization via Learned Layer-Wise Importance", ECCV, 2022. [[paper](https://arxiv.org/abs/2203.08368)] [[Code](https://github.com/1hunters/LIMPQ)] [[code](https://github.com/1hunters/LIMPQ)] [**`MP`**]
- "Symmetry Regularization and Saturating Nonlinearity for Robust Quantization", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710207.pdf)]
- "RAPQ: Rescuing Accuracy for Power-of-Two Low-bit Post-training Quantization", IJCAI, 2022. [[paper](https://www.ijcai.org/proceedings/2022/219)] [[code](https://github.com/billamihom/rapq)]  [**`PTQ`**]
- "MultiQuant: Training Once for Multi-bit Quantization of Neural Networks", IJCAI, 2022. [[paper](https://www.ijcai.org/proceedings/2022/504)]
- "F8Net: Fixed-Point 8-bit Only Multiplication for Network Quantization", ICLR, 2022. [[paper](https://openreview.net/forum?id=_CfpJazzXT2)] 
- "8-bit Optimizers via Block-wise Quantization", ICLR, 2022. [[paper](https://openreview.net/forum?id=shpkpVXzo3h)] [[code](https://github.com/facebookresearch/bitsandbytes)]
- "Information Bottleneck: Exact Analysis of (Quantized) Neural Networks", ICLR, 2022. [[paper](https://openreview.net/forum?id=kF9DZQQrU0w)] [[code](https://github.com/StephanLorenzen/ExactIBAnalysisInQNNs)]
- "QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization", ICLR, 2022. [[paper](https://openreview.net/forum?id=ySQH0oDyp7)] [[code](https://github.com/wimh966/QDrop)]  [**`PTQ`**]
- "SQuant: On-the-Fly Data-Free Quantization via Diagonal Hessian Approximation", ICLR, 2022. [[paper](https://openreview.net/forum?id=JXhROKNZzOc)] [[code](https://github.com/clevercool/SQuant)]  [**`PTQ`**]
- "FILM-QNN: Efficient FPGA Acceleration of Deep Neural Networks with Intra-Layer, Mixed-Precision Quantization", FPGA, 2022. [[paper](https://dl.acm.org/doi/abs/10.1145/3490422.3502364)] [**`MP`**]
- "Accurate Post Training Quantization with Small Calibration Sets", ICML, 2021. [[paper](http://proceedings.mlr.press/v139/hubara21a.html)] [[code](https://github.com/papers-submission/CalibTIP)]  [**`PTQ`**]
- "How Do Adam and Training Strategies Help BNNs Optimization?", ICML, 2021. [[paper](http://proceedings.mlr.press/v139/liu21t/liu21t.pdf)] [[code](https://github.com/liuzechun/AdamBNN)]
- "ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training", ICML, 2021. [[paper](https://proceedings.mlr.press/v139/chen21z.html)] [[code](https://github.com/ucbrise/actnn)]
- "HAWQ-V3: Dyadic Neural Network Quantization", ICML, 2021. [[paper](https://proceedings.mlr.press/v139/yao21a.html)] [[code](https://github.com/Zhen-Dong/HAWQ)] [**`MP`**]
- "Differentiable Dynamic Quantization with Mixed Precision and Adaptive Resolution", ICML, 2021. [[paper](https://proceedings.mlr.press/v139/zhang21r.html)] [**`MP`**]
- "Auto-NBA: Efficient and Effective Search Over the Joint Space of Networks, Bitwidths, and Accelerators", ICML, 2021. [[paper](https://proceedings.mlr.press/v139/fu21d.html)] [[code](https://github.com/RICE-EIC/Auto-NBA)]
- "Qimera: Data-free Quantization with Synthetic Boundary Supporting Samples", NeurIPS, 2021. [[paper](https://openreview.net/forum?id=ejo1_Weiart)] [[code](https://github.com/iamkanghyunchoi/qimera)]
- "Post-Training Sparsity-Aware Quantization", NeurIPS, 2021. [[paper](https://openreview.net/forum?id=qe9z54E_cqE)] [[code](https://github.com/gilshm/sparq)]  [**`PTQ`**]
- "Diversifying Sample Generation for Accurate Data-Free Quantization", CVPR, 2021. [[paper](https://arxiv.org/abs/2103.01049)]  [**`PTQ`**]
- "Permute, Quantize, and Fine-tune: Efficient Compression of Neural Networks.", CVPR, 2021. [[paper](https://arxiv.org/abs/2010.15703)] [[code](https://github.com/uber-research/permute-quantize-finetune)]
- "Learnable Companding Quantization for Accurate Low-bit Neural Networks", CVPR, 2021. [[paper](https://arxiv.org/abs/2103.07156)]
- "Zero-shot Adversarial Quantization", CVPR, 2021. [[paper](https://arxiv.org/abs/2103.15263)] [[code](https://github.com/FLHonker/ZAQ-code)]
- "Network Quantization with Element-wise Gradient Scaling", CVPR, 2021. [[paper](https://arxiv.org/abs/2104.00903)] [[code](https://github.com/cvlab-yonsei/EWGS)]
- "High-Capacity Expert Binary Networks", ICLR, 2021. [[paper](https://openreview.net/forum?id=MxaY4FzOTa)] [[code](https://github.com/1adrianb/expert-binary-networks)] [**`Extreme`**]
- "Multi-Prize Lottery Ticket Hypothesis: Finding Accurate Binary Neural Networks by Pruning A Randomly Weighted Network", ICLR, 2021. [[paper](https://openreview.net/forum?id=U_mat0b9iv)] [[code](https://github.com/chrundle/biprop)] [**`Extreme`**]
- "BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction", ICLR, 2021. [[paper](https://openreview.net/forum?id=POWv6hDd9XH)] [[code](https://github.com/yhhhli/BRECQ)]  [**`PTQ`**]
- "Neural gradients are near-lognormal: improved quantized and sparse training", ICLR, 2021. [[paper](https://openreview.net/forum?id=EoFNy62JGd)] 
- "Training with Quantization Noise for Extreme Model Compression", ICLR, 2021. [[paper](https://openreview.net/forum?id=dV19Yyi1fS3)]
- "BSQ: Exploring Bit-Level Sparsity for Mixed-Precision Neural Network Quantization", ICLR, 2021. [[paper](https://openreview.net/forum?id=TiXl51SCNw8)] [[code](https://github.com/yanghr/BSQ)] [**`MP`**]
- "Simple Augmentation Goes a Long Way: ADRL for DNN Quantization", ICLR, 2021. [[paper](https://openreview.net/forum?id=Qr0aRliE_Hb)]
- "Distribution Adaptive INT8 Quantization for Training CNNs", AAAI, 2021. [[paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj4-rjuq7nvAhUVPH0KHXlYCUQQFjAFegQIChAD&url=https%3A%2F%2Fwww.aaai.org%2FAAAI21Papers%2FAAAI-7144.ZhaoK.pdf&usg=AOvVaw3dnOXfzKkLIw_qWXj7p7Yc)]
- "Stochastic Precision Ensemble: Self‐Knowledge Distillation for Quantized Deep Neural Networks", AAAI, 2021. [[paper](https://arxiv.org/abs/2009.14502)]
- "Optimizing Information Theory Based Bitwise Bottlenecks for Efficient Mixed-Precision Activation Quantization", AAAI, 2021. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16474/16281)] [**`MP`**]
- "OPQ: Compressing Deep Neural Networks with One-shot Pruning-Quantization", AAAI, 2021. [[paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjD6aPrqbnvAhXeIDQIHWNdDCUQFjADegQIAxAD&url=https%3A%2F%2Fwww.aaai.org%2FAAAI21Papers%2FAAAI-1054.HuP.pdf&usg=AOvVaw2R_BcDlKyuuAPHMeO0Q-1c)] 
- "Scalable Verification of Quantized Neural Networks", AAAI, 2021. [[paper](https://arxiv.org/pdf/2012.08185)] [[code](https://github.com/mlech26l/qnn_robustness_benchmarks)]
- "Uncertainty Quantification in CNN through the Bootstrap of Convex Neural Networks", AAAI, 2021. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/17434/17241)]
- "FracBits: Mixed Precision Quantization via Fractional Bit-Widths", AAAI, 2021. [[paper](https://www.semanticscholar.org/paper/FracBits%3A-Mixed-Precision-Quantization-via-Yang-Jin/cb219432863778fa173925d51fbf02af1d17ad98)] [**`MP`**]
- "Post-training Quantization with Multiple Points: Mixed Precision without Mixed Precision", AAAI, 2021. [[paper](https://arxiv.org/pdf/2002.09049)]  [**`PTQ`**] [**`MP`**]
- "ZeroQ: A Novel Zero Shot Quantization Framework", CVPR, 2020. [[paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Cai_ZeroQ_A_Novel_Zero_Shot_Quantization_Framework_CVPR_2020_paper.html)] [[code](https://github.com/amirgholami/ZeroQ)]  [**`PTQ`**]
- "LSQ+: Improving Low-bit Quantization Through Learnable Offsets and Better Initialization", CVPR, 2020. [[paper](http://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Bhalgat_LSQ_Improving_Low-Bit_Quantization_Through_Learnable_Offsets_and_Better_Initialization_CVPRW_2020_paper.html)]
- "HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks", NeurIPS, 2020. [[paper](https://proceedings.neurips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html)] [**`MP`**]
- "Learned step size quantization", ICLR, 2020. [[paper](https://openreview.net/forum?id=rkgO66VKDS)]
- "HAWQ: Hessian AWare Quantization of Neural Networks With Mixed-Precision", ICCV, 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html)] [**`MP`**]
- "Data-Free Quantization Through Weight Equalization and Bias Correction", ICCV, 2019. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.html)]  [**`PTQ`**]
- "HAQ: Hardware-Aware Automated Quantization with Mixed Precision", CVPR, 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf)] [[code](https://github.com/mit-han-lab/haq)] [**`MP`**]
- "PACT: Parameterized Clipping Activation for Quantized Neural Networks", arXiv, 2018. [[paper](https://arxiv.org/abs/1805.06085)]
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR, 2018. [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf)]


[[Back to Overview](#overview)]

### Other Tasks
#### Object Detection
- "Improving Post-Training Quantization on Object Detection with Task Loss-Guided Lp Metric", arXiv, 2023. [[paper](https://arxiv.org/abs/2304.09785)]  [**`PTQ`**]
- "AQD: Towards Accurate Quantized Object Detection", CVPR, 2021. [[paper](http://arxiv.org/abs/2007.06919)]
- "BiDet: An Efficient Binarized Object Detector", CVPR, 2020. [[paper](https://arxiv.org/abs/2003.03961)] [[code](https://github.com/ZiweiWangTHU/BiDet)] [**`Extreme`**]
- "Fully Quantized Network for Object Detection", CVPR, 2019. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf)]


#### Super Resolution
- "QuantSR: Accurate Low-bit Quantization for Efficient Image Super-Resolution", NeurIPS, 2023. [[paper](https://neurips.cc/virtual/2023/poster/72890)]
- "Toward Accurate Post-Training Quantization for Image Super Resolution", CVPR, 2023. [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.pdf)] [[code]( https://github.com/huawei-noah/Efficient-Computing/tree/master/Quantization/PTQ4SR)]  [**`PTQ`**]
- "EBSR: Enhanced Binary Neural Network for Image Super-Resolution", arXiv, 2023. [[paper](https://arxiv.org/abs/2303.12270)] [**`Extreme`**]
- "CADyQ: Content-Aware Dynamic Quantization for Image Super-Resolution
", ECCV, 2022. [[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670360.pdf)] [[code](https://github.com/cheeun/cadyq)]
- "Dynamic Dual Trainable Bounds for Ultra-low Precision Super-Resolution Networks", ECCV, 2022. [[paper](https://arxiv.org/abs/2203.03844)] [[code](https://github.com/zysxmu/ddtb)]
- "DAQ: Channel-Wise Distribution-Aware Quantization for Deep Image Super-Resolution Networks", WACV, 2022. [[paper](http://openaccess.thecvf.com/content/WACV2022/html/Hong_DAQ_Channel-Wise_Distribution-Aware_Quantization_for_Deep_Image_Super-Resolution_Networks_WACV_2022_paper.html)] [[code](https://github.com/Cheeun/DAQ-pytorch)]
- "Fully Quantized Image Super-Resolution Networks", ACM MM, 2021. [[paper](https://arxiv.org/abs/2011.14265)] [[code](https://github.com/billhhh/FQSR)]
- "PAMS: Quantized Super-Resolution via Parameterized Max Scale", ECCV, 2020. [[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700562.pdf)] [[code](https://github.com/colorjam/PAMS)]
- "Training Binary Neural Network without Batch Normalization for Image Super-Resolution", AAAI, 2021. [[paper](https://ojs.aaai.org/index.php/AAAI/article/download/16263/16070)] [**`Extreme`**]

#### Point Cloud
- "LiDAR-PTQ: Post-Training Quantization for Point Cloud 3D Object Detection", arXiv, 2023. [[paper](http://arxiv.org/abs/2401.15865)]  [**`PTQ`**]
- "Binarizing Sparse Convolutional Networks for Efficient Point Cloud Analysis", arXiv, 2023. [[paper](https://arxiv.org/abs/2303.15493)] [**`Extreme`**]
- "BiPointNet: Binary Neural Network for Point Clouds", ICLR, 2021. [[paper](https://openreview.net/forum?id=9QLRCVysdlO)]  [[code](https://github.com/htqin/BiPointNet)] [**`Extreme`**]

[[Back to Overview](#overview)]



---

## References
* Online Resources:
    * [MQBench (Benchmark)](http://mqbench.tech/)
    * [Awesome Model Quantization (GitHub)](https://github.com/htqin/awesome-model-quantization)
    * [Awesome Transformer Attention (GitHub)](https://github.com/cmhungsteve/Awesome-Transformer-Attention)


