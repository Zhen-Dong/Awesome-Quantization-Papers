# Awesome Quantization Papers [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
This repo collects and categorizes quantization-related papers on AI conferences/journals/arXiv.

Currently the joint table is in progress, and contributions are more than welcomed.

## A Survey on Quantization
Our [quantization survey paper](https://arxiv.org/abs/2103.13630) is included in the Book [Low-Power Computer Vision: Improve the Efficiency of Artificial Intelligence](https://www.routledge.com/Low-Power-Computer-Vision-Improve-the-Efficiency-of-Artificial-Intelligence/Thiruvathukal-Lu-Kim-Chen-Chen/p/book/9780367744700).

## Sort by Categories
Please search using the [cvs file](https://github.com/Zhen-Dong/Awesome-Quantization-Papers/blob/main/awesome_quantization_papers/awesome_quantization_papers.csv).

## Joint Table
**Bitwidth Settings:** **MP**&#8594; Mixed-Precision, **Uni**&#8594; Uniform Quantization, **T**&#8594; Ternarization, **B**&#8594; Binarization

**Quantizer Type:** **Linear**&#8594; Linear Quantizer, **Log**&#8594; Logarithmic Non-linear Quantizer, **OptN**&#8594; Optimization-based Non-linear Quantizer, **K**&#8594; K-means based Quantizer, **PQ**&#8594; Vector/Product Quantizer, **LQ**&#8594; Learnable Quantizer

**Finetuning Method:** **QAT**&#8594; Quantization-Aware Training, **PTQ**&#8594; Post-Training Quantization

**Task:** **C**&#8594; Image Classification, **O**&#8594; Object Detection, **S**&#8594; Semantic Segmentation, **N**&#8594; Natural Language Processing

**Special:** **Factor**&#8594; Factorization, **Distl**&#8594; Distillation

<table width="90px" style="table-layout:fixed; overflow-x: hidden; width:90px;">
<thead>
<tr>
<th>Title</th>
<th>Publication</th>
<th>Bit</th>
<th>Quantizer</th>
<th>Finetune</th>
<th>Task</th>
<th>Special</th>
</tr>
</thead>
<tbody>
<tr>
<th>Fully integer-based quantization for mobile convolutional neural network inference</th>
<th><a href="https://www.sciencedirect.com/science/article/pii/S0925231220319354">Neurocomputing 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fully Quantized Image Super-Resolution Networks</th>
<th><a href="https://dl.acm.org/doi/abs/10.1145/3474085.3475227">MM 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>VQMG: Hierarchical Vector Quantised and Multi-hops Graph Reasoning for Explicit Representation Learning</th>
<th><a href="https://dl.acm.org/doi/abs/10.1145/3474085.3475224">MM 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HAO: Hardware-aware neural Architecture Optimization for Efficient Inference</th>
<th><a href="https://arxiv.org/abs/2104.12766">FCCM 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Survey of Quantization Methods for Efficient Neural Network Inference:fire:47</th>
<th><a href="https://arxiv.org/abs/2103.13630">BLPCV 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>S^3: Sign-Sparse-Shift Reparametrization for Effective Training of Low-bit Shift Networks</th>
<th><a href="https://arxiv.org/pdf/2107.03453.pdf">NeurIPS 2021</a></th>
<th>T</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BatchQuant: Quantized-for-all Architecture Search with Robust Quantizer</th>
<th><a href="https://arxiv.org/pdf/2105.08952.pdf">NeurIPS 2021</a></th>
<th>MP</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>CBP: backpropagation with constraint on weight precision using a pseudo-Lagrange multiplier method<a href="https://github.com/dooseokjeong/CBP">[PyTorch]</a></th>
<th><a href="https://arxiv.org/pdf/2110.02550.pdf">NeurIPS 2021</a></th>
<th>B/T/Uni</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Divergence Frontiers for Generative Models: Sample Complexity, Quantization Effects, and Frontier Integrals</th>
<th><a href="https://arxiv.org/pdf/2106.07898.pdf">NeurIPS 2021</a></th>
<th>B/Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning Frequency Domain Approximation for Binary Neural Networks</th>
<th><a href="https://arxiv.org/pdf/2103.00841.pdf">NeurIPS 2021</a></th>
<th>B</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Post-Training Quantization for Vision Transformer</th>
<th><a href="https://arxiv.org/pdf/2106.14156.pdf">NeurIPS 2021</a></th>
<th>Uni</th>
<th></th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Post-Training Sparsity-Aware Quantization<a href="https://github.com/gilshm/sparq">[PyTorch]</a>:star:5</th>
<th><a href="https://arxiv.org/pdf/2105.11010.pdf">NeurIPS 2021</a></th>
<th>Uni</th>
<th></th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Qimera: Data-free Quantization with Synthetic Boundary Supporting Samples<a href="https://github.com/iamkanghyunchoi/qimera?ref=pythonawesome.com">[PyTorch]</a>:star:2</th>
<th><a href="https://arxiv.org/pdf/2111.02625.pdf">NeurIPS 2021</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes<a href="https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation">[PyTorch]</a>:star:2</th>
<th><a href="https://arxiv.org/pdf/2110.13541.pdf">NeurIPS 2021</a></th>
<th>Uni</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>QuPeD: Quantized Personalization via Distillation with Applications to Federated Learning</th>
<th><a href="https://arxiv.org/pdf/2107.13892.pdf">NeurIPS 2021</a></th>
<th>B/Uni</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>VQ-GNN: A Universal Framework to Scale up Graph Neural Networks using Vector Quantization</th>
<th><a href="https://arxiv.org/pdf/2110.14363.pdf">NeurIPS 2021</a></th>
<th>MP</th>
<th>PQ</th>
<th>QAT</th>
<th>Node Classification</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Beyond Preserved Accuracy: Evaluating Loyalty and Robustness of BERT Compression</th>
<th><a href="https://arxiv.org/pdf/2109.03228.pdf">EMNLP 2021</a></th>
<th>Uni</th>
<th></th>
<th>QAT/PTQ</th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Compressing Word Embeddings via Deep Compositional Code Learning:fire:89<a href="https://github.com/zomux/neuralcompressor">[PyTorch]</a>:star:81</th>
<th><a href="https://openreview.net/forum?id=BJRZzFlRb">EMNLP 2021</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Matching-oriented Embedding Quantization For Ad-hoc Retrieval<a href="https://github.com/microsoft/MoPQ">[PyTorch]</a></th>
<th><a href="https://arxiv.org/abs/2104.07858">EMNLP 2021</a></th>
<th>MP</th>
<th>PQ</th>
<th>QAT</th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Understanding and Overcoming the Challenges of Efficient Transformer Quantization<a href="https://github.com/qualcomm-ai-research/transformer-quantization">[PyTorch]</a>:star:9</th>
<th><a href="https://arxiv.org/abs/2109.12948">EMNLP 2021</a></th>
<th>Uni</th>
<th></th>
<th>QAT/PTQ</th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Cluster-Promoting Quantization With Bit-Drop for Minimizing Network Quantization Loss</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Cluster-Promoting_Quantization_With_Bit-Drop_for_Minimizing_Network_Quantization_Loss_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>T/Uni</th>
<th>LQ</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Distance-Aware Quantization</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Distance-Aware_Quantization_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>B/T/Uni</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Dynamic Network Quantization for Efficient Video Inference</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Dynamic_Network_Quantization_for_Efficient_Video_Inference_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>Video Recognition</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Generalizable Mixed-Precision Quantization via Attribution Rank Preservation<a href="https://github.com/ZiweiWangTHU/GMPQ">[PyTorch]</a>:star:15</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Generalizable_Mixed-Precision_Quantization_via_Attribution_Rank_Preservation_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>MP</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Improving Low-Precision Network Quantization via Bin Regularization</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Han_Improving_Low-Precision_Network_Quantization_via_Bin_Regularization_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>B/T/Uni</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Improving Neural Network Efficiency via Post-Training Quantization With Adaptive Floating-Point<a href="https://github.com/MXHX7199/ICCV_2021_AFP">[PyTorch]</a></th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Improving_Neural_Network_Efficiency_via_Post-Training_Quantization_With_Adaptive_Floating-Point_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>MP</th>
<th>Linear</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Integer-Arithmetic-Only Certified Robustness for Quantized Neural Networks</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Integer-Arithmetic-Only_Certified_Robustness_for_Quantized_Neural_Networks_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>MixMix: All You Need for Data-Free Compression Are Feature and Data Mixing</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Li_MixMix_All_You_Need_for_Data-Free_Compression_Are_Feature_and_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT/PTQ</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Once Quantization-Aware Training: High Performance Extremely Low-Bit Architecture Search<a href="https://github.com/LaVieEnRoseSMZ/OQA">[PyTorch]</a>:star:19</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Shen_Once_Quantization-Aware_Training_High_Performance_Extremely_Low-Bit_Architecture_Search_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>Uni</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Product Quantizer Aware Inverted Index for Scalable Nearest Neighbor Search</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Noh_Product_Quantizer_Aware_Inverted_Index_for_Scalable_Nearest_Neighbor_Search_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>MP</th>
<th>PQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>RMSMP: A Novel Deep Neural Network Quantization Framework With Row-Wise Mixed Schemes and Multiple Precisions</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Chang_RMSMP_A_Novel_Deep_Neural_Network_Quantization_Framework_With_Row-Wise_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>MP</th>
<th></th>
<th>QAT/PTQ</th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>ReCU: Reviving the Dead Weights in Binary Neural Networks</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_ReCU_Reviving_the_Dead_Weights_in_Binary_Neural_Networks_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Self-Supervised Product Quantization for Deep Unsupervised Image Retrieval<a href="https://github.com/youngkyunJang/SPQ">[PyTorch]</a>:star:22</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Jang_Self-Supervised_Product_Quantization_for_Deep_Unsupervised_Image_Retrieval_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>Uni</th>
<th>PQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Sub-bit Neural Networks: Learning to Compress and Accelerate Binary Neural Networks<a href="https://github.com/yikaiw/SNN">[PyTorch]</a>:star:7</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Sub-Bit_Neural_Networks_Learning_To_Compress_and_Accelerate_Binary_Neural_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Mixed-Precision Quantization of Neural Networks via Constrained Optimization</th>
<th><a href="https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Towards_Mixed-Precision_Quantization_of_Neural_Networks_via_Constrained_Optimization_ICCV_2021_paper.pdf">ICCV 2021</a></th>
<th>MP</th>
<th>Linear</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>1-bit Adam: Communication Efficient Large-Scale Training with Adam’s Convergence Speed</th>
<th><a href="http://proceedings.mlr.press/v139/tang21a/tang21a.pdf">ICML 2021</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Accurate Post Training Quantization With Small Calibration Sets<a href="https://github.com/papers-submission/CalibTIP">[PyTorch]</a>:star:14</th>
<th><a href="http://proceedings.mlr.press/v139/hubara21a/hubara21a.pdf">ICML 2021</a></th>
<th>MP</th>
<th></th>
<th>PTQ</th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training</th>
<th><a href="http://proceedings.mlr.press/v139/chen21z/chen21z.pdf">ICML 2021</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Communication-Efficient Distributed Optimization with Quantized Preconditioners</th>
<th><a href="http://proceedings.mlr.press/v139/alimisis21a/alimisis21a.pdf">ICML 2021</a></th>
<th></th>
<th>PQ</th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Differentiable Dynamic Quantization with Mixed Precision and Adaptive Resolution</th>
<th><a href="http://proceedings.mlr.press/v139/zhang21r/zhang21r.pdf">ICML 2021</a></th>
<th>MP</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Double-Win Quant: Aggressively Winning Robustness of Quantized Deep Neural Networks via Random Precision Training and Inference<a href="https://github.com/RICE-EIC/Double-Win-Quant">[PyTorch]</a>:star:3</th>
<th><a href="http://proceedings.mlr.press/v139/fu21c/fu21c.pdf">ICML 2021</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Estimation and Quantization of Expected Persistence Diagrams</th>
<th><a href="http://proceedings.mlr.press/v139/divol21a/divol21a.pdf">ICML 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HAWQ-V3: Dyadic Neural Network Quantization<a href="https://github.com/Zhen-Dong/HAWQ">[PyTorch]</a>:star:193</th>
<th><a href="https://arxiv.org/abs/2011.10680">ICML 2021</a></th>
<th>MP</th>
<th>Linear</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>I-BERT: Integer-only BERT Quantization</th>
<th><a href="http://proceedings.mlr.press/v139/kim21d/kim21d.pdf">ICML 2021</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization Algorithms for Random Fourier Features</th>
<th><a href="http://proceedings.mlr.press/v139/li21i/li21i.pdf">ICML 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Soft then Hard: Rethinking the Quantization in Neural Image Compression</th>
<th><a href="http://proceedings.mlr.press/v139/guo21c/guo21c.pdf">ICML 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Quantized Neural Networks to Global Optimality via Semidefinite Programming</th>
<th><a href="http://proceedings.mlr.press/v139/bartan21a/bartan21a.pdf">ICML 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Vector Quantized Models for Planning</th>
<th><a href="http://proceedings.mlr.press/v139/ozair21a/ozair21a.pdf">ICML 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Distribution-aware Adaptive Multi-bit Quantization</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Distribution-Aware_Adaptive_Multi-Bit_Quantization_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Layer importance estimation with imprinting for neural network quantization</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Liu_Layer_Importance_Estimation_With_Imprinting_for_Neural_Network_Quantization_CVPRW_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Adaptive binary-ternary quantization</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021W/BiVision/papers/Razani_Adaptive_Binary-Ternary_Quantization_CVPRW_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>AQD: Towards Accurate Quantized Object Detection<a href="https://github.com/aim-uofa/model-quantization">[PyTorch]</a>:star:11</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_AQD_Towards_Accurate_Quantized_Object_Detection_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Automated Log-Scale Quantization for Low-Cost Deep Neural Networks</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/html/Oh_Automated_Log-Scale_Quantization_for_Low-Cost_Deep_Neural_Networks_CVPR_2021_paper.html">CVPR 2021</a></th>
<th>T</th>
<th>Log</th>
<th>QAT</th>
<th>C/S</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binary TTC: A Temporal Geofence for Autonomous Navigation</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Badki_Binary_TTC_A_Temporal_Geofence_for_Autonomous_Navigation_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Diversifying Sample Generation for Accurate Data-Free Quantization</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Diversifying_Sample_Generation_for_Accurate_Data-Free_Quantization_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Generative Zero-shot Network Quantization</th>
<th><a href="https://arxiv.org/abs/2101.08430">CVPR 2021</a></th>
<th>Uni</th>
<th>OptN</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Improving Accuracy of Binary Neural Networks using Unbalanced Activation Distribution</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Kim_Improving_Accuracy_of_Binary_Neural_Networks_Using_Unbalanced_Activation_Distribution_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Is In-Domain Data Really Needed? A Pilot Study on Cross-Domain Calibration for Network Quantization</th>
<th><a href="https://arxiv.org/abs/2105.07331">CVPR 2021</a></th>
<th>Uni</th>
<th>OptN</th>
<th>PTQ</th>
<th>C/O/S	</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learnable Companding Quantization for Accurate Low-Bit Neural Networks</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Yamamoto_Learnable_Companding_Quantization_for_Accurate_Low-Bit_Neural_Networks_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Network Quantization With Element-Wise Gradient Scaling</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Network_Quantization_With_Element-Wise_Gradient_Scaling_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Optimal Quantization Using Scaled Codebook</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Idelbayev_Optimal_Quantization_Using_Scaled_Codebook_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Permute, Quantize, and Fine-tune: Efficient Compression of Neural Networks<a href="https://github.com/uber-research/permute-quantize-finetune">[PyTorch]</a>:star:109</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Martinez_Permute_Quantize_and_Fine-Tune_Efficient_Compression_of_Neural_Networks_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>QPP: Real-Time Quantization Parameter Prediction for Deep Neural Networks</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/html/Kryzhanovskiy_QPP_Real-Time_Quantization_Parameter_Prediction_for_Deep_Neural_Networks_CVPR_2021_paper.html">CVPR 2021</a></th>
<th>Uni</th>
<th>LQ</th>
<th>PTQ</th>
<th>C/O/S</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Zero-shot Adversarial Quantization</th>
<th><a href="https://arxiv.org/abs/2103.15263">CVPR 2021</a></th>
<th>Uni</th>
<th>Linear</th>
<th>PTQ</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>iVPF: Numerical Invertible Volume Preserving Flow for Efficient Lossless Compression</th>
<th><a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_iVPF_Numerical_Invertible_Volume_Preserving_Flow_for_Efficient_Lossless_Compression_CVPR_2021_paper.pdf">CVPR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Bipointnet: Binary neural network for point clouds</th>
<th><a href="https://openreview.net/forum?id=9QLRCVysdlO">ICLR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Sparse quantized spectral clustering</th>
<th><a href="https://arxiv.org/abs/2010.01376">ICLR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Neural gradients are near-lognormal: improved quantized and sparse training</th>
<th><a href="https://arxiv.org/abs/2006.08173">ICLR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Multiprize lottery ticket hypothesis: Finding accurate binary neural networks by pruning a randomly weighted network</th>
<th><a href="https://arxiv.org/abs/2103.09377">ICLR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>High-capacity expert binary networks</th>
<th><a href="https://arxiv.org/abs/2010.03558">ICLR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction<a href="https://github.com/yhhhli/BRECQ">[PyTorch]</a>:star:58</th>
<th><a href="https://openreview.net/forum?id=POWv6hDd9XH">ICLR 2021</a></th>
<th>Uni/MP</th>
<th>Linear</th>
<th>PTQ</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BSQ: Exploring Bit-Level Sparsity for Mixed-Precision Neural Network Quantization<a href="https://github.com/yanghr/BSQ">[PyTorch]</a>:star:9</th>
<th><a href="https://openreview.net/forum?id=TiXl51SCNw8">ICLR 2021</a></th>
<th>MP</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Degree-Quant: Quantization-Aware Training for Graph Neural Networks<a href="https://github.com/camlsys/degree-quant">[PyTorch]</a>:star:17</th>
<th><a href="https://openreview.net/forum?id=NSBrFgJAHg">ICLR 2021</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Incremental few-shot learning via vector quantization in deep embedded space</th>
<th><a href="https://openreview.net/pdf?id=3SV-ZePhnZM">ICLR 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Simple Augmentation Goes a Long Way: ADRL for DNN Quantization</th>
<th><a href="https://openreview.net/forum?id=Qr0aRliE_Hb">ICLR 2021</a></th>
<th>MP</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training with Quantization Noise for Extreme Model Compression</th>
<th><a href="https://openreview.net/forum?id=dV19Yyi1fS3">ICLR 2021</a></th>
<th>Uni</th>
<th>PQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>CoDeNet: Algorithm-hardware Co-design for Deformable Convolution</th>
<th><a href="https://openreview.net/forum?id=_aaR7LeZOiG">FPGA 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Opq: Compressing deep neural networks with one-shot pruning quantization</th>
<th><a href="https://ojs.aaai.org/index.php/AAAI/article/view/16950">AAAI 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A white paper on neural network quantization</th>
<th><a href="https://arxiv.org/abs/2106.08295">arXiv 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Kdlsq-bert: A quantized bert combining knowledge distillation with learned step size quantization</th>
<th><a href="https://arxiv.org/abs/2101.05938">arXiv 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Pruning and quantization for deep neural network acceleration: A survey</th>
<th><a href="https://arxiv.org/abs/2101.09671">arXiv 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Confounding tradeoffs for neural network quantization</th>
<th><a href="https://arxiv.org/abs/2102.06366">arXiv 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Dynamic precision analog computing for neural networks</th>
<th><a href="https://arxiv.org/abs/2102.06365">arXiv 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Boolnet: Minimizing the energy consumption of binary neural networks</th>
<th><a href="https://arxiv.org/abs/2106.06991">arXiv 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization-aware pruning for efficient low latency neural network inference</th>
<th><a href="https://arxiv.org/abs/2102.11289">arXiv 2021</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Any-Precision Deep Neural Networks<a href="https://github.com/SHI-Labs/Any-Precision-DNNs">[PyTorch]</a>:star:33</th>
<th><a href="https://arxiv.org/abs/1911.07346">arXiv 2021</a></th>
<th>MP</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Phoenix: A Low-Precision Floating-Point Quantization Oriented Architecture for Convolutional Neural Networks</th>
<th><a href="https://arxiv.org/pdf/2003.02628.pdf">TVLSI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Hierarchical Binary CNNs for Landmark Localization with Limited Resources</th>
<th><a href="https://arxiv.org/abs/1808.04803">TPAMI 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Neural Network Compression by In-Parallel Pruning-Quantization</th>
<th><a href="https://ieeexplore.ieee.org/document/8573867">TPAMI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Efficient U-Nets: A Coupled and Quantized Approach</th>
<th><a href="https://ieeexplore.ieee.org/document/8674614">TPAMI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>SIMBA: A Skyrmionic In-Memory Binary Neural Network Accelerator</th>
<th><a href="https://arxiv.org/abs/2003.05132">TMAG 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Design of High Robustness BNN Inference Accelerator Based on Binary Memristors</th>
<th><a href="https://ieeexplore.ieee.org/document/9112690">TED 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Resource-Efficient Inference Accelerator for Binary Convolutional Neural Networks</th>
<th><a href="https://ieeexplore.ieee.org/document/9144282">TCSII 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Compressing deep neural networks on FPGAs to binary and ternary precision with HLS4ML</th>
<th><a href="https://arxiv.org/abs/2003.06308">MLST 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification, and Local Computations:fire:126</th>
<th><a href="https://arxiv.org/pdf/1906.02367.pdf">JSAIT 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Gradient
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BNN Pruning: Pruning Binary Neural Network Guided by Weight Flipping Frequency</th>
<th><a href="https://ieeexplore.ieee.org/document/9136977">ISQED 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>MuBiNN: Multi-Level Binarized Recurrent Neural Network for EEG Signal Classification</th>
<th><a href="https://arxiv.org/pdf/2004.08914.pdf">ISCS 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Novel In-DRAM Accelerator Architecture for Binary Neural Network</th>
<th><a href="https://ieeexplore.ieee.org/abstract/document/9097642">COOLCHIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>DFQF: Data Free Quantization-aware Fine-tuning</th>
<th><a href="https://proceedings.mlr.press/v129/li20a.html">ACML 2020</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>An Energy-Efficient and High Throughput in-Memory Computing Bit-Cell With Excellent Robustness Under Process Variations for Binary Neural Network</th>
<th><a href="https://ieeexplore.ieee.org/document/9091590">ACCESS 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>CP-NAS: Child-Parent Neural Architecture Search for Binary Neural Networks</th>
<th><a href="https://www.ijcai.org/proceedings/2020/0144.pdf">IJCAI 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Fully 8-bit Integer Inference for the Transformer Model</th>
<th><a href="https://www.ijcai.org/proceedings/2020/520">IJCAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Soft Threshold Ternary Networks</th>
<th><a href="https://www.ijcai.org/Proceedings/2020/0318.pdf">IJCAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Overflow Aware Quantization: Accelerating Neural Network Inference by Low-bit Multiply-Accumulate Operations</th>
<th><a href="https://www.ijcai.org/proceedings/2020/121">IJCAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Direct Quantization for Training Highly Accurate Low Bit-width Deep Neural Networks</th>
<th><a href="https://www.ijcai.org/proceedings/2020/292">IJCAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fully Nested Neural Network for Adaptive Compression and Quantization</th>
<th><a href="https://www.ijcai.org/proceedings/2020/288">IJCAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Path sample-analytic gradient estimators for stochastic binary networks</th>
<th><a href="https://arxiv.org/abs/2006.03143">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Efficient exact verification of binarized neural networks</th>
<th><a href="https://arxiv.org/abs/2005.03597">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Comparing fisher information regularization with distillation for dnn quantization</th>
<th><a href="https://openreview.net/forum?id=JsRdc90lpws">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Position-based scaled gradient for model quantization and sparse training</th>
<th><a href="https://arxiv.org/abs/2005.11035">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Flexor: Trainable fractional quantization</th>
<th><a href="https://arxiv.org/abs/2009.04126">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Adaptive Gradient Quantization for Data-Parallel SGD<a href="https://github.com/tabrizian/learning-to-quantize">[PyTorch]</a>:star:13</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/20b5e1cf8694af7a3c1ba4a87f073021-Paper.pdf">NeurIPS 2020</a></th>
<th>T</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Bayesian Bits: Unifying Quantization and Pruning</th>
<th><a href="https://nips.cc/virtual/2020/public/poster_3f13cf4ddf6fc50c0d39a1d5aeb57dd8.html">NeurIPS 2020</a></th>
<th>MP</th>
<th></th>
<th>QAT/PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Distribution-free binary classification: prediction sets, confidence intervals and calibration</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/26d88423fc6da243ffddf161ca712757-Paper.pdf">NeurIPS 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>FleXOR: Trainable Fractional Quantization</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/0e230b1a582d76526b7ad7fc62ae937d-Paper.pdf">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks:fire:60</th>
<th><a href="https://proceedings.neurips.cc//paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf">NeurIPS 2020</a></th>
<th>MP</th>
<th>Linear</th>
<th>QAT</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Hierarchical Quantized Autoencoders<a href="https://github.com/speechmatics/hqa">[PyTorch]</a>:star:22</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/309fee4e541e51de2e41f21bebb342aa-Paper.pdf">NeurIPS 2020</a></th>
<th></th>
<th>Linear</th>
<th></th>
<th>Image Compression</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Position-based Scaled Gradient for Model Quantization and Pruning<a href="https://github.com/Jangho-Kim/PSG-pytorch">[PyTorch]</a>:star:14</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/eb1e78328c46506b46a4ac4a1e378b91-Paper.pdf">NeurIPS 2020</a></th>
<th>Uni</th>
<th></th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Pushing the Limits of Narrow Precision Inferencing at Cloud Scale with Microsoft Floating Point</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/747e32ab0fea7fbd2ad9ec03daa3f840-Paper.pdf">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantized Variational Inference<a href="https://github.com/amirdib/quantized-variational-inference">[PyTorch]</a></th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/e2a23af417a2344fe3a23e652924091f-Paper.pdf">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Robust Quantization: One Model to Rule Them All</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/3948ead63a9f2944218de038d8934305-Paper.pdf">NeurIPS 2020</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT/PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Rotated Binary Neural Network<a href="https://github.com/lmbxmu/RBNN">[PyTorch]</a>:star:63</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/53c5b2affa12eed84dfec9bfd83550b1-Paper.pdf">NeurIPS 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Searching for Low-Bit Weights in Quantized Neural Networks<a href="https://github.com/zhaohui-yang/Binary-Neural-Networks/tree/main/SLB">[PyTorch]</a>:star:20</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/2a084e55c87b1ebcdaad1f62fdbbac8e-Paper.pdf">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Ultra-Low Precision 4-bit Training of Deep Neural Networks</th>
<th><a href="https://proceedings.neurips.cc/paper/2020/file/13b919438259814cd5be8cb45877d577-Paper.pdf">NeurIPS 2020</a></th>
<th>Uni</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Universally Quantized Neural Compression</th>
<th><a href="https://nips.cc/virtual/2020/public/poster_92049debbe566ca5782a3045cf300a3c.html">NeurIPS 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>TernaryBERT: Distillation-aware Ultra-low Bit BERT</th>
<th><a href="aclweb.org/anthology/2020.emnlp-main.37.pdf">EMNLP 2020</a></th>
<th>T</th>
<th>OptN</th>
<th>QAT</th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>One weight bitwidth to rule them all</th>
<th><a href="https://arxiv.org/abs/2008.09916">ECCV 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>DBQ: A Differentiable Branch Quantizer for Lightweight Deep Neural Networks</th>
<th><a href="https://arxiv.org/abs/2007.09818">ECCV 2020</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Transferring Quantization<a href="https://github.com/xiezheng-cs/DTQ">[PyTorch]</a>:star:15</th>
<th><a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530613.pdf">ECCV 2020</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Differentiable Joint Pruning and Quantization for Hardware Efficiency</th>
<th><a href="https://arxiv.org/abs/2007.10463">ECCV 2020</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Finding Non-Uniform Quantization Schemes using Multi-Task Gaussian Processes</th>
<th><a href="https://arxiv.org/abs/2007.07743">ECCV 2020</a></th>
<th>MP</th>
<th>PQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Generative Low-bitwidth Data Free Quantization<a href="https://github.com/xushoukai/GDFQ">[PyTorch]</a>:star:23</th>
<th><a href="https://arxiv.org/abs/2003.03603">ECCV 2020</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HMQ: Hardware Friendly Mixed Precision Quantization Block for CNNs<a href="https://github.com/sony-si/ai-research">[PyTorch]</a>:star:37</th>
<th><a href="https://arxiv.org/abs/2007.09952">ECCV 2020</a></th>
<th>MP</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>PAMS: Quantized Super-Resolution via Parameterized Max Scale</th>
<th><a href="https://arxiv.org/abs/2011.04212">ECCV 2020</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Post-Training Piecewise Linear Quantization for Deep Neural Networks</th>
<th><a href="https://arxiv.org/pdf/2002.00104.pdf">ECCV 2020</a></th>
<th>T/Uni</th>
<th>Linear</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>QuEST: Quantized Embedding Space for Transferring Knowledge</th>
<th><a href="https://arxiv.org/abs/1912.01540">ECCV 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization Guided JPEG Artifact Correction</th>
<th><a href="https://arxiv.org/abs/2004.09320">ECCV 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Search What You Want: Barrier Panelty NAS for Mixed Precision Quantization</th>
<th><a href="https://arxiv.org/abs/2007.10026">ECCV 2020</a></th>
<th>MP</th>
<th>Linear</th>
<th></th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Task-Aware Quantization Network for JPEG Image Compression</th>
<th><a href="https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650307.pdf">ECCV 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>End to End Binarized Neural Networks for Text Classification</th>
<th><a href="https://arxiv.org/abs/2010.05223">ACL 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Differentiable Product Quantization for End-to-End Embedding Compression<a href="https://github.com/chentingpc/dpq_embedding_compression">[PyTorch]</a>:star:38</th>
<th><a href="http://proceedings.mlr.press/v119/chen20l/chen20l.pdf">ICML 2020</a></th>
<th>MP</th>
<th>PQ</th>
<th>QAT</th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Don’t Waste Your Bits! Squeeze Activations and Gradients for Deep Neural Networks via TinyScript</th>
<th><a href="http://proceedings.mlr.press/v119/fu20c/fu20c.pdf">ICML 2020</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Moniqua: Modulo Quantized Communication in Decentralized SGD</th>
<th><a href="http://proceedings.mlr.press/v119/lu20a/lu20a.pdf">ICML 2020</a></th>
<th>B/T</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Online Learned Continual Compression with Adaptive Quantization Modules<a href="https://github.com/pclucas14/adaptive-quantization-modules">[PyTorch]</a>:star:19</th>
<th><a href="https://arxiv.org/pdf/1911.08019.pdf">ICML 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Accurate Post-training Network Quantization via Bit-Split and Stitching<a href="https://github.com/wps712/BitSplit">[PyTorch]</a>:star:23</th>
<th><a href="http://proceedings.mlr.press/v119/wang20c/wang20c.pdf">ICML 2020</a></th>
<th>T/Uni</th>
<th>OptN</th>
<th>PTQ</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Binary Neural Networks through Learning with Noisy Supervision</th>
<th><a href="https://arxiv.org/pdf/2010.04871.pdf">ICML 2020</a></th>
<th>B</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Up or Down? Adaptive Rounding for Post-Training Quantization</th>
<th><a href="https://arxiv.org/pdf/2004.10568.pdf">ICML 2020</a></th>
<th></th>
<th></th>
<th>PTQ</th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Variational Bayesian Quantization<a href="https://github.com/mandt-lab/vbq">[PyTorch]</a>:star:19</th>
<th><a href="http://proceedings.mlr.press/v119/yang20a/yang20a.pdf">ICML 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Balanced binary neural networks with gated residual</th>
<th><a href="https://arxiv.org/abs/1909.12117">ICASSP 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Spatial RNN Codec for End-To-End Image Compression</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_A_Spatial_RNN_Codec_for_End-to-End_Image_Compression_CVPR_2020_paper.pdf">CVPR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>APQ: Joint Search for Network Architecture, Pruning and Quantization Policy</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_APQ_Joint_Search_for_Network_Architecture_Pruning_and_Quantization_Policy_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>MP</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>AdaBits: Neural Network Quantization With Adaptive Bit-Widths</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Jin_AdaBits_Neural_Network_Quantization_With_Adaptive_Bit-Widths_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>MP</th>
<th>Linear</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Adaptive Loss-Aware Quantization for Multi-Bit Networks</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Qu_Adaptive_Loss-Aware_Quantization_for_Multi-Bit_Networks_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>MP</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Automatic Neural Network Compression by Sparsity-Quantization Joint Learning: A Constrained Optimization-Based Approach</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Automatic_Neural_Network_Compression_by_Sparsity-Quantization_Joint_Learning_A_Constrained_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>MP</th>
<th>OptN</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Central Similarity Quantization for Efficient Image and Video Retrieval<a href="https://github.com/yuanli2333/Hadamard-Matrix-for-hashing">[PyTorch]</a>:star:161</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Data-Free Network Quantization With Adversarial Knowledge Distillation</th>
<th><a href="https://arxiv.org/abs/2005.04136">CVPR 2020</a></th>
<th>Uni</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Forward and Backward Information Retention for Accurate Binary Neural Networks<a href="https://github.com/htqin/IR-Net">[PyTorch]</a>:star:133</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Qin_Forward_and_Backward_Information_Retention_for_Accurate_Binary_Neural_Networks_CVPR_2020_paper.pdf">CVPR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Generalized Product Quantization Network for Semi-Supervised Image Retrieval</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Jang_Generalized_Product_Quantization_Network_for_Semi-Supervised_Image_Retrieval_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>MP</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>LSQ+: Improving low-bit quantization through learnable offsets and better initialization</th>
<th><a href="https://arxiv.org/abs/2004.09576">CVPR 2020</a></th>
<th>MP</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>M-LVC: Multiple Frames Prediction for Learned Video Compression<a href="https://github.com/JianpingLin/M-LVC_CVPR2020">[PyTorch]</a>:star:51</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_M-LVC_Multiple_Frames_Prediction_for_Learned_Video_Compression_CVPR_2020_paper.pdf">CVPR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>OctSqueeze: Octree-Structured Entropy Model for LiDAR Compression</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_OctSqueeze_Octree-Structured_Entropy_Model_for_LiDAR_Compression_CVPR_2020_paper.pdf">CVPR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Structured Compression by Weight Encryption for Unstructured Pruning and Quantization</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Kwon_Structured_Compression_by_Weight_Encryption_for_Unstructured_Pruning_and_Quantization_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>T</th>
<th>OptN</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Quantized Neural Networks With a Full-Precision Auxiliary Module</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhuang_Training_Quantized_Neural_Networks_With_a_Full-Precision_Auxiliary_Module_CVPR_2020_paper.pdf">CVPR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>ZeroQ: A Novel Zero Shot Quantization Framework:fire:106<a href="https://github.com/amirgholami/ZeroQ">[PyTorch]</a>:star:188</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2020/html/Cai_ZeroQ_A_Novel_Zero_Shot_Quantization_Framework_CVPR_2020_paper.html">CVPR 2020</a></th>
<th>MP</th>
<th>Linear</th>
<th>PTQ</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Neural network quantization with adaptive bitwidths</th>
<th><a href="https://arxiv.org/abs/1912.09666">CVPR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BNNsplit: Binarized Neural Networks for embedded distributed FPGA-based computing systems</th>
<th><a href="https://ieeexplore.ieee.org/abstract/document/9116220/">DATE 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>PhoneBit: Efficient GPU-Accelerated Binary Neural Network Inference Engine for Mobile Phones</th>
<th><a href="https://dl.acm.org/doi/abs/10.5555/3408352.3408531">DATE 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>OrthrusPE: Runtime Reconfigurable Processing Elements for Binary Neural Networks</th>
<th><a href="https://ieeexplore.ieee.org/document/9116308">DATE 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training binary neural networks with real-to-binary convolutions:fire:66</th>
<th><a href="https://arxiv.org/abs/2003.11535">ICLR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binaryduo: Reducing gradient mismatch in binary activation network by coupling binary activations</th>
<th><a href="https://arxiv.org/abs/2002.06517">ICLR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Dms: Differentiable dimension search for binary neural networks</th>
<th><a href="https://openreview.net/pdf?id=XKeyCSUWusK">ICLR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Once-for-all: Train one network and specialize it for efficient deployment</th>
<th><a href="https://arxiv.org/abs/1908.09791">ICLR 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks<a href="https://github.com/yhhhli/APoT_Quantization">[PyTorch]</a>:star:150</th>
<th><a href="https://openreview.net/pdf?id=BkgXT24tDS">ICLR 2020</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>And the Bit Goes Down: Revisiting the Quantization of Neural Networks:fire:64<a href="https://github.com/facebookresearch/kill-the-bits">[PyTorch]</a>:star:619</th>
<th><a href="https://openreview.net/pdf?id=rJehVyrKwH">ICLR 2020</a></th>
<th>MP</th>
<th>PQ</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>AutoQ: Automated Kernel-Wise Neural Network Quantization</th>
<th><a href="https://openreview.net/pdf?id=rygfnn4twS">ICLR 2020</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>FSNet: Compression of Deep Convolutional Neural Networks by Filter Summary</th>
<th><a href="https://openreview.net/pdf?id=S1xtORNFwH">ICLR 2020</a></th>
<th>Uni</th>
<th>Linear</th>
<th>PTQ</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Gradient $\ell_1$ Regularization for Quantization Robustness</th>
<th><a href="https://openreview.net/pdf?id=ryxK0JBtPr">ICLR 2020</a></th>
<th>Uni</th>
<th>Linear</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning Hierarchical Discrete Linguistic Units from Visually-Grounded Speech</th>
<th><a href="https://openreview.net/pdf?id=B1elCp4KwH">ICLR 2020</a></th>
<th>MP</th>
<th>PQ</th>
<th></th>
<th>Speech</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Linear Symmetric Quantization of Neural Networks for Low-precision Integer Hardware</th>
<th><a href="https://openreview.net/pdf?id=H1lBj2VFPS">ICLR 2020</a></th>
<th>B/T/Uni</th>
<th>LQ</th>
<th>QAT</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Mixed Precision DNNs: All you need is a good parametrization</th>
<th><a href="https://openreview.net/pdf?id=Hyx0slrFvH">ICLR 2020</a></th>
<th>MP</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Precision Gating: Improving Neural Network Efficiency with Dynamic Dual-Precision Activations</th>
<th><a href="https://openreview.net/pdf?id=SJgVU0EKwS">ICLR 2020</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Shifted and Squeezed 8-bit Floating Point format for Low-Precision Training of Deep Neural Networks</th>
<th><a href="https://openreview.net/pdf?id=Bkxe2AVtPS">ICLR 2020</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Riptide: Fast End-to-End Binarized Neural Networks</th>
<th><a href="https://ubicomplab.cs.washington.edu/pdfs/riptide.pdf">SysML 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Adversarial Attack on Deep Product Quantization Network for Image Retrieval</th>
<th><a href="https://arxiv.org/abs/2002.11374">AAAI 2020</a></th>
<th></th>
<th>PQ</th>
<th></th>
<th>Image Retrieval</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Aggregated Learning: A Vector-Quantization Approach to Learning Neural Network Classifiers<a href="https://github.com/SITE5039/AgrLearn">[PyTorch]</a>:star:3</th>
<th><a href="https://arxiv.org/abs/2001.03955">AAAI 2020</a></th>
<th></th>
<th>PQ</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Embedding Compression with Isotropic Iterative Quantization</th>
<th><a href="https://arxiv.org/abs/2001.05314">AAAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th>N/Image Retrieval</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HLHLp: Quantized Neural Networks Training for Reaching Flat Minima in Loss Surface</th>
<th><a href="https://ojs.aaai.org//index.php/AAAI/article/view/6035">AAAI 2020</a></th>
<th>B</th>
<th>Linear</th>
<th>QAT</th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Indirect Stochastic Gradient Quantization and its Application in Distributed</th>
<th><a href="https://ojs.aaai.org//index.php/AAAI/article/view/5707">AAAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Norm-Explicit Quantization: Improving Vector Quantization for Maximum Inner Product Search</th>
<th><a href="https://arxiv.org/abs/1911.04654">AAAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT:fire:125</th>
<th><a href="https://ojs.aaai.org/index.php/AAAI/article/view/6409">AAAI 2020</a></th>
<th>MP</th>
<th>Linear</th>
<th>QAT</th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantized Compressive Sampling of Stochastic Gradients for Efficient Communication in Distributed Deep Learning</th>
<th><a href="https://ojs.aaai.org/index.php/AAAI/article/view/5706">AAAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>RTN: Reparameterized Ternary Network</th>
<th><a href="https://arxiv.org/abs/1912.02057">AAAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Accurate Low Bit-width Quantization with Multiple Phase Adaptations</th>
<th><a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhuang_Towards_Effective_Low-Bitwidth_CVPR_2018_paper.pdf">AAAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Accurate Quantization and Pruning via Data-free Knowledge Transfer</th>
<th><a href="https://arxiv.org/abs/2010.07334">AAAI 2020</a></th>
<th>MP</th>
<th>LQ</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Vector Quantization-Based Regularization for Autoencoders<a href="https://github.com/AlbertOh90/Soft-VQ-VAE/">[PyTorch]</a>:star:11</th>
<th><a href="https://ojs.aaai.org/index.php/AAAI/article/view/6108">AAAI 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Binary Neural Networks using the Bayesian Learning Rule</th>
<th><a href="https://arxiv.org/abs/2002.10778">CoRR 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Integer quantization for deep learning inference: Principles and empirical evaluation</th>
<th><a href="https://arxiv.org/abs/2004.09602">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Wrapnet: Neural net inference with ultra-low-resolution arithmetic</th>
<th><a href="https://arxiv.org/abs/2007.13242">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Leveraging automated mixed-low-precision quantization for tiny edge microcontrollers</th>
<th><a href="https://arxiv.org/abs/2008.05124">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Biqgemm: matrix multiplication with lookup table for binary-coding-based quantized dnns</th>
<th><a href="https://arxiv.org/abs/2005.09904">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Near-lossless post-training quantization of deep neural networks via a piecewise linear approximation</th>
<th><a href="https://arxiv.org/abs/2002.00104">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Efficient execution of quantized deep learning models: A compiler approach</th>
<th><a href="https://arxiv.org/abs/2006.10226">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A statistical framework for low-bitwidth training of deep neural networks</th>
<th><a href="https://arxiv.org/abs/2010.14298">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>What is the state of neural network pruning?</th>
<th><a href="https://arxiv.org/abs/2003.03033">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Language models are fewshot learners</th>
<th><a href="https://arxiv.org/abs/2005.14165">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Shifted and squeezed 8-bit floating point format for low-precision training of deep neural networks</th>
<th><a href="https://arxiv.org/abs/2001.05674">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Gradient l1 regularization for quantization robustness</th>
<th><a href="https://arxiv.org/abs/2002.07520">arXiv 2020</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BinaryBERT: Pushing the Limit of BERT Quantization</th>
<th><a href="https://arxiv.org/abs/2012.15701">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Understanding Learning Dynamics of Binary Neural Networks via Information Bottleneck</th>
<th><a href="https://arxiv.org/abs/2006.07522">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Lossless Binary Convolutional Neural Networks Using Piecewise Approximation</th>
<th><a href="https://arxiv.org/abs/2008.03520">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>RPR: Random Partition Relaxation for Training; Binary and Ternary Weight Neural Networks</th>
<th><a href="https://arxiv.org/abs/2001.01091">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?</th>
<th><a href="https://arxiv.org/abs/2001.05936">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Accelerating Binarized Neural Networks via Bit-Tensor-Cores in Turing GPUs</th>
<th><a href="https://arxiv.org/abs/2006.16578">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Distillation Guided Residual Learning for Binary Convolutional Neural Networks</th>
<th><a href="https://arxiv.org/abs/2007.05223">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>How Does Batch Normalization Help Binary Training?</th>
<th><a href="https://arxiv.org/abs/1909.09139">arXiv 2020</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming<a href="https://github.com/itayhubara/CalibTIP">[PyTorch]</a></th>
<th><a href="https://arxiv.org/abs/2006.10518">arXiv 2020</a></th>
<th>MP</th>
<th>OptN</th>
<th>PTQ</th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Product Engine for Energy-Efficient Execution of Binary Neural Networks Using Resistive Memories</th>
<th><a href="https://ieeexplore.ieee.org/abstract/document/8920343/">VLSI-SoC 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Binary Reconstruction for Cross-Modal Hashing:fire:78</th>
<th><a href="https://arxiv.org/abs/1708.05127">TMM 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Compact Hash Code Learning With Binary Deep Neural Network</th>
<th><a href="https://arxiv.org/pdf/1712.02956.pdf">TM 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Recursive Binary Neural Network Training Model for Efficient Usage of On-Chip Memory</th>
<th><a href="https://ieeexplore.ieee.org/document/8643565">TCSI 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Xcel-RAM: Accelerating Binary Neural Networks in High-Throughput SRAM Compute Arrays</th>
<th><a href="https://ieeexplore.ieee.org/document/8698312">TCSI 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>PXNOR: Perturbative Binary Neural Network</th>
<th><a href="https://ieeexplore.ieee.org/document/8909493">ROEDUNET 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>An Energy-Efficient Reconfigurable Processor for Binary-and Ternary-Weight Neural Networks With Flexible Data Bit Width</th>
<th><a href="https://ieeexplore.ieee.org/document/8581485">JSSC 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Accurate Binary Neural Networks from Scratch</th>
<th><a href="https://ieeexplore.ieee.org/document/8802610">ICIP 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binarized Depthwise Separable Neural Network for Object Tracking in FPGA</th>
<th><a href="https://dl.acm.org/doi/abs/10.1145/3299874.3318034">GLSVLSI 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Review of Binarized Neural Networks</th>
<th><a href="https://www.mdpi.com/2079-9292/8/6/661">Electronics 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BiScaled-DNN: Quantizing Long-tailed Datastructures with Two Scale Factors for Deep Neural Networks</th>
<th><a href="https://dl.acm.org/doi/10.1145/3316781.3317783">DAC 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Using Neuroevolved Binary Neural Networks to solve reinforcement learning environments</th>
<th><a href="https://ieeexplore.ieee.org/abstract/document/8953134">APCCAS 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binarized Neural Networks for Resource-Efficient Hashing with Minimizing Quantization Loss</th>
<th><a href="https://www.ijcai.org/proceedings/2019/145">IJCAI 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binarized Collaborative Filtering with Distilling Graph Convolutional Network</th>
<th><a href="https://arxiv.org/abs/1906.01829">IJCAI 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Latent Weights Do Not Exist: Rethinking Binarized Neural Network Optimization:fire:53</th>
<th><a href="https://arxiv.org/abs/1906.02107">NeurIPS 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Mean Field Theory of Quantized Deep Networks: The Quantization-Depth Trade-Off<a href="https://github.com/yanivbl6/quantized_meanfield">[PyTorch]:star:12</a></th>
<th><a href="https://papers.nips.cc/paper/2019/file/38ef4b66cb25e92abe4d594acb841471-Paper.pdf">NeurIPS 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Theory
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Bit Efficient Quantization for Deep Neural Networks</th>
<th><a href="https://arxiv.org/abs/1910.04877">NeurIPS 2019</a></th>
<th>Uni</th>
<th>Linear/Log</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Communication-Efficient Distributed Learning via Lazily Aggregated Quantized Gradients</th>
<th><a href="https://papers.nips.cc/paper/2019/file/4e87337f366f72daa424dae11df0538c-Paper.pdf">NeurIPS 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Dimension-Free Bounds for Low-Precision Training</th>
<th><a href="https://papers.nips.cc/paper/2019/file/d4cd91e80f36f8f3103617ded9128560-Paper.pdf">NeurIPS 2019</a></th>
<th></th>
<th>Log</th>
<th></th>
<th></th>
<th>Theory
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Double Quantization for Communication-Efficient Distributed Optimization</th>
<th><a href="https://papers.nips.cc/paper/2019/file/ea4eb49329550caaa1d2044105223721-Paper.pdf">NeurIPS 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Focused Quantization for Sparse CNNs</th>
<th><a href="https://papers.nips.cc/paper/2019/file/58aaee7ae94b52697ad3b9275d46ec7f-Paper.pdf">NeurIPS 2019</a></th>
<th>Uni</th>
<th>LQ</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Generalization Error Analysis of Quantized Compressive Learning</th>
<th><a href="https://papers.nips.cc/paper/2019/file/1a638db8311430c6c018bf21e1a0b7fb-Paper.pdf">NeurIPS 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Theory
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks:fire:56</th>
<th><a href="https://papers.nips.cc/paper/2019/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf">NeurIPS 2019</a></th>
<th>Uni</th>
<th></th>
<th>PTQ</th>
<th>C/O/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>MetaQuant: Learning to Quantize by Learning to Penetrate Non-differentiable Quantization<a href="https://github.com/csyhhu/MetaQuant">[PyTorch]</a>:star:48</th>
<th><a href="https://papers.nips.cc/paper/2019/file/f8e59f4b2fe7c5705bf878bbd494ccdf-Paper.pdf">NeurIPS 2019</a></th>
<th>B</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Model Compression with Adversarial Robustness: A Unified Optimization Framework</th>
<th><a href="https://papers.nips.cc/paper/2019/file/2ca65f58e35d9ad45bf7f3ae5cfd08f1-Paper.pdf">NeurIPS 2019</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Normalization Helps Training of Quantized LSTM</th>
<th><a href="https://papers.nips.cc/paper/2019/file/f8eb278a8bce873ef365b45e939da38a-Paper.pdf">NeurIPS 2019</a></th>
<th>B/T/Uni</th>
<th></th>
<th></th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Post-training 4-bit quantization of convolution networks for rapid-deployment:fire:161<a href="https://github.com/submission2019/cnn-quantization">[PyTorch]</a>:star:163</th>
<th><a href="https://proceedings.neurips.cc/paper/2019/file/c0a62e133894cdce435bcb4a5df1db2d-Paper.pdf">NeurIPS 2019</a></th>
<th>T/Uni</th>
<th></th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Qsparse-local-SGD: Distributed SGD with Quantization, Sparsification and Local Computations:fire:126</th>
<th><a href="https://papers.nips.cc/paper/2019/file/d202ed5bcfa858c15a9f383c3e386ab2-Paper.pdf">NeurIPS 2019</a></th>
<th>B</th>
<th>Randomized/Sign</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BinaryDenseNet: Developing an architecture for binary neural networks</th>
<th><a href="https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Bethge_BinaryDenseNet_Developing_an_Architecture_for_Binary_Neural_Networks_ICCVW_2019_paper.pdf">ICCVW 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Low-bit quantization of neural networks for efficient inference:fire:112</th>
<th><a href="https://arxiv.org/abs/1902.06822">ICCV 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Bayesian Optimized 1-Bit CNNs</th>
<th><a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/Gu_Bayesian_Optimized_1-Bit_CNNs_ICCV_2019_paper.pdf">ICCV 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Data-Free Quantization Through Weight Equalization and Bias Correction:fire:135</th>
<th><a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf">ICCV 2019</a></th>
<th>Uni</th>
<th>Linear</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Differentiable Soft Quantization: Bridging Full-Precision and Low-Bit Neural Networks:fire:129</th>
<th><a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Differentiable_Soft_Quantization_Bridging_Full-Precision_and_Low-Bit_Neural_Networks_ICCV_2019_paper.pdf">ICCV 2019</a></th>
<th>B/T/Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision:fire:155</th>
<th><a href="https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html">ICCV 2019</a></th>
<th>MP</th>
<th>Linear</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Proximal Mean-Field for Neural Network Quantization</th>
<th><a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/Ajanthan_Proximal_Mean-Field_for_Neural_Network_Quantization_ICCV_2019_paper.pdf">ICCV 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Unsupervised Neural Quantization for Compressed-Domain Similarity Search<a href="https://github.com/stanis-morozov/unq">[PyTorch]</a>:star:28</th>
<th><a href="https://openaccess.thecvf.com/content_ICCV_2019/papers/Morozov_Unsupervised_Neural_Quantization_for_Compressed-Domain_Similarity_Search_ICCV_2019_paper.pdf">ICCV 2019</a></th>
<th>MP</th>
<th>LQ</th>
<th></th>
<th>Image Retrieval</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design:fire:169<a href="https://github.com/aravindsrinivas/flowpp">[PyTorch]</a>:star:152</th>
<th><a href="https://arxiv.org/abs/1902.00275">ICML 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Improving Neural Network Quantization without Retraining using Outlier Channel Splitting:fire:114<a href="https://github.com/cornell-zhang/dnn-quant-ocs">[PyTorch]</a>:star:80</th>
<th><a href="https://arxiv.org/abs/1901.09504">ICML 2019</a></th>
<th>Uni</th>
<th>Linear</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Lossless or Quantized Boosting with Integer Arithmetic</th>
<th><a href="http://proceedings.mlr.press/v97/nock19a.html">ICML 2019</a></th>
<th></th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>SWALP : Stochastic Weight Averaging in Low-Precision Training<a href="https://github.com/stevenygd/SWALP">[PyTorch]</a>:star:52</th>
<th><a href="https://arxiv.org/abs/1904.11943">ICML 2019</a></th>
<th>Uni</th>
<th>Linear</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning channel-wise interactionsfor binary convolutional neural networks</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Channel-Wise_Interactions_for_Binary_Convolutional_Neural_Networks_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Circulant binary convolutional networks: Enhancing the performance of 1-bit dcnns with circulant back propagation</th>
<th><a href="https://arxiv.org/abs/1910.10853">CVPR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fighting quantization bias with bias</th>
<th><a href="https://arxiv.org/abs/1906.03193">CVPR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Main/Subsidiary Network Framework for Simplifying Binary Neural Networks</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_A_MainSubsidiary_Network_Framework_for_Simplifying_Binary_Neural_Networks_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binary Ensemble Neural Network: More Bits per Network or More Networks per Bit?:fire:84</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Binary_Ensemble_Neural_Network_More_Bits_per_Network_or_More_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Compressing Unknown Images With Product Quantizer for Efficient Zero-Shot Classification</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Compressing_Unknown_Images_With_Product_Quantizer_for_Efficient_Zero-Shot_Classification_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>MP</th>
<th>PQ/LQ</th>
<th></th>
<th>C/ZSL/GZSL</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Spherical Quantization for Image Search</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Eghbali_Deep_Spherical_Quantization_for_Image_Search_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th>Image Search</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>End-To-End Supervised Product Quantization for Image Search and Retrieval</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Klein_End-To-End_Supervised_Product_Quantization_for_Image_Search_and_Retrieval_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th></th>
<th>PQ</th>
<th></th>
<th>Image Search/Retrieval</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fully Quantized Network for Object Detection:fire:59</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HAQ: Hardware-Aware Automated Quantization With Mixed Precision:fire:305<a href="https://github.com/mit-han-lab/haq">[PyTorch]</a>:star:243</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html">CVPR 2019</a></th>
<th>MP</th>
<th>Linear/K</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning Channel-Wise Interactions for Binary Convolutional Neural Networks</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Learning_Channel-Wise_Interactions_for_Binary_Convolutional_Neural_Networks_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning to Quantize Deep Networks by Optimizing Quantization Intervals With Task Loss:fire:168</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Jung_Learning_to_Quantize_Deep_Networks_by_Optimizing_Quantization_Intervals_With_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>T/Uni</th>
<th>LQ</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization Networks:fire:84</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Quantization_Networks_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>Uni</th>
<th></th>
<th>QAT</th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>SeerNet: Predicting Convolutional Neural Network Feature-Map Sparsity Through Low-Bit Quantization</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Cao_SeerNet_Predicting_Convolutional_Neural_Network_Feature-Map_Sparsity_Through_Low-Bit_Quantization_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>T/Uni</th>
<th>Linear</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Simultaneously Optimizing Weight and Quantizer of Ternary Neural Network Using Truncated Gaussian Approximation</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Simultaneously_Optimizing_Weight_and_Quantizer_of_Ternary_Neural_Network_Using_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>T</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation:fire:91</th>
<th><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhuang_Structured_Binary_Neural_Networks_for_Accurate_Image_Classification_and_Semantic_CVPR_2019_paper.pdf">CVPR 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th>C/S</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Variational information distillation for knowledge transfer:fire:188</th>
<th><a href="https://arxiv.org/abs/1904.05835">CVPR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Proxylessnas: Direct neural architecture search on target task and hardware</th>
<th><a href="https://arxiv.org/abs/1812.00332">ICLR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Accumulation Bit-Width Scaling For Ultra-Low Precision Training Of Deep Networks</th>
<th><a href="https://openreview.net/pdf?id=BklMjsRqY7">ICLR 2019</a></th>
<th>MP</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>Theory
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Analysis of Quantized Models</th>
<th><a href="https://openreview.net/pdf?id=ryM_IoAqYX">ICLR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Theory
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Defensive Quantization: When Efficiency Meets Robustness:fire:81</th>
<th><a href="https://openreview.net/pdf?id=ryetZ20ctX">ICLR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Double Viterbi: Weight Encoding for High Compression Ratio and Fast On-Chip Reconstruction for Deep Neural Network</th>
<th><a href="https://openreview.net/forum?id=HkfYOoCcYX">ICLR 2019</a></th>
<th>B/T/Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>CoDesign
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>From Hard to Soft: Understanding Deep Network Nonlinearities via Vector Quantization and Statistical Inference</th>
<th><a href="https://openreview.net/pdf?id=Syxt2jC5FX">ICLR 2019</a></th>
<th>MP</th>
<th>PQ</th>
<th>QAT</th>
<th>N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning Recurrent Binary/Ternary Weights<a href="https://github.com/arashardakani/Learning-Recurrent-Binary-Ternary-Weights">[PyTorch]</a>:star:13</th>
<th><a href="https://openreview.net/pdf?id=HkNGYjR9FX">ICLR 2019</a></th>
<th>B/T</th>
<th></th>
<th>QAT</th>
<th>C/N</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>On the Universal Approximability and Complexity Bounds of Quantized ReLU Neural Networks</th>
<th><a href="https://openreview.net/pdf?id=SJe9rh0cFX">ICLR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th>C</th>
<th>Theory
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Per-Tensor Fixed-Point Quantization of the Back-Propagation Algorithm</th>
<th><a href="https://openreview.net/pdf?id=rkxaNjA9Ym">ICLR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>ProxQuant: Quantized Neural Networks via Proximal Operators:fire:56<a href="https://github.com/allenbai01/ProxQuant">[PyTorch]</a>:star:17</th>
<th><a href="https://openreview.net/pdf?id=HyzMyhCcK7">ICLR 2019</a></th>
<th>B</th>
<th></th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Relaxed Quantization for Discretized Neural Networks:fire:74</th>
<th><a href="https://openreview.net/pdf?id=HkxjYoCqKX">ICLR 2019</a></th>
<th></th>
<th>LQ</th>
<th></th>
<th>C</th>
<th>Stochastic
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets:fire:81</th>
<th><a href="https://openreview.net/pdf?id=Skh4jRcKQ">ICLR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th>C</th>
<th>Theory
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Fast and Energy-Efficient Binarized Neural Network Inference on FPGA</th>
<th><a href="https://arxiv.org/abs/1810.02068">FPGA 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Neural Network Quantization via Layer-Wise Optimization using Limited Training Data<a href="https://github.com/csyhhu/L-DNQ">[PyTorch]</a>:star:30</th>
<th><a href="https://ojs.aaai.org//index.php/AAAI/article/view/4206">AAAI 2019</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Efficient Quantization for Compact Neural Networks with Binary Weights and Low Bitwidth Activations</th>
<th><a href="https://ojs.aaai.org//index.php/AAAI/article/view/4273">AAAI 2019</a></th>
<th>B</th>
<th>Linear/Log</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Multi-Precision Quantized Neural Networks via Encoding Decomposition of {-1,+1}</th>
<th><a href="https://arxiv.org/abs/1905.13389">AAAI 2019</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C/O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Similarity Preserving Deep Asymmetric Quantization for Image Retrieval</th>
<th><a href="https://ojs.aaai.org//index.php/AAAI/article/view/4828">AAAI 2019</a></th>
<th></th>
<th></th>
<th>QAT</th>
<th>Image Retrieval</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>RBCN: Rectified Binary Convolutional Networks for Enhancing the Performance of 1-bit DCNNs</th>
<th><a href="https://arxiv.org/abs/1908.07748">CoRR 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>TentacleNet: A Pseudo-Ensemble Template for Accurate Binary Convolutional Neural Networks</th>
<th><a href="https://arxiv.org/abs/1912.10103">CoRR 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Improved training of binary networks for human pose estimation and image recognition</th>
<th><a href="https://arxiv.org/abs/1904.05868">CoRR 2019</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binarized Neural Architecture Search</th>
<th><a href="https://arxiv.org/abs/1911.10862">CoRR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Matrix and tensor decompositions for training binary neural networks</th>
<th><a href="https://arxiv.org/abs/1904.07852">CoRR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Back to Simplicity: How to Train Accurate BNNs from Scratch?</th>
<th><a href="https://arxiv.org/abs/1906.08637">CoRR 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>MoBiNet: A Mobile Binary Network for Image Classification</th>
<th><a href="https://arxiv.org/abs/1907.12629">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training high-performance and large-scale deep neural networks with full 8-bit integers</th>
<th><a href="https://arxiv.org/abs/1909.02384">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Knowledge distillation for optimization of quantized deep neural networks</th>
<th><a href="https://arxiv.org/abs/1909.01688">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Accurate and compact convolutional neural networks with trained binarization</th>
<th><a href="https://arxiv.org/abs/1909.11366">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Mixed precision training with 8-bit floating point</th>
<th><a href="https://arxiv.org/abs/1905.12334">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Additive powers-of-two quantization: An efficient nonuniform discretization for neural networks</th>
<th><a href="https://arxiv.org/abs/1909.13144">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Regularizing activation distribution for training binarized deep networks:fire:61</th>
<th><a href="https://arxiv.org/abs/1904.02823">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>The knowledge within: Methods for data-free model compression:fire:50</th>
<th><a href="https://arxiv.org/abs/1912.01274">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Xnornet++: Improved binary neural networks</th>
<th><a href="https://arxiv.org/abs/1909.13863">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Mixed Precision Quantization of ConvNets via Differentiable Neural Architecture Search</th>
<th><a href="https://arxiv.org/abs/1812.00090">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>QKD: Quantization-aware Knowledge Distillation</th>
<th><a href="https://arxiv.org/abs/1911.12491">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>daBNN: A Super Fast Inference Framework for Binary Neural Networks on ARM devices</th>
<th><a href="https://arxiv.org/abs/1908.05858">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Unified INT8 Training for Convolutional Neural Network</th>
<th><a href="https://arxiv.org/abs/1912.12607">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BNN+: Improved Binary Network Training:fire:72</th>
<th><a href="https://openreview.net/forum?id=SJfHg2A5tQ">arXiv 2019</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learned Step Size Quantization:fire:129</th>
<th><a href="https://arxiv.org/abs/1902.08153">arXiv 2019</a></th>
<th>MP</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Same, Same But Different: Recovering Neural Network Quantization Error Through Weight Factorization</th>
<th><a href="https://arxiv.org/abs/1902.01917">arXiv 2019</a></th>
<th>Uni</th>
<th>Linear</th>
<th>PTQ</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>An Energy-Efficient Architecture for Binary Weight Convolutional Neural Networks</th>
<th><a href="https://ieeexplore.ieee.org/document/8103902">TVLSI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks</th>
<th><a href="https://arxiv.org/abs/1809.04570">TRETS 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Inference of quantized neural networks on heterogeneous all-programmable devices</th>
<th><a href="https://arxiv.org/abs/1806.08085">NE 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A survey of FPGA-based accelerators for convolutional neural networks</th>
<th><a href="https://link.springer.com/article/10.1007/s00521-018-3761-1">NCA 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BitStream: Efficient Computing Architecture for Real-Time Low-Power Inference of Binary Neural Networks on CPUs</th>
<th><a href="https://dl.acm.org/doi/10.1145/3240508.3240673">MM 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>ReBNet: Residual Binarized Neural Network</th>
<th><a href="https://arxiv.org/abs/1711.01243">ISFPCCM 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BitFlow: Exploiting Vector Parallelism for Binary Neural Networks on CPU</th>
<th><a href="https://www.semanticscholar.org/paper/BitFlow%3A-Exploiting-Vector-Parallelism-for-Binary-Hu-Zhai/8c9fc9be222b684cee88d287a829020433bdd132">IPDPS 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Distilled binary neural network for monaural speech separation</th>
<th><a href="https://ieeexplore.ieee.org/document/8489456">IJCNN 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fast object detection based on binary deep convolution neural networks</th>
<th><a href="https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/trit.2018.1026">IJCNN 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Analysis and Implementation of Simple Dynamic Binary Neural Networks</th>
<th><a href="https://ieeexplore.ieee.org/document/8489259">IJCNN 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Deep Look into Logarithmic Quantization of Model Parameters in Neural Networks</th>
<th><a href="https://dl.acm.org/doi/abs/10.1145/3291280.3291800">IAIT 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>FBNA: A Fully Binarized Neural Network Accelerator</th>
<th><a href="https://ieeexplore.ieee.org/document/8532584">FPL 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Quantization-Friendly Separable Convolution for MobileNets</th>
<th><a href="https://arxiv.org/abs/1803.08607">EMC2 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Gap-8: A risc-v soc for ai at the edge of the iot</th>
<th><a href="https://ieeexplore.ieee.org/document/8445101/">ASAP 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deterministic Binary Filters for Convolutional Neural Networks</th>
<th><a href="https://www.ijcai.org/proceedings/2018/0380.pdf">IJCAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Planning in Factored State and Action Spaces with Learned Binarized Neural Network Transition Models</th>
<th><a href="https://www.ijcai.org/proceedings/2018/669">IJCAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Moonshine: Distilling with cheap convolutions</th>
<th><a href="https://arxiv.org/abs/1711.02613">NeurIPS 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Linear Speedup Analysis of Distributed Deep Learning with Sparse and Quantized Communication:fire:104</th>
<th><a href="https://papers.nips.cc/paper/2018/file/17326d10d511828f6b34fa6d751739e2-Paper.pdf">NeurIPS 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>GradiVeQ: Vector Quantization for Bandwidth-Efficient Gradient Aggregation in Distributed CNN Training</th>
<th><a href="https://papers.nips.cc/paper/2018/file/cf05968255451bdefe3c5bc64d550517-Paper.pdf">NeurIPS 2018</a></th>
<th></th>
<th>PQ</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Heterogeneous Bitwidth Binarization in Convolutional Neural Networks</th>
<th><a href="https://papers.nips.cc/paper/2018/file/1b36ea1c9b7a1c3ad668b8bb5df7963f-Paper.pdf">NeurIPS 2018</a></th>
<th>MP</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>HitNet: Hybrid Ternary Recurrent Neural Network</th>
<th><a href="https://papers.nips.cc/paper/2018/file/82cec96096d4281b7c95cd7e74623496-Paper.pdf">NeurIPS 2018</a></th>
<th>T/Uni</th>
<th></th>
<th></th>
<th>N</th>
<th>Stochastic
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Scalable methods for 8-bit training of neural networks:fire:151<a href="https://github.com/eladhoffer/quantized.pytorch">[PyTorch]</a>:star:191</th>
<th><a href="https://papers.nips.cc/paper/2018/file/e82c4b19b8151ddc25d4d93baf7b908f-Paper.pdf">NeurIPS 2018</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Deep Neural Networks with 8-bit Floating Point Numbers:fire:213</th>
<th><a href="https://arxiv.org/pdf/1812.08011.pdf">NeurIPS 2018</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>Stochastic
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Bi-real net: Enhancing the performance of 1-bit cnns with improved representational capability and advanced training algorithm:fire:222<a href="https://github.com/liuzechun/Bi-Real-net">[Caffe&pytorch]</a>:star:138</th>
<th><a href="https://arxiv.org/pdf/1808.00278.pdf">ECCV 2018</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks:fire:325<a href="https://github.com/Microsoft/LQ-Nets">[PyTorch]</a>:star:207</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/html/Dongqing_Zhang_Optimized_Quantization_for_ECCV_2018_paper.html">ECCV 2018</a></th>
<th>B/Uni</th>
<th>LQ</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>LSQ++: Lower running time and higher recall in multi-codebook quantization</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Julieta_Martinez_LSQ_lower_runtime_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th>MP</th>
<th>LQ</th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning Compression from limited unlabeled Data</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiangyu_He_Learning_Compression_from_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Product Quantization Network for Fast Image Retrieval</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Tan_Yu_Product_Quantization_Network_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th>Uni</th>
<th>PQ</th>
<th></th>
<th>Image Retrieval</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization Mimic: Towards Very Tiny CNN for Object Detection:fire:55</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Yi_Wei_Quantization_Mimic_Towards_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th>O</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantized Densely Connected U-Nets for Efficient Landmark Localization:fire:105</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhiqiang_Tang_Quantized_Densely_Connected_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>TBN: Convolutional Neural Network with Ternary Inputs and Binary Weights:fire:57</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Diwen_Wan_TBN_Convolutional_Neural_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Binary Weight Networks via Semi-Binary Decomposition</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Qinghao_Hu_Training_Binary_Weight_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Value-aware Quantization for Training and Inference of Neural Networks:fire:75</th>
<th><a href="https://openaccess.thecvf.com/content_ECCV_2018/papers/Eunhyeok_Park_Value-aware_Quantization_for_ECCV_2018_paper.pdf">ECCV 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>SIGNSGD: compressed optimisation for non-convex problems:fire:393<a href="https://github.com/jxbz/signSGD">[PyTorch]</a>:star:54</th>
<th><a href="https://arxiv.org/pdf/1802.04434.pdf">ICML 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Gradient
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization of Fully Convolutional Networks for Accurate Biomedical Image Segmentation:fire:59</th>
<th><a href="https://arxiv.org/abs/1803.04907">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Explicit loss-error-aware quantization for low-bit deep neural networks:fire:67</th>
<th><a href="https://openaccess.thecvf.com/content_cvpr_2018/html/Zhou_Explicit_Loss-Error-Aware_Quantization_CVPR_2018_paper.html">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A biresolution spectral framework for product quantization</th>
<th><a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Mukherjee_A_Biresolution_Spectral_CVPR_2018_paper.pdf">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Amc: Automl for model compression and acceleration on mobile devices:fire:814</th>
<th><a href="https://arxiv.org/abs/1802.03494">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Effective Training of Convolutional Neural Networks with Low-bitwidth Weights and Activations</th>
<th><a href="https://arxiv.org/pdf/1908.04680.pdf">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Modulated convolutional networks</th>
<th><a href="http://pure.aber.ac.uk/ws/files/42035958/Modulated_Convolutional_Networks_final.pdf">CVPR 2018</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference:fire:1013</th>
<th><a href="https://arxiv.org/pdf/1712.05877.pdf">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>SYQ: Learning Symmetric Quantization For Efficient Deep Neural Networks:fire:84<a href="https://github.com/julianfaraone/SYQ">[PyTorch]</a>:star:31</th>
<th><a href="https://arxiv.org/pdf/1807.00301.pdf">CVPR 2018</a></th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Effective Low-bitwidth Convolutional Neural Networks:fire:121</th>
<th><a href="https://arxiv.org/pdf/1711.00205.pdf">CVPR 2018</a></th>
<th></th>
<th></th>
<th>QAT</th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Two-Step Quantization for Low-bit Neural Networks:fire:72</th>
<th><a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Two-Step_Quantization_for_CVPR_2018_paper.pdf">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>CLIP-Q: Deep Network Compression Learning by In-parallel Pruning-Quantization:fire:165</th>
<th><a href="https://ieeexplore.ieee.org/document/8578919">CVPR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Mixed Precision Training of Convolutional Neural Networks using Integer Operations:fire:117</th>
<th><a href="https://arxiv.org/abs/1802.00930">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>An empirical study of binary neural networks’ optimisation</th>
<th><a href="https://openreview.net/forum?id=rJfUCoR5KX">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Adaptive Quantization of Neural Networks</th>
<th><a href="https://openreview.net/pdf?id=SyOK1Sg0W">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Alternating Multi-bit Quantization for Recurrent Neural Networks:fire:87</th>
<th><a href="https://openreview.net/pdf?id=S19dR9x0b">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy:fire:208</th>
<th><a href="https://arxiv.org/pdf/1711.05852.pdf">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Energy-Constrained Compression for Deep Neural Networks via Weighted Sparse Projection and Layer Input Masking<a href="https://github.com/hyang1990/model_based_energy_constrained_compression">[PyTorch]</a>:star:15</th>
<th><a href="https://www.semanticscholar.org/paper/Energy-Constrained-Compression-for-Deep-Neural-via-Yang-Zhu/0986f2ac6755df5d196ceb09b5bdf19593cbbaef">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>CoDesign
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Loss-aware Weight Quantization of Deep Networks:fire:94</th>
<th><a href="https://openreview.net/pdf?id=BkrSv0lA-">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Model compression via distillation and quantization:fire:353<a href="https://github.com/antspy/quantized_distillation">[PyTorch]</a>:star:293</th>
<th><a href="https://openreview.net/forum?id=S1XolQbRW">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training and Inference with Integers in Deep Neural Networks:fire:231<a href="https://github.com/boluoweifenda/WAGE">[[tensorflow]]</a>:star:132</th>
<th><a href="https://arxiv.org/pdf/1802.04680.pdf">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Variational Network Quantization:fire:58</th>
<th><a href="https://openreview.net/pdf?id=ry-TW-WAb">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>WRPN: Wide Reduced-Precision Networks:fire:180</th>
<th><a href="https://arxiv.org/pdf/1709.01134.pdf">ICLR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Adaptive Quantization for Deep Neural Networ:fire:70</th>
<th><a href="https://arxiv.org/pdf/1712.01048.pdf">AAAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Neural Network Compression with Single and Multiple Level Quantization:fire:65<a href="https://github.com/yuhuixu1993/WLQ">[PyTorch]</a>:star:20</th>
<th><a href="https://arxiv.org/pdf/1803.03289.pdf">AAAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Distributed Composite Quantization</th>
<th><a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16470/15671">AAAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM:fire:207</th>
<th><a href="https://arxiv.org/pdf/1707.09870.pdf">AAAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>From Hashing to CNNs: Training Binary Weight Networks via Hashing:fire:62</th>
<th><a href="https://arxiv.org/pdf/1802.02733.pdf">AAAI 2018</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Product Quantized Translation for Fast Nearest Neighbor Search</th>
<th><a href="https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16953/16702">AAAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantized Memory-Augmented Neural Networks</th>
<th><a href="https://arxiv.org/pdf/1711.03712.pdf">AAAI 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>LightNN: Filling the Gap between Conventional Deep Neural Networks and Binarized Networks</th>
<th><a href="https://arxiv.org/abs/1802.02178">CoRR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BinaryRelax: A Relaxation Approach For Training Deep Neural Networks With Quantized Weights</th>
<th><a href="https://arxiv.org/abs/1801.06313">CoRR 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning low precision deep neural networks through regularization</th>
<th><a href="https://arxiv.org/abs/1809.00095">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Blended coarse gradient descent for full quantization of deep neural networks:fire:48</th>
<th><a href="https://arxiv.org/abs/1808.05240">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>XNOR Neural Engine: A Hardware Accelerator IP for 21.6-fJ/op Binary Neural Network Inference:fire:72</th>
<th><a href="https://arxiv.org/pdf/1807.03010.pdf">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A Survey on Methods and Theories of Quantized Neural Networks:fire:128</th>
<th><a href="https://arxiv.org/abs/1808.04752">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantizing Convolutional Neural Networks for Low-Power High-Throughput Inference Engines</th>
<th><a href="https://arxiv.org/abs/1805.07941">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Discovering low-precision networks close to full-precision networks for efficient embedded inference:fire:83</th>
<th><a href="https://arxiv.org/abs/1809.04191">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>On periodic functions as regularizers for quantization of neural networks</th>
<th><a href="https://arxiv.org/abs/1811.09862">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Rethinking floating point for deep learning:fire:95</th>
<th><a href="https://arxiv.org/abs/1811.01721">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantizing deep convolutional networks for efficient inference: A whitepaper:fire:425</th>
<th><a href="https://arxiv.org/abs/1806.08342">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantization for rapid deployment of deep neural networks</th>
<th><a href="https://arxiv.org/abs/1810.05488">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Simultaneously optimizing weight and quantizer of ternary neural network using truncated gaussian approximation</th>
<th><a href="https://arxiv.org/abs/1810.01018">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Uniq: Uniform noise injection for non-uniform quantization of neural networks</th>
<th><a href="https://arxiv.org/abs/1804.10969">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Competitive Binary Neural Networks from Scratch</th>
<th><a href="https://arxiv.org/abs/1812.01965">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Joint Neural Architecture Search and Quantization</th>
<th><a href="https://arxiv.org/abs/1811.09426">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BinaryRelax: A Relaxation Approach For Training Deep Neural Networks With Quantized Weights:fire:55</th>
<th><a href="https://arxiv.org/pdf/1801.06313.pdf">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients:fire:1209<a href="https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net">[PyTorch]</a></th>
<th><a href="https://arxiv.org/pdf/1606.06160.pdf">arXiv 2018</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Espresso: Efficient Forward Propagation for BCNNs</th>
<th><a href="https://arxiv.org/pdf/1705.07175.pdf">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>CoDesign
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Mixed Precision Training:fire:601</th>
<th><a href="https://arxiv.org/pdf/1710.03740.pdf">arXiv 2018</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>PACT: Parameterized Clipping Activation for Quantized Neural Networks:fire:341</th>
<th><a href="https://arxiv.org/pdf/1805.06085.pdf">arXiv 2018</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binary Deep Neural Networks for Speech Recognition</th>
<th><a href="https://www.researchgate.net/publication/319185108_Binary_Deep_Neural_Networks_for_Speech_Recognition">Interspeech 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>On-Chip Memory Based Binarized Convolutional Deep Neural Network Applying Batch Normalization Free Technique on an FPGA</th>
<th><a href="https://ieeexplore.ieee.org/document/7965031">IPDPSW 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>CoDesign
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Ternary neural networks for resource-efficient AI applications</th>
<th><a href="https://arxiv.org/abs/1609.00222">IJCNN 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>A GPU-Outperforming FPGA Accelerator Architecture for Binary Convolutional Neural Networks</th>
<th><a href="https://arxiv.org/abs/1702.06392">DC 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>CoDesign
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning Accurate Low-Bit Deep Neural Networks with Stochastic Quantization<a href="https://github.com/dongyp13/Stochastic-Quantization">[Caffe]</a></th>
<th><a href="https://arxiv.org/abs/1708.01001">BMVC 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Terngrad: Ternary gradients to reduce communication in distributed deep learning:fire:649</th>
<th><a href="https://arxiv.org/abs/1705.07878">NeurIPS 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding:fire:696</th>
<th><a href="https://proceedings.neurips.cc/paper/2017/file/6c340f25839e6acdc73414517203f5f0-Paper.pdf">NeurIPS 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Gradient
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards Accurate Binary Convolutional Neural Network:fire:193<a href="https://github.com/layog/Accurate-Binary-Convolution-Network">[TensorFlow]</a>:star:49</th>
<th><a href="https://github.com/layog/Accurate-Binary-Convolution-Network">NeurIPS 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Quantized Nets: A Deeper Understanding:fire:134</th>
<th><a href="https://arxiv.org/pdf/1706.02379.pdf">NeurIPS 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Performance guaranteed network acceleration via high-order residual quantization</th>
<th><a href="https://arxiv.org/abs/1708.08687">ICCV 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binarized Convolutional Landmark Localizers for Human Pose Estimation and Face Alignment with Limited Resources:fire: 130<a href="https://github.com/1adrianb/binary-human-pose-estimation">[PyTorch]</a>:star:207</th>
<th><a href="https://arxiv.org/pdf/1603.05279.pdf">ICCV 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Performance Guaranteed Network Acceleration via High-Order Residual Quantization:fire:55</th>
<th><a href="https://arxiv.org/abs/1708.08687">ICCV 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fixed-point optimization of deep neural networks with adaptive step size retraining</th>
<th><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7952347&tag=1">ICASSP 2017</a></th>
<th>MP</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Learning with Low Precision by Half-wave Gaussian Quantization:fire:288<a href="https://github.com/zhaoweicai/hwgq">[Caffe]</a>:star:118</th>
<th><a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Cai_Deep_Learning_With_CVPR_2017_paper.pdf">CVPR 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fixed-point Factorized Networks</th>
<th><a href="https://arxiv.org/pdf/1611.01972.pdf">CVPR 2017</a></th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th>Factor
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Local Binary Convolutional Neural Networks:star:94:fire:156</th>
<th><a href="https://arxiv.org/abs/1608.06049">CVPR 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Network Sketching: Exploiting Binary Structure in Deep CNNs:fire:71</th>
<th><a href="https://arxiv.org/pdf/1706.02021.pdf">CVPR 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Weighted-Entropy-Based Quantization for Deep Neural Networks:fire:144</th>
<th><a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Park_Weighted-Entropy-Based_Quantization_for_CVPR_2017_paper.pdf">CVPR 2017</a></th>
<th></th>
<th>Non</th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights:fire:607<a href="https://github.com/AojunZhou/Incremental-Network-Quantization">[PyTorch]</a>:star:181</th>
<th><a href="https://arxiv.org/abs/1702.03044">ICLR 2017</a></th>
<th>T/Uni</th>
<th>Log</th>
<th>QAT</th>
<th>C</th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning Discrete Weights Using the Local Reparameterization Trick:fire:61</th>
<th><a href="https://arxiv.org/pdf/1710.07739.pdf">ICLR 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Stochastic
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Loss-aware Binarization of Deep Networks:fire:119<a href="https://github.com/houlu369/Loss-aware-Binarization">[PyTorch]</a>:star:18</th>
<th><a href="https://arxiv.org/pdf/1603.05279.pdf">ICLR 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Soft Weight-Sharing for Neural Network Compression:fire:222:star:18</th>
<th><a href="https://arxiv.org/abs/1702.04008">ICLR 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Towards the Limit of Network Quantization:fire:114</th>
<th><a href="https://arxiv.org/pdf/1612.01543.pdf">ICLR 2017</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>FINN: A Framework for Fast, Scalable Binarized Neural Network Inference:fire:463</th>
<th><a href="https://arxiv.org/abs/1612.07119">FPGA 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>How to train a compact binary neural network with high accuracy?:fire:205</th>
<th><a href="https://dl.acm.org/doi/abs/10.5555/3298483.3298617">AAAI 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Adaptive Quantization for Deep Neural Network:fire:67</th>
<th><a href="https://arxiv.org/pdf/1712.01048.pdf">AAAI 2017</a></th>
<th>MP</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>The high-dimensional geometry of binary neural networks</th>
<th><a href="https://openreview.net/forum?id=B1IDRdeCW">CoRR 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BMXNet: An Open-Source Binary Neural Network Implementation Based on MXNet</th>
<th><a href="https://arxiv.org/abs/1705.09864">CoRR 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep learning binary neural network on an FPGA</th>
<th><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8052915">arXiv 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>FP-BNN: Binarized neural network on FPGA:126:</th>
<th><a href="https://www.sciencedirect.com/science/article/pii/S0925231217315655">arXiv 2017</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>CoDesign
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Accelerating Deep Convolutional Networks using low-precision and sparsity:fire:111</th>
<th><a href="https://arxiv.org/pdf/1610.00324.pdf">arXiv 2017</a></th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Bit-regularized optimization of neural nets</th>
<th><a href="https://arxiv.org/pdf/1708.04788.pdf">arXiv 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Balanced Quantization: An Effective and Efficient Approach to Quantized Neural Networks</th>
<th><a href="https://arxiv.org/abs/1706.07145">arXiv 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Learning deep binary descriptor with multi-quantization:fire:97</th>
<th><a href="https://openaccess.thecvf.com/content_cvpr_2017/papers/Duan_Learning_Deep_Binary_CVPR_2017_paper.pdf">arXiv 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Gxnor-net: Training deep neural networks with ternary weights and activations without full-precision memory under a unified discretization framework</th>
<th><a href="https://arxiv.org/abs/1705.09283">arXiv 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Soft-to-hard vector quantization for end-to-end learning compressible representations</th>
<th><a href="https://arxiv.org/abs/1704.00648">arXiv 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>ShiftCNN: Generalized Low-Precision Architecture for Inference of Convolutional Neural Networks<a href="https://github.com/gudovskiy/ShiftCNN">[TensorFlow]</a>:star:53</th>
<th><a href="https://arxiv.org/abs/1706.02393">arXiv 2017</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Ternary Neural Networks with Fine-Grained Quantization:fire:71</th>
<th><a href="https://arxiv.org/abs/1705.01462">arXiv 2017</a></th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Trained Ternary Quantization:fire:734</th>
<th><a href="https://arxiv.org/pdf/1612.01064.pdf">arXiv 2017</a></th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Communication quantization for data-parallel training of deep neural networks:fire:130</th>
<th><a href="https://ieeexplore.ieee.org/document/7835789">MLHPC 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Decision making with quantized priors leads to discrimination</th>
<th><a href="https://ieeexplore.ieee.org/document/7605524">JPROC 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks:fire:3469<a href="https://github.com/allenai/XNOR-Net">[PyTorch]</a>:star:807</th>
<th><a href="https://arxiv.org/pdf/1603.05279.pdf">ECCV 2016</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Overcoming challenges in fixed point training of deep convolutional networks</th>
<th><a href="https://arxiv.org/abs/1607.02241">ICMLW 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fixed point quantization of deep convolutional networks:fire:696</th>
<th><a href="https://arxiv.org/abs/1511.06393">ICML 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding:fire:5045</th>
<th><a href="https://arxiv.org/pdf/1510.00149.pdf">CVPR 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantized convolutional neural networks for mobile devices:fire:270</th>
<th><a href="https://arxiv.org/abs/1512.06473">CVPR 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fixed-point Performance Analysis of Recurrent Neural Networks:fire:67</th>
<th><a href="https://arxiv.org/abs/1512.01322">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Qsgd: Randomized quantization for communication-optimal stochastic gradient descent:fire:801</th>
<th><a href="https://arxiv.org/abs/1610.02132">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Effective quantization methods for recurrent neural networks:fire:62</th>
<th><a href="https://arxiv.org/abs/1611.10176">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Sigma delta quantized networks</th>
<th><a href="https://arxiv.org/abs/1611.02024">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Recurrent neural networks with limited numerical precision:fire:65</th>
<th><a href="https://arxiv.org/abs/1608.06902">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training bit fully convolutional network for fast semantic segmentation</th>
<th><a href="https://arxiv.org/abs/1612.00212">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations:fire:1347</th>
<th><a href="https://arxiv.org/abs/1609.07061">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Convolutional neural networks using logarithmic data representation:fire:320</th>
<th><a href="https://arxiv.org/abs/1603.01025">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Layer normalization:fire:4125</th>
<th><a href="https://arxiv.org/abs/1607.06450">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binarized Neural Networks on the ImageNet Classification Task</th>
<th><a href="https://arxiv.org/pdf/1604.03058.pdf">arXiv 2016</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to +1 or −1:fire:1574<a href="https://github.com/itayhubara/BinaryNet">[PyTorch]</a>:star:252</th>
<th><a href="https://arxiv.org/abs/1602.02830">arXiv 2016</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep neural networks are robust to weight binarization and other non-linear distortions:fire:77</th>
<th><a href="https://arxiv.org/pdf/1606.01981.pdf">arXiv 2016</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Neural Networks with Few Multiplications:fire:258<a href="https://github.com/hantek/BinaryConnect">[PyTorch]</a>:star:81</th>
<th><a href="https://arxiv.org/pdf/1510.03009.pdf">arXiv 2016</a></th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Ternary weight networks:fire:647<a href="https://github.com/fengfu-chris/caffe-twns">[Caffe]</a>:star:63</th>
<th><a href="https://arxiv.org/pdf/1605.04711.pdf">arXiv 2016</a></th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Batch normalization: Accelerating deep network training by reducing internal covariate shift:fire:32893</th>
<th><a href="https://arxiv.org/abs/1502.03167">PMLR 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>BinaryConnect: Training Deep Neural Networks with binary weights during propagations:fire:2267<a href="https://github.com/MatthieuCourbariaux/BinaryConnect">[PyTorch]</a>:star:344</th>
<th><a href="https://arxiv.org/pdf/1511.00363.pdf">NeurIPS 2015</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Bitwise Neural Networks:fire:191</th>
<th><a href="https://arxiv.org/abs/1601.06071">ICML 2015</a></th>
<th>B</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Compressing neural networks with hashing trick:fire:887</th>
<th><a href="https://arxiv.org/pdf/1504.04788.pdf">ICML 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Deep Learning with Limited Numerical Precision:fire:1378</th>
<th><a href="https://arxiv.org/pdf/1502.02551.pdf">ICML 2015</a></th>
<th>Uni</th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fixed point optimization of deep convolutional neural networks for object recognition:fire:226</th>
<th><a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7178146">ICASSP 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>8-Bit Approximations for Parallelism in Deep Learning:fire:114</th>
<th><a href="https://arxiv.org/abs/1511.04561">ICLR 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training deep neural networks with low precision multiplications:fire:498</th>
<th><a href="https://arxiv.org/abs/1412.7024">ICLR 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Rounding methods for neural networks with low resolution synaptic weights</th>
<th><a href="https://arxiv.org/abs/1504.05767">arXiv 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Training Binary Multilayer Neural Networks for Image Classification using Expectation Backpropagation:fire:50</th>
<th><a href="https://arxiv.org/abs/1503.03562">arXiv 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Resiliency of Deep Neural Networks under quantizations:fire:123</th>
<th><a href="https://arxiv.org/abs/1511.06488">arXiv 2015</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Fixed-point feedforward deep neural network design using weights +1, 0, and −1:fire:269</th>
<th><a href="https://ieeexplore.ieee.org/document/6986082">SiPS 2014</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns:fire:679</th>
<th><a href="https://www.microsoft.com/en-us/research/publication/1-bit-stochastic-gradient-descent-and-application-to-data-parallel-distributed-training-of-speech-dnns/">Interspeech 2014</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Expectation Backpropagation: Parameter-Free Training of Multilayer Neural Networks with Continuous or Discrete Weights:fire:190</th>
<th><a href="https://proceedings.neurips.cc/paper/2014/file/076a0c97d09cf1a0ec3e19c7f2529f2b-Paper.pdf">NeurIPS 2014</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>Stochastic
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Compressing deep convolutional networks using vector quantization:fire:981</th>
<th><a href="https://arxiv.org/abs/1412.6115">arXiv 2014</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Lowrank matrix factorization for deep neural network training with high-dimensional output targets:fire:563</th>
<th><a href="https://ieeexplore.ieee.org/abstract/document/6638949/">ICASSP 2013</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Estimating or propagating gradients through stochastic neurons for conditional computation:fire:1346</th>
<th><a href="https://arxiv.org/abs/1308.3432">arXiv 2013</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>Product quantization for nearest neighbor search:fire:2268</th>
<th><a href="https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf">TPAMI 2010</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
<tbody>
<tr>
<th>An introduction to natural computation:fire:309</th>
<th><a href="https://mitpress.mit.edu/books/introduction-natural-computation">MITPress 1999</a></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>
</th>
</tr>
</tbody>
</table>
                
                
## Related Resources
[Awesome-Deep-Neural-Network-Compression](https://github.com/csyhhu/Awesome-Deep-Neural-Network-Compression/blob/master/Paper/Quantization.md) \
[awesome-model-quantization](https://github.com/htqin/awesome-model-quantization#awesome-model-quantization-) 

## LICENSE
The repo is released under the [MIT license](LICENSE).
