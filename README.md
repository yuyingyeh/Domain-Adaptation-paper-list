# Domain Adaptation / Transfer learning / Semi-supervised paper list
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

List recent papers related to domain adaptation in different type of applications. Some are referred from [Awesome Domain Adaptation](https://github.com/zhaoxin94/awsome-domain-adaptation/blob/master/README.md).

## Theory
- [An analytic theory of generalization dynamics and transfer learning in deep linear networks](https://openreview.net/forum?id=ryfMLoCqtQ) (ICLR'19)

## Classification
- [Contrastive Adaptation Network for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) (CVPR'19)
- [Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1903.04064.pdf) (CVPR'19)
- [AdaGraph: Unifying Predictive and Continuous Domain Adaptation through Graphs](https://arxiv.org/pdf/1903.07062.pdf) (CVPR'19)
- [Learning to Transfer Examples for Partial Domain Adaptation](https://arxiv.org/pdf/1903.12230.pdf) (CVPR'19)
- [Domain-Symmetric Networks for Adversarial Domain Adaptation](https://arxiv.org/pdf/1904.04663.pdf) (CVPR'19)
- [Transferability vs. Discriminability: Batch Spectral Penalization for Adversarial Domain Adaptation](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) (ICML'19)
- [Learning What and Where to Transfer](http://proceedings.mlr.press/v97/jang19b/jang19b.pdf) (ICML'19)
- [Bridging Theory and Algorithm for Domain Adaptation](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) (ICML'19)
- [On Learning Invariant Representations for Domain Adaptation](http://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf) (ICML'19)
- [Augmented Cyclic Adversarial Learning for Low Resource Domain Adaptation](https://openreview.net/forum?id=B1G9doA9F7) (ICLR'19)
- [Regularized Learning for Domain Adaptation under Label Shifts](https://openreview.net/forum?id=rJl0r3R9KX) (ICLR'19)
- [Improving the Generalization of Adversarial Training with Domain Adaptation](https://openreview.net/forum?id=SyfIfnC5Ym) (ICLR'19)
- [Multi-Domain Adversarial Learning](https://openreview.net/forum?id=Sklv5iRqYX) (ICLR'19)
- [Learning Factorized Representations for Open-set Domain Adaptation](https://openreview.net/forum?id=SJe3HiC5KX) (ICLR'19)
- [Transferable attention for domain adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/transferable-attention-aaai19.pdf) (AAAI'19): attention mechanism
- [Co-regularized Alignment for Unsupervised Domain Adaptation](http://papers.nips.cc/paper/8146-co-regularized-alignment-for-unsupervised-domain-adaptation.pdf) (NIPS'18)
- [Graph Adaptive Knowledge Transfer for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhengming_Ding_Graph_Adaptive_Knowledge_ECCV_2018_paper.pdf) (ECCV'18)
- [Deep Adversarial Attention Alignment for Unsupervised Domain Adaptation:the Benefit of Target Expectation Maximization](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guoliang_Kang_Deep_Adversarial_Attention_ECCV_2018_paper.pdf) (ECCV'18)
- [DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf) (ECCV'18): Optimal transport, Wasserstein distance
- [Maximum Classifier Discrepancy for Unsupervised Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) (CVPR'18 Oral): Discriminative domain invariant feature, Ensemble-based DA
- [Generate To Adapt: Aligning Domains using Generative Adversarial Networks](https://arxiv.org/pdf/1704.01705.pdf) (CVPR'18 Spotlight)
- [Collaborative and Adversarial Network for Unsupervised domain adaptation](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1410.pdf) (CVPR'18): Adversarial-based DA
- [Unsupervised Domain Adaptation with Similarity Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pinheiro_Unsupervised_Domain_Adaptation_CVPR_2018_paper.pdf) (CVPR'18)
- [Adversarial Dropout Regularization](https://openreview.net/forum?id=HJIoJWZCZ) (ICLR'18): Discriminative domain invariant feature, Ensemble-based DA
- [Self-ensembling for visual domain adaptation](https://openreview.net/forum?id=rkpoTaxA-) (ICLR'18): Ensemble-based DA
- [Adaptive Batch Normalization for practical domain adaptation](http://winsty.net/papers/adabn.pdf) (Pattern Recognition'18): normalized-based method
- [Multi-Adversarial Domain Adaptation](https://arxiv.org/abs/1809.02176) (AAAI'18 Oral): class-aware domain discrepancy, discriminative domain invariant feature
- [Label efficient learning of transferable representations acrosss domains and tasks](https://papers.nips.cc/paper/6621-label-efficient-learning-of-transferable-representations-acrosss-domains-and-tasks.pdf) (NIPS'17): entropy-based DA
- [Joint distribution optimal transportation for domainadaptation](https://arxiv.org/pdf/1705.08848.pdf) (NIPS'17): Optimal transport, Wasserstein distance
- [Associative Domain Adaptation](https://arxiv.org/pdf/1708.00938.pdf) (ICCV'17): Discriminative domain invariant feature
- [Adversarial discriminative domain adaptation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf) (CVPR'17)
- [Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/pdf/1612.05424.pdf) (CVPR'17)
- [Mind the Class Weight Bias: Weighted Maximum Mean Discrepancyfor Unsupervised Domain Adaptation](https://zpascal.net/cvpr2017/Yan_Mind_the_Class_CVPR_2017_paper.pdf) (CVPR'17): MMD based method
- [Deep Transfer Learning with Joint Adaptation Networks](https://arxiv.org/pdf/1605.06636.pdf) (ICML'17): JMMD
- [Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning](https://arxiv.org/abs/1702.08811) (ICLR'17): MMD based method
- [Revisiting Batch Normalization For Practical Domain Adaptation](https://openreview.net/forum?id=Hk6dkJQFx) (ICLR'17 workshop): normalization-based method
- [Coupled Generative Adversarial Networks](https://arxiv.org/pdf/1606.07536.pdf) (NIPS'16)
- [Learning Transferrable Representations for Unsupervised Domain Adaptation](https://papers.nips.cc/paper/6360-learning-transferrable-representations-for-unsupervised-domain-adaptation.pdf) (NIPS'16): Discriminative domain invariant feature
- [Unsupervised Domain Adaptation with ResidualTransfer Networks](https://papers.nips.cc/paper/6110-unsupervised-domain-adaptation-with-residual-transfer-networks.pdf) (NIPS'16): MMD based method
- [Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://arxiv.org/pdf/1607.01719.pdf) (ECCV'16 workshop)
- [Domain adversarial training of neural networks](http://jmlr.org/papers/volume17/15-239/15-239.pdf) (JMLR'16)
- [Optimal Transport for Domain Adaptation](https://arxiv.org/pdf/1507.00504.pdf) (TPAMI'16): Optimal transport, Wasserstein distance
- [Simultaneous deep transfer across domains and tasks](https://people.eecs.berkeley.edu/~jhoffman/papers/Tzeng_ICCV2015.pdf) (CVPR'15)
- [Learning transferable features with deep adaptation networks](http://proceedings.mlr.press/v37/long15.pdf) (ICML'15): MMD
- [Deep Domain Confusion: Maximizing for Domain Invariance](https://arxiv.org/pdf/1412.3474.pdf) (ArXiv'14): MMD
- [Transfer Feature Learning with Joint Distribution Adaptation](http://ise.thss.tsinghua.edu.cn/~mlong/doc/joint-distribution-adaptation-iccv13.pdf) (ICCV'13): class-aware domain discrepancy

#### Open-Set DA
- [Learning Factorized Representations for Open-Set Domain Adaptation](https://openreview.net/pdf?id=SJe3HiC5KX) (ICLR'19)
- [Open Set Domain Adaptation by Backpropagation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.pdf) (ECCV'18)

#### Partial DA
Target domain classes are subset of source domain
- [Partial Adversarial Domain Adaptation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhangjie_Cao_Partial_Adversarial_Domain_ECCV_2018_paper.pdf) (ECCV'18)

#### Domain Generalization
Labeled source domain data only, no unlabeled/labeled target domain data
- [Deep Domain Generalization via Conditional Invariant Adversarial Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf) (ECCV'18)

#### Video Classification
Human action recognition
- [Temporal Attentive Alignment for Large-Scale Video Domain Adaptation](https://arxiv.org/pdf/1907.12743.pdf) (ICCV'19 Oral)
- [Learning Transferable Self-attentive Representations for Action Recognition in Untrimmed Videos with Weak Supervision](https://arxiv.org/pdf/1902.07370.pdf) (AAAI'19)
- [Deep domain adaptation in action space](http://bmvc2018.org/contents/papers/0960.pdf) (BMVC'18)
- [Dual many-to-one-encoder-based transfer learning for cross-dataset human action recognition](https://www.sciencedirect.com/science/article/pii/S0262885616000020) (Image and Vision Computing'16)
- [Human action recognition acrossdatasets by foreground-weighted histogram decomposition](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6909498) (CVPR'14)

Face Recognition
- [Unsupervised Domain Adaptation for Face Recognition in Unlabeled Videos](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/iccv17_videoface_camera.pdf) (ICCV'17)

## Semantic Segmentation
### Unsupervised Domain Adaptation
- [Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation](https://arxiv.org/pdf/1901.05946.pdf) (ICCV'19)
- [DLOW: Domain Flow for Adaptation and Generalization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_DLOW_Domain_Flow_for_Adaptation_and_Generalization_CVPR_2019_paper.pdf) (CVPR'19 Oral)
- [ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf) (CVPR'19)
- [Learning Semantic Segmentation from Synthetic Data: A Geometrically Guided Input-Output Adaptation Approach](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Semantic_Segmentation_From_Synthetic_Data_A_Geometrically_Guided_Input-Output_CVPR_2019_paper.pdf) (CVPR'19)
- [Bidirectional Learning for Domain Adaptation of Semantic Segmentation](https://arxiv.org/pdf/1904.10620v1.pdf) (CVPR'19)
- [CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency](https://filebox.ece.vt.edu/~jbhuang/papers/CVPR_2019_CrDoCo.pdf) (CVPR'19)
- [Taking A Closer Look at Domain Shift: Category-level Adversaries for Semantics Consistent Domain Adaptation](https://arxiv.org/pdf/1809.09478.pdf) (CVPR'19)
- [All about Structure: Adapting Structural Information across Domains for Boosting Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_All_About_Structure_Adapting_Structural_Information_Across_Domains_for_Boosting_CVPR_2019_paper.pdf) (CVPR'19)
- [Weakly Supervised Adversarial Domain Adaptation for Semantic Segmentation in Urban Scenes](https://arxiv.org/abs/1904.09092v1) (TIP)
- [SPIGAN: Privileged Adversarial Learning from Simulation](https://openreview.net/forum?id=rkxoNnC5FQ) (ICLR'19)
- [Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yang_Zou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf) (ECCV'18)
- [Penalizing Top Performers: Conservative Loss for Semantic Segmentation Adaptation](https://arxiv.org/abs/1809.00903) (ECCV'18)
- [Domain transfer through deep activation matching](http://openaccess.thecvf.com/content_ECCV_2018/papers/Haoshuo_Huang_Domain_transfer_through_ECCV_2018_paper.pdf) (ECCV'18)
- [Cycada: Cycle-consistent adversarial domain adaptation](http://proceedings.mlr.press/v80/hoffman18a/hoffman18a.pdf) (ICML'18)
- [Learning to adapt structured output space for semantic segmentation](http://faculty.ucmerced.edu/mhyang/papers/cvpr2018_semantic_segmentation.pdf) (CVPR'18 Spotlight)
- [Learning from synthetic data: Addressing domain shift for semantic segmentation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sankaranarayanan_Learning_From_Synthetic_CVPR_2018_paper.pdf) (CVPR'18 Spotlight)
- [ROAD: Reality Oriented Adaptation for Semantic Segmentation of Urban Scenes](https://arxiv.org/abs/1711.11556) (CVPR'18)
- [Conditional Generative Adversarial Network for Structured Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.pdf) (CVPR'18)
- [Image to Image Translation for Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Murez_Image_to_Image_CVPR_2018_paper.pdf) (CVPR'18)
- [Curriculum Domain Adaptation for Semantic Segmentation of Urban Scenes](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_Curriculum_Domain_Adaptation_ICCV_2017_paper.pdf) (ICCV'17)
- [FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation](https://arxiv.org/pdf/1612.02649.pdf) (ArXiv'16)

### Weakly supervised domain adaptation
Abundant source domain data, limited target domain data
- [Not All Areas Are Equal: Transfer Learning for Semantic Segmentation via Hierarchical Region Selection](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Not_All_Areas_Are_Equal_Transfer_Learning_for_Semantic_Segmentation_CVPR_2019_paper.pdf) (CVPR'19 Oral)

## Instance Segmentation
- [SRDA: Generating Instance Segmentation Annotation Via Scanning, Reasoning And Domain Adaptation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wenqiang_Xu_SRDA_Generating_Instance_ECCV_2018_paper.pdf) (ECCV'18)

## Feature Disentanglement
- [A Unified Feature Disentangler for Multi-Domain Image Translation and Manipulation](https://papers.nips.cc/paper/7525-a-unified-feature-disentangler-for-multi-domain-image-translation-and-manipulation.pdf) (NIPS'18)
- [Detach and Adapt: Learning Cross-Domain Disentangled Deep Representation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Detach_and_Adapt_CVPR_2018_paper.pdf) (CVPR'18 Spotlight)

## Person Re-identification
- [Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification](https://arxiv.org/pdf/1904.01990.pdf) (CVPR'19)
- [Domain Adaptation through Synthesis for Unsupervised Person Re-identification](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf) (ECCV'18)
- [Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf) (CVPR'18)
- [Adaptation and Re-Identification Network: An Unsupervised Deep Transfer Learning Approach to Person Re-Identification](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Li_Adaptation_and_Re-Identification_CVPR_2018_paper.pdf) (CVPRW'18)

## Depth Estimation
- [Geometry-Aware Symmetric Domain Adaptation for Monocular Depth Estimation](https://arxiv.org/pdf/1904.01870.pdf) (CVPR'19)
- [T2net: Synthetic-to-realistic translation for solving single-image depth estimation tasks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chuanxia_Zheng_T2Net_Synthetic-to-Realistic_Translation_ECCV_2018_paper.pdf) (ECCV'18)
- [AdaDepth: Unsupervised Content Congruent Adaptation for Depth Estimation](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2583.pdf) (CVPR'18)
- [Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation via Image Style Transfer](http://openaccess.thecvf.com/content_cvpr_2018/papers/Atapour-Abarghouei_Real-Time_Monocular_Depth_CVPR_2018_paper.pdf) (CVPR'18)

## Stereo
- [Learning to Adapt for Stereo](https://arxiv.org/pdf/1904.02957.pdf) (CVPR'19)
- [Learning Monocular Depth by Distilling Cross-domain Stereo Networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaoyang_Guo_Learning_Monocular_Depth_ECCV_2018_paper.pdf) (ECCV'18)
- [Unsupervised adaptation for deep stereo](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tonioni_Unsupervised_Adaptation_for_ICCV_2017_paper.pdf) (ICCV'17)

## Object Detection
- [Automatic adaptation of object detectors to new domains using self-training](https://arxiv.org/pdf/1904.07305.pdf) (CVPR'19)
- [AugGAN: Cross Domain Adaptation with GAN-based Data Augmentation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sheng-Wei_Huang_AugGAN_Cross_Domain_ECCV_2018_paper.pdf) (ECCV'18)

## Reinforcement Learning
- [Sim-Real Joint Reinforcement Transfer for 3D Indoor Navigation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Sim-Real_Joint_Reinforcement_Transfer_for_3D_Indoor_Navigation_CVPR_2019_paper.pdf) (CVPR'19)

## Multi-Domain
- [Multi-Domain Adversarial Learning](https://openreview.net/forum?id=Sklv5iRqYX) (ICLR'19)

## Zero-shot Learning
- [Zero-Shot Deep Domain Adaptation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kuan-Chuan_Peng_Zero-Shot_Deep_Domain_ECCV_2018_paper.pdf) (ECCV'18)

## Other

#### Scene Text Detection and Recognition
- [GA-DAN: Geometry-Aware Domain Adaptation Network for Scene Text Detection and Recognition](https://arxiv.org/pdf/1907.09653.pdf) (ICCV'19)

#### Adversarial Attack
- [Improving the Generalization of Adversarial Training with Domain Adaptation](https://openreview.net/forum?id=SyfIfnC5Ym) (ICLR'19)

#### Brain-Machine Interfaces
- [Adversarial Domain Adaptation for Stable Brain-Machine Interfaces](https://openreview.net/forum?id=Hyx6Bi0qYm) (ICLR'19)

#### Metric Learning
- [Unsupervised Domain Adaptation for Distance Metric Learning](https://openreview.net/forum?id=BklhAj09K7) (ICLR'19)

#### 3D Keypoint Estimation
- [Unsupervised Domain Adaptation for 3D Keypoint Estimation via View Consistency](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xingyi_Zhou_Unsupervised_Domain_Adaptation_ECCV_2018_paper.pdf) (ECCV'18)

#### Museum Exhibit Identification
- [Museum Exhibit Identification Challenge for the Supervised Domain Adaptation and Beyond](http://openaccess.thecvf.com/content_ECCV_2018/papers/Piotr_Koniusz_Museum_Exhibit_Identification_ECCV_2018_paper.pdf) (ECCV'18)

#### Autonomous Driving
- [Real-to-Virtual Domain Unification for End-to-End Autonomous Driving](http://openaccess.thecvf.com/content_ECCV_2018/papers/Luona_Yang_Real-to-Virtual_Domain_Uni_ECCV_2018_paper.pdf) (ECCV'18)

#### Gaze estimation and human hand pose estimation
- [Learning from Simulated and Unsupervised Images through Adversarial Training](http://openaccess.thecvf.com/content_cvpr_2017/papers/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.pdf) (CVPR'17)


# Semi-supervised Learning
## Classification
- [Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning](https://arxiv.org/pdf/1704.03976.pdf) (TPAMI'18)
- [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://papers.nips.cc/paper/7585-realistic-evaluation-of-deep-semi-supervised-learning-algorithms.pdf) (NIPS'18)
- [mixup: Beyond Empirical Risk Minimization](https://openreview.net/forum?id=r1Ddp1-Rb) (ICLR'18)
- [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/pdf/1606.01583.pdf) (ICML'16 workshop)
- [Semi-supervised Learningby Entropy Minimization](http://papers.nips.cc/paper/2740-semi-supervised-learning-by-entropy-minimization.pdf) (NIPS'04)

## Semantic Segmentation
- [Adversarial Learning for Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/1802.07934) (BMVC'18)

