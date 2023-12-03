# SODAS-Net: Side-infOrmation-aided Deep Adaptive Shrinkage Network for Compressive Sensing (IEEE TIM, 2023)
This repository is for SODAS-Net introduced in the following paper：

[Jiechong Song](https://scholar.google.com/citations?hl=en&user=EBOtupAAAAAJ), [Jian Zhang](http://jianzhang.tech/), "SODAS-Net: Side-infOrmation-aided Deep Adaptive Shrinkage Network for Compressive Sensing", IEEE Transactions on Instrumentation and Measurement (TIM), 2023. [PDF](https://ieeexplore.ieee.org/document/10217074)

## :art: Abstract

As a kind of network structure increasingly studied in compressive sensing (CS), deep unfolding networks (DUNs), which unroll the iterative reconstruction procedure as deep neural networks (DNNs) for end-to-end training, have high interpretability and remarkable performance. Every phase of the DUN corresponds to one iteration. The input and output of each phase in most DUNs are inherently images, which heavily restricts information transmission. Besides, existing DUNs unfolded by ℓ1 -regularized optimization usually utilize fixed thresholds for soft-shrinkage operation, which lacks adaptability. To solve these issues, a novel side-information-aided deep adaptive shrinkage network (SODAS-Net) is designed for CS. Utilizing the side information (SI) allows SODAS-Net to send large volumes of information between adjacent phases, substantially augmenting the network representation capacity and optimizing network performance. Furthermore, an effective adaptive soft-shrinkage strategy is developed, which enables our SODAS-Net to solve ℓ1 -regularized proximal mapping with content-aware thresholds. The results from extensive experiments on various testing datasets demonstrate that SODAS-Net achieves superior performance.

## :fire: Network Architecture
<span style="display:block;text-align:center">![Network](/Figs/network.png)</span>

## 🔧 Requirements
- Python == 3.8.5
- Pytorch == 1.8.0

## 🚩 Results
![Network](/Figs/result.png)

## 👀 Datasets
- Train data: [train400](https://drive.google.com/file/d/15FatS3wYupcoJq44jxwkm6Kdr0rATPd0/view?usp=sharing)
- Test data: Set11, [CBSD68](https://drive.google.com/file/d/1Q_tcV0d8bPU5g0lNhVSZXLFw0whFl8Nt/view?usp=sharing), [Urban100](https://drive.google.com/file/d/1cmYjEJlR2S6cqrPq8oQm3tF9lO2sU0gV/view?usp=sharing)

## :computer: Command
### Test
`python TEST_CS_SODAS.py --cs_ratio 10/25/30/40/50 --test_name Set11/CBSD68/Urban100`

## 📑 Citation
If you find our work helpful in your resarch or work, please cite the following paper.

```
@article{song2023sodas,
  title={SODAS-Net: Side-infOrmation-aided Deep Adaptive Shrinkage Network for Compressive Sensing},
  author={Song, Jiechong and Zhang, Jian},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
```

## :e-mail: Contact
If you have any question, please email `songjiechong@pku.edu.cn`.

## :hugs: Acknowledgements
This code is built on [ISTA-Net-PyTorch](https://github.com/jianzhangcs/ISTA-Net-PyTorch). We thank the authors for sharing their codes.

