# All supported attacks

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Class Name</th>
      <th>Publication</th>
      <th>Paper (Open Access)</th>
    </tr>
  </thead>
  <tbody>
    <!-- Gradient-based attacks -->
    <tr>
      <th colspan="4">Gradient-based attacks</th>
    </tr>
    <tr>
      <td><a href="./fgsm">FGSM</a></td>
      <td><code>FGSM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2015-62B959?labelColor=2D3339" alt="ICLR 2015"></td>
      <td><a href="https://arxiv.org/abs/1412.6572">Explaining and Harnessing Adversarial Examples</a></td>
    </tr>
    <tr>
      <td><a href="./pgd">PGD</a></td>
      <td><code>PGD</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2018-62B959?labelColor=2D3339" alt="ICLR 2018"></td>
      <td><a href="https://arxiv.org/abs/1706.06083">Towards Deep Learning Models Resistant to Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./pgdl2">PGD (L2)</a></td>
      <td><code>PGDL2</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2018-62B959?labelColor=2D3339" alt="ICLR 2018"></td>
      <td><a href="https://arxiv.org/abs/1706.06083">Towards Deep Learning Models Resistant to Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./mifgsm">MI-FGSM</a></td>
      <td><code>MIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2018-1A407F?labelColor=2D3339" alt="CVPR 2018"></td>
      <td><a href="https://arxiv.org/abs/1710.06081">Boosting Adversarial Attacks with Momentum</a></td>
    </tr>
    <tr>
      <td><a href="./difgsm">DI-FGSM</a></td>
      <td><code>DIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2019-1A407F?labelColor=2D3339" alt="CVPR 2019"></td>
      <td><a href="https://arxiv.org/abs/1803.06978">Improving Transferability of Adversarial Examples with Input Diversity</a></td>
    </tr>
    <tr>
      <td><a href="./tifgsm">TI-FGSM</a></td>
      <td><code>TIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2019-1A407F?labelColor=2D3339" alt="CVPR 2019"></td>
      <td><a href="https://arxiv.org/abs/1904.02884">Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./nifgsm">NI-FGSM</a></td>
      <td><code>NIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2020-62B959?labelColor=2D3339" alt="ICLR 2020"></td>
      <td><a href="https://arxiv.org/abs/1908.06281">Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./sinifgsm">SI-NI-FGSM</a></td>
      <td><code>SINIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2020-62B959?labelColor=2D3339" alt="ICLR 2020"></td>
      <td><a href="https://arxiv.org/abs/1908.06281">Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./dr">DR</a></td>
      <td><code>DR</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2020-1A407F?labelColor=2D3339" alt="CVPR 2020"></td>
      <td><a href="https://arxiv.org/abs/1911.11616">Enhancing Cross-Task Black-Box Transferability of Adversarial Examples With Dispersion Reduction</a></td>
    </tr>
    <tr>
      <td><a href="./vmifgsm">VMI-FGSM</a></td>
      <td><code>VMIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2021-1A407F?labelColor=2D3339" alt="CVPR 2021"></td>
      <td><a href="https://arxiv.org/abs/2103.15571">Enhancing the Transferability of Adversarial Attacks through Variance Tuning</a></td>
    </tr>
    <tr>
      <td><a href="./vnifgsm">VNI-FGSM</a></td>
      <td><code>VNIFGSM</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2021-1A407F?labelColor=2D3339" alt="CVPR 2021"></td>
      <td><a href="https://arxiv.org/abs/2103.15571">Enhancing the Transferability of Adversarial Attacks through Variance Tuning</a></td>
    </tr>
    <tr>
      <td><a href="./admix">Admix</a></td>
      <td><code>Admix</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2021-5A428D?labelColor=2D3339" alt="ICCV 2021"></td>
      <td><a href="https://arxiv.org/abs/2102.00436">Admix: Enhancing the Transferability of Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./fia">FIA</a></td>
      <td><code>FIA</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2021-5A428D?labelColor=2D3339" alt="ICCV 2021"></td>
      <td><a href="https://arxiv.org/abs/2107.14185">Feature Importance-aware Transferable Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./pnapatchout">PNA-PatchOut</a></td>
      <td><code>PNAPatchOut</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2022-C8172C?labelColor=2D3339" alt="AAAI 2022"></td>
      <td><a href="https://arxiv.org/abs/2109.04176">Towards Transferable Adversarial Attacks on Vision Transformers</a></td>
    </tr>
    <tr>
      <td><a href="./naa">NAA</a></td>
      <td><code>NAA</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2022-1A407F?labelColor=2D3339" alt="CVPR 2022"></td>
      <td><a href="https://arxiv.org/abs/2204.00008">Improving Adversarial Transferability via Neuron Attribution-Based Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./ssa">SSA</a></td>
      <td><code>SSA</code></td>
      <td><img src="https://img.shields.io/badge/ECCV-2022-E16B4C?labelColor=2D3339" alt="ECCV 2022"></td>
      <td><a href="https://arxiv.org/abs/2207.05382">Frequency Domain Model Augmentation for Adversarial Attack</a></td>
    </tr>
    <tr>
      <td><a href="./tgr">TGR</a></td>
      <td><code>TGR</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2023-1A407F?labelColor=2D3339" alt="CVPR 2023"></td>
      <td><a href="https://arxiv.org/abs/2303.15754">Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization</a></td>
    </tr>
    <tr>
      <td><a href="./ilpd">ILPD</a></td>
      <td><code>ILPD</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2023-654287?labelColor=2D3339" alt="NeurIPS 2023"></td>
      <td><a href="https://arxiv.org/abs/2304.13410">Improving Adversarial Transferability via Intermediate-level Perturbation Decay</a></td>
    </tr>
    <tr>
      <td><a href="./mig">MIG</a></td>
      <td><code>MIG</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2023-5A428D?labelColor=2D3339" alt="ICCV 2023"></td>
      <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html">Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients</a></td>
    </tr>
    <tr>
      <td><a href="./gra">GRA</a></td>
      <td><code>GRA</code></td>
      <td><img src="https://img.shields.io/badge/ICCV-2023-5A428D?labelColor=2D3339" alt="ICCV 2023"></td>
      <td><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Boosting_Adversarial_Transferability_via_Gradient_Relevance_Attack_ICCV_2023_paper.html">Boosting Adversarial Transferability via Gradient Relevance Attack</a></td>
    </tr>
    <tr>
      <td><a href="./decowa">DeCoWA</a></td>
      <td><code>DeCoWA</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2024-C8172C?labelColor=2D3339" alt="AAAI 2024"></td>
      <td><a href="https://arxiv.org/abs/2402.03951">Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping</a></td>
    </tr>
    <tr>
      <td><a href="./vdc">VDC</a></td>
      <td><code>VDC</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2024-C8172C?labelColor=2D3339" alt="AAAI 2024"></td>
      <td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/28541">Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection</a></td>
    </tr>
    <tr>
      <td><a href="./bsr">BSR</a></td>
      <td><code>BSR</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2024-1A407F?labelColor=2D3339" alt="CVPR 2024"></td>
      <td><a href="https://arxiv.org/abs/2308.10299">Boosting Adversarial Transferability by Block Shuffle and Rotation</a></td>
    </tr>
    <tr>
      <td><a href="./l2t">L2T</a></td>
      <td><code>L2T</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2024-1A407F?labelColor=2D3339" alt="CVPR 2024"></td>
      <td><a href="https://arxiv.org/abs/2405.14077">Learning to Transform Dynamically for Better Adversarial Transferability</a></td>
    </tr>
    <tr>
      <td><a href="./att">ATT</a></td>
      <td><code>ATT</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2024-654287?labelColor=2D3339" alt="NeurIPS 2024"></td>
      <td><a href="https://openreview.net/forum?id=sNz7tptCH6">Boosting the Transferability of Adversarial Attack on Vision Transformer with Adaptive Token Tuning</a></td>
    </tr>
    <tr>
      <td><a href="./bfa">BFA</a></td>
      <td><code>BFA</code></td>
      <td><img src="https://img.shields.io/badge/Neurocomputing-cfb225" alt="Neurocomputing' 2024"></td>
      <td><a href="https://www.sciencedirect.com/science/article/pii/S0925231224006349">Improving the transferability of adversarial examples through black-box feature attacks</a></td>
    </tr>
    <tr>
      <td><a href="./mumodig">MuMoDIG</a></td>
      <td><code>MuMoDIG</code></td>
      <td><img src="https://img.shields.io/badge/AAAI-2025-C8172C?labelColor=2D3339" alt="AAAI 2025"></td>
      <td><a href="https://www.arxiv.org/abs/2412.18844">Improving Integrated Gradient-based Transferable Adversarial Examples by Refining the Integration Path</a></td>
    </tr>
    <!-- Generative attacks -->
    <tr>
      <th colspan="4">Generative attacks</th>
    </tr>
    <tr>
      <td><a href="./cda">CDA</a></td>
      <td><code>CDA</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2019-654287?labelColor=2D3339" alt="NeurIPS 2019"></td>
      <td><a href="https://arxiv.org/abs/1905.11736">Cross-Domain Transferability of Adversarial Perturbations</a></td>
    </tr>
    <tr>
      <td><a href="./ltp">LTP</a></td>
      <td><code>LTP</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2021-654287?labelColor=2D3339" alt="NeurIPS 2021"></td>
      <td><a href="https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html">Learning Transferable Adversarial Perturbations</a></td>
    </tr>
    <tr>
      <td><a href="./bia">BIA</a></td>
      <td><code>BIA</code></td>
      <td><img src="https://img.shields.io/badge/ICLR-2022-62B959?labelColor=2D3339" alt="ICLR 2022"></td>
      <td><a href="https://arxiv.org/abs/2201.11528">Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains</a></td>
    </tr>
    <tr>
      <td><a href="./gama">GAMA</a></td>
      <td><code>GAMA</code></td>
      <td><img src="https://img.shields.io/badge/NeurIPS-2022-654287?labelColor=2D3339" alt="NeurIPS 2022"></td>
      <td><a href="https://arxiv.org/abs/2209.09502">GAMA: Generative Adversarial Multi-Object Scene Attacks</a></td>
    </tr>
    <!-- Others -->
    <tr>
      <th colspan="4">Others</th>
    </tr>
    <tr>
      <td><a href="./deepfool">DeepFool</a></td>
      <td><code>DeepFool</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2016-1A407F?labelColor=2D3339" alt="CVPR 2016"></td>
      <td><a href="https://arxiv.org/abs/1511.04599">DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks</a></td>
    </tr>
    <tr>
      <td><a href="./geoda">GeoDA</a></td>
      <td><code>GeoDA</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2020-1A407F?labelColor=2D3339" alt="CVPR 2020"></td>
      <td><a href="https://arxiv.org/abs/2003.06468">GeoDA: A Geometric Framework for Black-box Adversarial Attacks</a></td>
    </tr>
    <tr>
      <td><a href="./ssp">SSP</a></td>
      <td><code>SSP</code></td>
      <td><img src="https://img.shields.io/badge/CVPR-2020-1A407F?labelColor=2D3339" alt="CVPR 2020"></td>
      <td><a href="https://arxiv.org/abs/2006.04924">A Self-supervised Approach for Adversarial Robustness</a></td>
    </tr>
  </tbody>
</table>
