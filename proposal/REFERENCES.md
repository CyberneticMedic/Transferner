# Transferner — verified references (IEEE format)

All entries were checked on **2026-10-18** against the primary source listed.
In the compiled PDF, `IEEEtran.bst` numbers them in citation order and uses
"First Author *et al.*" for more than 6 authors. Only cited entries print.
`references.bib` is the source of truth.

**Cited in the 2-page proposal (21):** Long 2023 (app accuracy) · Noor 2022 (toxic-plant CNN) ·
Pl@ntNet observations/GBIF · TreeOfLife-200M · NC State Plant Toolbox · Pl@ntNet-300K (long tail) ·
Holly 2026 (Fig. 1 photo) · BioCLIP · BioCLIP 2 · BioCLIP 2.5 Huge · MobileNetV4 · Kornblith 2019 (transfer) ·
LoRA · CLIP-LoRA · class-balanced loss · Jacob 2018 (INT8) · Guo 2017 (calibration) · Elkan 2001 (cost-sensitive) ·
Geifman 2017 (abstention) · ROCm system requirements · LiteRT NPU.
Everything else below stays in `references.bib` for the final report.

- H. Holly, "Wild carrot vs poison hemlock – A deep dive," *The Northwest Forager* (blog), Jan. 2026. [Online]. Available: https://thenorthwestforager.com/2026/01/02/wild-carrot-vs-poison-hemlock/ (accessed Oct. 18, 2026). Photo © 2015 H. Holly, all rights reserved: unofficial proposal use only, do not redistribute.

## Motivation: toxic plants and plant-ID apps
- J. Otter, S. Mayer, and C. A. Tomaszewski, "Swipe right: A comparison of accuracy of plant identification apps for toxic plants," *J. Med. Toxicol.*, vol. 17, no. 1, pp. 42–47, 2021, doi: 10.1007/s13181-020-00803-6. — Crossref
- K. Long *et al.*, "Plant identification applications do not reliably identify toxic and edible plants in the American Midwest," *Clin. Toxicol.*, vol. 61, no. 7, pp. 524–528, 2023, doi: 10.1080/15563650.2023.2237282. — Crossref
- I. McNeill, R. Knoeckel, A. Goggin, C. Wylie, and K. Isoardi, "Accuracy of plant identification applications to identify plants in suspected poisoning cases referred to the Queensland Poisons Information Centre," *Toxicol. Commun.*, vol. 8, no. 1, Art. no. 2377523, 2024, doi: 10.1080/24734306.2024.2377523. — Crossref
- M. C. Beuhler *et al.*, "2024 Annual report of the National Poison Data System (NPDS) from America's Poison Centers: 42nd annual report," *Clin. Toxicol.*, vol. 63, no. 12, pp. 1029–1280, 2025, doi: 10.1080/15563650.2025.2571299. — Crossref
- T. H. Noor, A. Noor, and M. Elmezain, "Poisonous plants species prediction using a convolutional neural network and support vector machine hybrid model," *Electronics*, vol. 11, no. 22, Art. no. 3690, 2022, doi: 10.3390/electronics11223690. — Crossref
- R. H. Hridoy, F. Akter, and M. Afroz, "An efficient computer vision approach for rapid recognition of poisonous plants by classifying leaf images using transfer learning," in *Proc. 12th Int. Conf. Comput. Commun. Netw. Technol. (ICCCNT)*, 2021, pp. 1–7, doi: 10.1109/ICCCNT51525.2021.9580011. — Crossref
- J. Cho *et al.*, "Identification of toxic herbs using deep learning with focus on the *Sinomenium acutum*, *Aristolochiae manshuriensis caulis*, *Akebiae caulis*," *Appl. Sci.*, vol. 9, no. 24, Art. no. 5456, 2019, doi: 10.3390/app9245456. — Crossref

## Plant data
- A. Affouard *et al.*, "Pl@ntNet observations," Occurrence dataset, v1.9, GBIF, 2025, doi: 10.15468/gtebaa. — gbif.org
- A. Affouard *et al.*, "Pl@ntNet automatically identified occurrences," Occurrence dataset, v1.8, GBIF, 2023, doi: 10.15468/mma2ec. — gbif.org
- C. Garcin *et al.*, "Pl@ntNet-300K: A plant image dataset with high label ambiguity and a long-tailed distribution," in *Proc. Neural Inf. Process. Syst. Track Datasets Benchmarks*, vol. 1, 2021. — datasets-benchmarks-proceedings.neurips.cc (no pages published)
- A. Affouard, H. Goëau, P. Bonnet, J.-C. Lombardo, and A. Joly, "Pl@ntNet app in the era of deep learning," in *Proc. 5th Int. Conf. Learn. Represent. (ICLR)*, Toulon, France, 2017. — HAL hal-01629195
- A. Joly *et al.*, "Interactive plant identification based on social image data," *Ecol. Inform.*, vol. 23, pp. 22–34, Sep. 2014, doi: 10.1016/j.ecoinf.2013.07.006. — Crossref
- G. Van Horn *et al.*, "The iNaturalist species classification and detection dataset," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 8769–8778, doi: 10.1109/CVPR.2018.00914. — CVF + Crossref
- NC State Extension, "North Carolina Extension Gardener Plant Toolbox." [Online]. Available: https://plants.ces.ncsu.edu/ (accessed Oct. 18, 2026).

## BioCLIP family and TreeOfLife-200M
- S. Stevens *et al.*, "BioCLIP: A vision foundation model for the tree of life," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024, pp. 19412–19424, doi: 10.1109/CVPR52733.2024.01836. — CVF + Crossref
- J. Gu *et al.*, "BioCLIP 2: Emergent properties from scaling hierarchical contrastive learning," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 38, 2025, pp. 102778–102811, doi: 10.52202/085713-3436. — proceedings.neurips.cc (Crossref gives pp. 113880–113913; we use the publisher's page)
- Imageomics Institute, "BioCLIP 2.5 Huge (bioclip-2.5-vith14), version 1.0.0," Hugging Face, 2026, doi: 10.57967/hf/10131. [Online]. Available: https://huggingface.co/imageomics/bioclip-2.5-vith14 — HF model card. There is no separate paper; the card asks users to cite BioCLIP 2.
- Imageomics Institute, "TreeOfLife-200M," Hugging Face, 2025. [Online]. Available: https://huggingface.co/datasets/imageomics/TreeOfLife-200M — HF dataset card
- A. Radford *et al.*, "Learning transferable visual models from natural language supervision," in *Proc. 38th Int. Conf. Mach. Learn. (ICML)*, vol. 139, 2021, pp. 8748–8763. — proceedings.mlr.press
- A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021. — iclr.cc + arXiv:2010.11929
- G. Ilharco *et al.*, "OpenCLIP," Zenodo, v0.1, Jul. 2021, doi: 10.5281/zenodo.5143773. — Zenodo + repo README

## MobileNet family, transfer, and long tail
- D. Qin *et al.*, "MobileNetV4: Universal models for the mobile ecosystem," in *Computer Vision – ECCV 2024* (Lecture Notes in Computer Science, vol. 15098). Cham, Switzerland: Springer, 2024, pp. 78–96, doi: 10.1007/978-3-031-73661-2_5. — Springer + Crossref
- A. G. Howard *et al.*, "MobileNets: Efficient convolutional neural networks for mobile vision applications," arXiv:1704.04861, 2017. — arXiv
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE/CVF CVPR*, 2018, pp. 4510–4520, doi: 10.1109/CVPR.2018.00474. — CVF + Crossref
- A. Howard *et al.*, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1314–1324, doi: 10.1109/ICCV.2019.00140. — CVF + Crossref
- S. Kornblith, J. Shlens, and Q. V. Le, "Do better ImageNet models transfer better?" in *Proc. IEEE/CVF CVPR*, 2019, pp. 2656–2666, doi: 10.1109/CVPR.2019.00277. — CVF + Crossref
- J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 27, 2014. — proceedings.neurips.cc (no pages published)
- Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. IEEE/CVF CVPR*, 2019, pp. 9260–9269, doi: 10.1109/CVPR.2019.00949. — CVF + Crossref
- G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015. — arXiv

## LoRA / parameter-efficient fine-tuning
- E. J. Hu *et al.*, "LoRA: Low-rank adaptation of large language models," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022. — iclr.cc + arXiv:2106.09685
- M. Zanella and I. Ben Ayed, "Low-rank few-shot adaptation of vision-language models," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)*, 2024, pp. 1593–1603, doi: 10.1109/CVPRW63382.2024.00166. — CVF + Crossref
- S. Mangrulkar *et al.*, "PEFT: State-of-the-art parameter-efficient fine-tuning methods," GitHub, 2022. [Online]. Available: https://github.com/huggingface/peft — repo README citation

## Quantization
- B. Jacob *et al.*, "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in *Proc. IEEE/CVF CVPR*, 2018, pp. 2704–2713, doi: 10.1109/CVPR.2018.00286. — CVF + Crossref
- M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, and T. Blankevoort, "A white paper on neural network quantization," arXiv:2106.08295, 2021. — arXiv
- R. Krishnamoorthi, "Quantizing deep convolutional networks for efficient inference: A whitepaper," arXiv:1806.08342, 2018. — arXiv

## Calibration, abstention, cost-sensitive decisions
- C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, vol. 70, 2017, pp. 1321–1330. — proceedings.mlr.press
- A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle introduction," *Found. Trends Mach. Learn.*, vol. 16, no. 4, pp. 494–591, 2023, doi: 10.1561/2200000101. — Crossref (the arXiv version 2107.07511 has a different title)
- Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017. — proceedings.neurips.cc (no pages published)
- C. Elkan, "The foundations of cost-sensitive learning," in *Proc. 17th Int. Joint Conf. Artif. Intell. (IJCAI)*, vol. 2, 2001, pp. 973–978. — ijcai.org proceedings PDF

## Software and platforms
- A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019. — proceedings.neurips.cc (pages left out: no primary source lists them)
- R. Wightman, "PyTorch image models," GitHub, 2019, doi: 10.5281/zenodo.4414861. — Zenodo
- Google, "NPU acceleration with LiteRT." [Online]. Available: https://developers.google.com/edge/litert/next/npu (accessed Oct. 18, 2026). The Google Tensor path is **beta and AOT-compile only**.
- Google AI Edge, "LiteRT Torch (formerly AI Edge Torch)," GitHub. [Online]. Available: https://github.com/google-ai-edge/litert-torch (accessed Oct. 18, 2026).
- Android Developers, "Neural Networks API." [Online]. Available: https://developer.android.com/ndk/guides/neuralnetworks (accessed Oct. 18, 2026). The page states NNAPI is deprecated in Android 15.
- AMD, "System requirements (Linux) — ROCm documentation." [Online]. Available: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html (accessed Oct. 18, 2026). Lists the RX 7900 XTX (gfx1100) as supported on Ubuntu 24.04.4/22.04.5 and RHEL 10.1/9.7.

## Not verified / judgment calls
- **Pl@ntNet ICLR 2017:** whether it was the workshop track or the main conference is unconfirmed (OpenReview blocked automated access). It is cited as "ICLR 2017" only.
- **BioCLIP 2 pages:** sources disagree (see above).
- **BioCLIP 2.5 parameter count (~986M total, ~632M image encoder):** counted from the safetensors header. The model card doesn't state it.
- **TreeOfLife-200M size:** the paper says ~214M images / 952K taxa. The current HF card says 233M / 934K, an updated release. The proposal uses the paper's figures.
