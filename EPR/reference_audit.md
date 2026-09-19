# `reference.bib` citation audit

## Subsequent user-supplied corrections

The bibliography now incorporates the user's individually checked records and
contains only author, title, abbreviated venue, and year. The original audit
below is historical and is not a description of the current bibliography.
VeraRetouch is now SIGGRAPH 2026; PerTouch AAAI 2026; MonetGPT ACM Trans. Graph.
2025; JarvisArt NeurIPS 2025; JarvisEvo CVPR 2026; SA-LUT ICCV 2025;
4D LUT IEEE Trans. Image Process. 2023; and HP-Edit CVPR 2026.
Following the supplied records, CURL uses 2021, PixelRL 2019, the lightweight
retouching network 2022, and Neural Color Operators uses ECCV.
The supplied author corrections were also applied: Trung Bui in Learning by
Planning, removal of Lijun Zhang from the preference-alignment survey, and
Shengyuan Ding before Changyao Tian in Trust Your Critic.
Existing citation keys and entries absent from the user's list are retained.
For JarvisEvo and SmartEdit, the existing full author lists are retained rather
than replacing their final authors with `others`. These changes transcribe the
user's records; they are not a new independent publication-status audit.

Audited on 2026-09-19. Sources were the publisher or proceedings record first,
then Crossref for DOI metadata and arXiv for preprints. `match` means title,
complete author list, venue, and year agree with the cited source (allowing only
capitalization/BibTeX name-order differences). Preprints are recorded as arXiv
preprints, not as presumed conference papers.

## Corrections applied

| Key | Result | Verified record / action |
|---|---|---|
| `zeng2022adaint` | corrected | Replaced a different paper's author list and shortened title. Correct record: Canqian Yang; Meiguang Jin; Xu Jia; Yi Xu; Ying Chen; CVPR 2022, pp. 17501--17510, DOI `10.1109/CVPR52688.2022.01700`. |
| `yang2022clutnet` | corrected | Corrected title and full author list to Fengyi Zhang; Hui Zeng; Tianjun Zhang; Lin Zhang; ACM MM 2022, pp. 6493--6501, DOI `10.1145/3503161.3547879`. |
| `qian2026picobanana400k` | corrected | Corrected first author and expanded all authors: Yusu Qian; Eli Bocek-Rivele; Liangchen Song; Jialing Tong; Yinfei Yang; Jiasen Lu; Wenze Hu; Zhe Gan; CVPR 2026, pp. 37226--37235. |
| `bar2022editme` | removed | No publisher, proceedings, Crossref, OpenAlex, or arXiv record was found for the exact claimed title/author/venue combination. It must not be cited until a stable primary identifier is supplied. |
| `wu2023blind` | corrected | The prior author list and subtitle belonged to no matching CVPR paper. Correct record: Weixia Zhang; Guangtao Zhai; Ying Wei; Xiaokang Yang; Kede Ma; *Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective*; CVPR 2023, pp. 14071--14081. |
| `wu2024qalign` | corrected | Replaced a wholly incorrect author list and pages. Correct record: Haoning Wu; Zicheng Zhang; Weixia Zhang; Chaofeng Chen; Liang Liao; Chunyi Li; Yixuan Gao; Annan Wang; Erli Zhang; Wenxiu Sun; Qiong Yan; Xiongkuo Min; Guangtao Zhai; Weisi Lin; ICML 2024 (PMLR 235), pp. 54015--54029. |
| `ying2020pipal` | corrected | Replaced incorrect authors, DOI, and LNCS volume. Correct record: Jinjin Gu; Haoming Cai; Haoyu Chen; Xiaoxing Ye; Jimmy S. Ren; Chao Dong; ECCV 2020, LNCS 12356, pp. 633--651, DOI `10.1007/978-3-030-58621-8_37`. |
| `gu2021neuralside` | corrected | The prior title and authors had no matching CVPR record. Correct record: Valentin Khrulkov; Artem Babenko; *Neural Side-by-Side: Predicting Human Preferences for No-Reference Super-Resolution Evaluation*; CVPR 2021, pp. 4988--4997. |
| `guo2020zerodce` | corrected | Corrected pages to 1777--1786 and DOI to `10.1109/CVPR42600.2020.00185`. The removed DOI `...00144` resolves to an unrelated face-super-resolution paper. |
| `yao2026photoagent`, `bai2026editcompass`, `zheng2026a2edit` | normalized | Corrected official arXiv title forms: PhotoAgent's title, `Edit-Compass \& EditReward-Compass`, and `A$^2$-Edit`. |

## Individually matched formal publications

| Key | Status | Primary identifier |
|---|---|---|
| `charbonnier1997deterministic` | match | DOI `10.1109/83.551699` |
| `bychkovsky2011learning` | match | DOI `10.1109/CVPR.2011.5995332` |
| `liang2021ppr10k` | match | CVPR 2021 / DOI `10.1109/CVPR46437.2021.00071` |
| `moran2020deeplpf` | match | DOI `10.1109/CVPR42600.2020.01284` |
| `brooks2023instructpix2pix` | match | DOI `10.1109/CVPR52729.2023.01764` |
| `shi2020benchmark` | match | ACCV 2020 / DOI `10.1007/978-3-030-69544-6_38` |
| `bianco2020personalized` | match | DOI `10.1109/TIP.2020.2989584` |
| `wang2022neural` | match | DOI `10.1007/978-3-031-19800-7_3` |
| `gharbi2017deep` | match | DOI `10.1145/3072959.3073592` |
| `huang2024smartedit` | match | DOI `10.1109/CVPR52733.2024.00799` |
| `shi2021learning` | match | DOI `10.1109/CVPR46437.2021.01338` |
| `talebi2018nima` | match | DOI `10.1109/TIP.2018.2831899` |
| `moran2021curl` | match | ICPR 2020 record / DOI `10.1109/ICPR48806.2021.9412677` |
| `park2018distort` | match | DOI `10.1109/CVPR.2018.00621` |
| `furuta2020pixelrl` | match | DOI `10.1109/TMM.2019.2960636` |
| `wang2019underexposed` | match | DOI `10.1109/CVPR.2019.00701` |
| `cheng2020sequential` | match | DOI `10.1145/3394171.3413551` |
| `wu2026preference` | match | DOI `10.1016/j.cosrev.2026.100900` |
| `conde2024nilut` | match | DOI `10.1609/AAAI.V38I2.27901` |
| `liu2022lightweightretouching` | match | IEEE TMM 25 (2023), pp. 4638--4652, DOI `10.1109/TMM.2022.3179904` |
| `ouyang2023rsfnet` | match | DOI `10.1109/ICCV51070.2023.01117` |
| `ho2021deeppreset` | match | DOI `10.1109/WACV48630.2021.00216` |

## Individually matched arXiv preprints

All entries below resolve on arXiv with matching title, author list, and first
submission year. Their `journal` field correctly denotes an arXiv preprint;
no unverified publication venue is asserted.

| Key | arXiv identifier | Status |
|---|---:|---|
| `guo2026veraretouch` | 2604.27375 | match |
| `wu2026instantretouch` | 2606.05071 | match |
| `wu2026retouchiq` | 2602.17558 | match |
| `chang2025pertouch` | 2511.12998 | match |
| `dutt2025monetgpt` | 2505.06176 | match |
| `moon2025retouchllm` | 2510.08054 | match |
| `lin2025jarvisart` | 2506.17612 | match |
| `lin2025jarvisevo` | 2511.23002 | match |
| `shen2026agenticretoucher` | 2601.02046 | match |
| `du2026aesformer` | 2605.22126 | match |
| `yao2026photoagent` | 2602.22809 | match after title normalization |
| `ye2026agentbanana` | 2602.09084 | match |
| `hu2026talkphoto` | 2601.01915 | match |
| `regionconstrained2026grpo` | 2604.09386 | match |
| `liang2025llmlvlm` | 2508.17435 | match |
| `zhao2026imageeditr1` | 2603.08059 | match |
| `yang2026ddathinker` | 2604.25477 | match |
| `zhang2026dsieqa` | 2604.12175 | match |
| `jiang2026unieditbench` | 2604.15871 | match |
| `xu2026edithf1m` | 2603.14916 | match |
| `bai2026editcompass` | 2605.13062 | match after title normalization |
| `zhao2026lorlut` | 2602.22607 | match |
| `xue2026glut` | 2605.19889 | match |
| `he2026realignicge` | 2601.05124 | match |
| `zheng2026a2edit` | 2603.10685 | match after title normalization |
| `ma2026acetone` | 2604.00530 | match |
| `gong2025salut` | 2506.13465 | match |
| `liu2022fourdlut` | 2209.01749 | match |
| `zhao2026trustcritic` | 2603.12247 | match |
| `li2026hpedit` | 2604.19406 | match |
| `chen2026reasonedit` | 2605.07477 | match |

## Sources used

- CVF Open Access: https://openaccess.thecvf.com/
- PMLR: https://proceedings.mlr.press/v235/wu24ah.html
- Crossref REST metadata: https://api.crossref.org/
- arXiv API: https://export.arxiv.org/api/query
