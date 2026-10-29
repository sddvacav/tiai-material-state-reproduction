# TiAI: Making Material States Count in Literature-Driven Design of Titanium Alloys and Titanium Matrix Composites

Numerical reproduction package for the manuscript of the same title. It contains the released inputs, the analysis code and the recorded results of nine analyses of literature-derived strength, ductility and related properties of titanium alloys and titanium matrix composites.

## Contents

| Path | Contents |
| --- | --- |
| `代码/` | Analysis code (Python) and the restore script |
| `数据/` | Numerical inputs, Source Data and recorded results |
| `说明/` | Data guide, data access, data licence, requirements and release notes |
| `复现文件清单.json` | Size and SHA-256 of each repository file and the path at which the analysis code reads it |

Files carry descriptive Chinese names. The restore script writes them, byte for byte, into the directory layout read by the analysis code (`tools/`, `data/` and `paper01/`). `说明/数据字典.md` relates the runtime file names to the figures and tables of the manuscript.

## Requirements

- Python 3.12 or 3.13
- An NVIDIA GPU with CUDA; the numerical comparisons run on the GPU in float64
- NumPy, pandas, PyTorch with CUDA support, PyArrow and Matplotlib (`说明/环境依赖.txt`)

The recorded results were obtained with Python 3.13.13, PyTorch 2.11.0+cu128 and CUDA 12.8 on an NVIDIA GeForce RTX 5080 Laptop GPU, on which `materials` takes about two minutes and the other two commands a few seconds each.

## Quick start

```bash
python 代码/恢复运行目录.py runtime
cd runtime
python -m pip install -r requirements.txt
python tools/validate_release.py
python tools/reproduce_analysis.py --analysis materials
python tools/reproduce_analysis.py --analysis saved-models
python tools/reproduce_analysis.py --analysis target-plane
```

`validate_release.py` checks every file against its recorded size and SHA-256 and confirms the composition of the selection cohort. Each reproduction command recomputes the quantities below from the released inputs, compares them with the released Source Data within the tolerances stated in the code, and writes `results/<analysis>/report.json`; it stops at the first difference outside tolerance. `数据/数值复现结果.json` collects the recorded reports of the three commands.

## Analyses

| Mode | Manuscript | Recomputed quantities |
| --- | --- | --- |
| `material-relations` | Figure 4 | Equal-source strength–elongation covariance moments and primary 2,000-draw within-source intervals |
| `associations` | Figure 5 | Within- and between-source rank effects with 500-draw intervals, leave-one-source-out estimates, random-effects summaries and false-discovery-rate control |
| `microstructure` | Figure 6 | Six same-record descriptor–property coordinate sets |
| `temporal` | Figure 7 | Five-year endpoint windows and 300-draw source-bootstrap intervals |
| `frontier` | Figure 8 | Observed Pareto membership, dated hypervolume, 400 source-balanced portfolios and the 81 × 81 attainment grid |
| `dose` | Figure S23 | Regressions of 3,646 dose series, source summaries and 2,000-draw intervals for 536 summary cells |
| `addition-maps` | Figures 11 and 12 | Designed-series selection, source medians, 4,000-draw intervals and reinforcement-content bins |
| `saved-models` | Figure 9 | Metrics and error distributions of the five property models on 4,808 qualified saved queries |
| `target-plane` | Figure 10 | The 40,401-cell strength–elongation target plane for 913 candidates from 168 sources |

`materials` runs the first seven modes in turn.

## Evaluation conditions

- The five property models are evaluated on their saved predictions under the recorded protocols of Figure 9 and Table S3, which use same-source or labelled target-source support. The models are not retrained here.
- The target plane and the selection tables use the recorded outcomes of the fixed cohort of 913 candidates from 168 sources and describe the stated single-choice setting.
- Source extraction, source adjudication and model training use licensed full texts; they are described in the Supporting Information and are not part of this package.

## Extraction benchmark materials

`数据/科研抽取任务与提示.jsonl`, `数据/科研抽取参考标注.jsonl` and `数据/科研抽取保存预测.jsonl` contain the instructions and output schema, the reference annotations and the parsed model outputs of the extraction benchmark (Table S1, Section S2). Source documents are identified by DOI and SHA-256 and are not distributed. `代码/文献抽取评分协议.py` implements the scoring rules.

## Licence and citation

The code is released under the MIT License (`LICENSE`). The authors' curation, annotations and numerical results are released under CC BY 4.0 (`说明/数据许可.md`) to the extent of the authors' rights; the measurements remain attributed to their original publications through the DOIs in each table. Please cite this release (`CITATION.cff`) together with the original studies whose measurements you use.
