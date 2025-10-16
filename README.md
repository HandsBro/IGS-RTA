# IGS-RTA

[TOC]

This repository implements **IGS-RTA**, an algorithm for the interactive graph search (IGS) problem under a generalized noisy setting. IGS-RTA employs a greedy strategy based on search uncertainty for query node selection and is able to learn latent oracle expertise and query difficulty on the fly.

## Prerequisites

- Python 3.9+
- NumPy
- SciPy

## Datasets

The benchmark dataset is located in the `/Data/Amazon` directory. This is a standard dataset widely used in IGS research.

## Usage

To run IGS-RTA, use the following command for help:

```bash
python igs_RTA.py -h

# usage: igs_RTA.py [-h] [--threshold {60,70,80,90}]
#                   [--workers {1,2,3,4,5,6,7,8,9}]
#
# Run experiment with configurable parameters.
#
# options:
#   -h, --help            Show this help message and exit
#   --threshold {60,70,80,90}
#                         Termination threshold (default: 90)
#   --workers {1,2,3,4,5,6,7,8,9}
#                         Number of oracles (default: 5)
```

By default, the number of oracles is set to 5 and the termination threshold to 90%. These settings can be adjusted via the command-line arguments above.

---

We also provide the source code for the classic **IGS** baseline for comparison. Use the following command for details:

```bash
python igs.py -h
# usage: igs.py [-h] [--workers {1,2,3,4,5,6,7,8,9}]
#
# Run experiment with configurable parameters.
#
# options:
#   -h, --help            Show this help message and exit
#   --workers {1,2,3,4,5,6,7,8,9}
#                         Number of oracles (default: 5)
```

The default oracle size is 5 and can be modified as needed.