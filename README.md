# IGS-RTA

[TOC]

This repository implements **IGS-RTA**, an algorithm for the interactive graph search problem under a generalized noisy setting. IGS-RTA employs an uncertainty-based strategy for query node selection and is able to learn latent oracle expertise and query difficulty on the fly.

## Prerequisites

- Python 3.9+
- NumPy
- SciPy

## Usage

To run IGS-RTA, use the following command for help:

```bash
python IGS-RTA.py -h

# usage: igs_RTA-revised.py [-h] [--dast {Amazon,Imagenet}] [--threshold   {60,70,80,90,99}] [--workers {3,5,7,9}]

# Run experiment with configurable parameters.

# optional arguments:
#   -h, --help            show this help message and exit
#   --dast {Amazon,Imagenet}
#                         Dataset name: 'Amazon' or 'Imagenet' (default: Amazon)
#   --threshold {60,70,80,90,99}
#                         Threshold value (default: 99).
#   --workers {3,5,7,9}   Number of workers (default: 3)
```

By default, the number of oracles is set to 3 and the termination threshold to 99%. These settings can be adjusted via the command-line arguments above.
