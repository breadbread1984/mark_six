# Introduction

This project try to figure out occurrence probability of every ball of mark six lottery with max loglikelihood evaluation (MLE).

# Usage

## Install prerequisite packages

```shell
python3 -m pip install -r requirements.txt
```

## Download history data of mark six

download history data in xlsx format [here](https://zh.lottolyzer.com/history/hong-kong/mark-six/)

## Run MLE

```shell
python3 train.py --input <path/to/mark/six/history/xlsx>
```

After execution, there is a **params.npy** generated. it stores the evaluated probability of occurrence of ball 1-49.

## Print balls from maximum to minimum occurrence probability

```python
import numpy as np
p = np.load('params.npy')
idx = np.argsort(p)[::-1]
print(idx+1)
```
