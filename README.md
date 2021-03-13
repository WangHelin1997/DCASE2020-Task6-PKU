# Audio captioning DCASE 2020 system by PKU team

A Pytorch implementation of the technical report : ["Automated Audio Captioning With Temporal Attention"](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Wang_5_t6.pdf)

We got the 3rd place in the DCASE2020 challenge.

The basic settings were following the DCASE 2020 baseline system.

See https://github.com/audio-captioning/dcase-2020-baseline for details.

## Results on the Clotho evaluation set

Metric | DCASE 2020 baseline system | Our system
-|-|-
BLEU1| 0.389|  0.489
BLEU2|  0.136|  0.285
BLEU3|  0.055|  0.177
BLEU4|  0.015|  0.107
ROUGEL|  0.262|  0.325
METEOR|  0.084|  0.148
CIDEr|  0.074|  0.252
SPICE|  0.033|  0.091
SPIDEr|  0.054|  0.172

## Results on the Clotho test set

Metric | DCASE 2020 baseline system | Our system
-|-|-
BLEU1| 0.344|  0.491
BLEU2|  0.082|  0.296
BLEU3|  0.023|  0.189
BLEU4|  0.000|  0.119
ROUGEL|  0.066|  0.153
METEOR|  0.234|  0.331
CIDEr|  0.022|  0.290
SPICE|  0.013|  0.102
SPIDEr|  0.018|  0.196

## Citation
If this code is helpful, please feel free to cite the following paper:
```
@techreport{wang2020_t6,
    Author = "Wang, Helin and Yang, Bang and Zou, Yuexian and Chong, Dading",
    title = "Automated Audio Captioning With Temporal Attention",
    institution = "DCASE2020 Challenge",
    year = "2020",
    month = "June",
    abstract = "This technical report describes the ADSPLAB team’s submission for Task6 of DCASE2020 challenge (automated audio captioning). Our audio captioning system is based on the sequence-to-sequence model. Convolutional neural network (CNN) is used as the encoder and a long-short term memory (LSTM)-based decoder with temporal attention is used to generate the captions. No extra data or pre-trained models are employed and no extra annotations are used. The experimental results show that our system could achieve the SPIDEr of 0.172 (official baseline: 0.054) on the evaluation split of the Clotho dataset."
}
```

## Acknowledgment
Thanks for the base code provided by https://github.com/audio-captioning/dcase-2020-baseline.


## Contact
If you have any problem about our code, feel free to contact
- wanghl15@pku.edu.cn

or describe your problem in Issues.

