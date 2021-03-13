# Audio captioning DCASE 2020 system by PKU team

A Pytorch implementation of the paper : ["Automated Audio Captioning With Temporal Attention"](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Wang_5_t6.pdf)

The basic settings follow the DCASE 2020 baseline system,
see https://github.com/audio-captioning/dcase-2020-baseline for details.

## Citation
If this code is helpful, please feel free to cite the following papers:
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

