| Metric | EfficientNetB0 | ResNet-50 | MobileNetV2 | Custom CNN |
| :--- | ---: | ---: | ---: | ---: |
| **Scaling philosophy** | Compound (d-w-r) | Depth (residual) | Width (depthwise) | None (from scratch) |
| **Total parameters** | 4,184,072 | 23,802,853 | 2,387,365 | 678,085 |
| **Trainable parameters** | 3,261,825 | 9,130,085 | 1,737,445 | 675,653 |
| **Float32 weights (MiB)** | 15.96 | 90.80 | 9.11 | 2.59 |
| **Training wall-clock (s)** | 13,159 | 16,481 | 11,862 | 14,340 |
| **Epochs** | 18 | 20 | 20 | 20 |
| **Validation Top-1** | 72.69% | 69.47% | 64.73% | 57.23% |
| **Validation Top-5** | 90.68% | 88.65% | 86.64% | 80.90% |
| **Test Top-1** | 77.26% | 73.54% | 68.69% | 61.69% |
| **Test Top-5** | 94.16% | 92.02% | 90.15% | — |
| **Test macro precision** | 0.7742 | 0.7411 | 0.6986 | 0.6372 |
| **Test macro recall** | 0.7726 | 0.7354 | 0.6869 | 0.6169 |
| **Test macro F1** | 0.7718 | 0.7355 | 0.6862 | 0.6169 |
| **Test loss** | 0.8091 | 1.1110 | 1.1692 | 1.4441 |
| **Train-val gap (pp)** | 1.47 | 13.06 | 9.62 | 0.03 |
| **Top-1 % per M-param** | 18.47 | 3.09 | 28.77 | 90.98 |
| **Top-1 % per MiB** | 4.84 | 0.81 | 7.54 | 23.85 |
