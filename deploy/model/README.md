---
license: cc-by-nc-4.0
tags:
  - image-classification
  - medical
  - dermatology
  - pytorch
library_name: pytorch
pipeline_tag: image-classification
base_model: microsoft/resnet-18
datasets:
  - marmal88/skin_cancer
---

# Skin lesion classifier (HAM10000)

ResNet18 pretrained on ImageNet, fine-tuned to name one of the seven
diagnoses in HAM10000 and to say whether a lesion should be treated as
malignant.

**Educational project. Not a diagnostic tool, not a medical device, and not
fit for making decisions about anyone's health.**

## What it does

Seven classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

The malignant/benign answer does not come from the winning class. It adds up
the probabilities of the three malignant diagnoses (`akiec`, `bcc`, `mel`) and
compares the total against **0.15**. A photo scoring 0.30 melanoma and 0.29
basal cell carcinoma is called an ordinary mole by the top-class rule, and
flagged as suspicious by this one.

## Results on the held-out test set

1508 images, 314 of them malignant. Split by lesion rather than by photo, so
no lesion appears in both training and test. Run once, on the finished model.

**Malignant or benign, at the 0.15 threshold:**

| | |
|---|---|
| recall on malignant | **92.0%** |
| precision on malignant | 43.0% |
| malignant lesions missed | 25 of 314 |
| false alarms | 383 of 1194 |

**Per class, taking the top class:**

| class | support | recall | precision |
|---|---|---|---|
| akiec | 60 | 53.3% | 46.4% |
| bcc | 79 | 72.2% | 63.3% |
| bkl | 195 | 45.6% | 74.2% |
| df | 12 | 75.0% | 24.3% |
| mel | 175 | 66.3% | 42.5% |
| nv | 967 | 83.4% | 91.1% |
| vasc | 20 | 100.0% | 58.8% |

Accuracy is 74.9%. That figure is close to meaningless here: two thirds of the
data are ordinary moles, so answering `nv` every time scores 67.2%.

## What it does badly

**Wrong kind of photograph.** HAM10000 is dermatoscope imagery. Phone
snapshots look different enough that accuracy on them is unknown and probably
lower. This is the largest limitation and it is not fixable without different
training data.

**Rare classes carry unreliable numbers.** `df` has 12 test images, `vasc` 20.
Their 75% and 100% move by tens of points on a handful of pictures.

**Melanoma recall is 66.3% under the top-class rule**, and 37 melanomas were
called ordinary moles. The threshold rule catches most of those, because the
model spreads its uncertainty across the malignant classes rather than
settling on `nv`. That gap is the argument for the threshold.

**It is confident about rubbish.** Shown random noise, the classifier returned
`bkl` at 86.5% certainty. Any application built on this needs its own check
that the input is a skin photograph at all.

## Using it

```python
import torch
from torchvision.models import resnet18
import torch.nn as nn

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")

model = resnet18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("resnet.pth", map_location="cpu"))
model.eval()
```

Input is 224x224 RGB normalised with the ImageNet statistics
`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`, because that is
what the pretrained layers were fed.

The full application, with the API, the container and the attention maps,
is at
[github.com/miloszkrawczyk/melanoma-classifier](https://github.com/miloszkrawczyk/melanoma-classifier).

## Training

| | |
|---|---|
| base | ResNet18, ImageNet weights, final layer replaced with 7 outputs |
| data | HAM10000, split 70/15/15 by `lesion_id` |
| augmentation | horizontal and vertical flips, ±20° rotation, colour jitter |
| loss | cross-entropy weighted by inverse class frequency |
| optimiser | Adam, lr 0.0001 |
| epochs | 15, best at 7 |

Splitting by lesion rather than by photo matters: HAM10000 photographs some
lesions more than once, and a random split would put the same lesion on both
sides and flatter the score.

## Licence and citation

CC BY-NC 4.0, inherited from HAM10000. Non-commercial use only.

> Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large
> collection of multi-source dermatoscopic images of common pigmented skin
> lesions.* Scientific Data 5, 180161 (2018).
> https://doi.org/10.1038/sdata.2018.161
