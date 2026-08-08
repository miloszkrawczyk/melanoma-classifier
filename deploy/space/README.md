---
title: Skin Lesion Classifier
emoji: 🔬
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-4.0
---

# Skin lesion classifier

**Educational project, not a diagnostic tool.** Upload a close-up of a skin
lesion and the model names one of seven diagnoses and says whether to treat it
as malignant. It also shows which part of the photo the answer rested on.

A ResNet18 pretrained on ImageNet, fine-tuned on
[HAM10000](https://doi.org/10.7910/DVN/DBW86T). Trained from scratch as a
portfolio project: data handling, training, evaluation, API, container,
interface.

## What it does badly

- **HAM10000 is dermatoscope imagery.** Phone photos look different enough
  that accuracy on them is unknown and probably worse.
- Two of the seven classes have fewer than 110 training images, so their
  numbers move by tens of points on a handful of pictures.
- The check for "is this even a skin photo" is a heuristic on colour and
  texture, not a classifier. It turns away noise and flat colours; it will
  accept a photograph of a beige wall.

Anything on your skin that is new, changing or bothering you is a question for
a dermatologist. Uploaded photos are not stored.

## Endpoints

The page is one way in. There is also a JSON API:

| method | path | returns |
|---|---|---|
| POST | `/predict` | the diagnosis as JSON |
| POST | `/explain` | the photo with a heatmap, as a PNG |
| GET | `/docs` | interactive API documentation |

## Source

- [Source on GitHub](https://github.com/miloszkrawczyk/melanoma-classifier)
- [Weights on the Hub](https://huggingface.co/miloszkrawczyk/skin-lesion-classifier)

## Citation

Tschandl P., Rosendahl C., Kittler H. *The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin
lesions.* Scientific Data 5, 180161 (2018). Licensed CC BY-NC.
