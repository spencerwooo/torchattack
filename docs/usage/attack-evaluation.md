# Loading a dataset and running evaluations

To run an attack on actual images, we would normally use a dataloader, just like loading any other dataset.

## Loading the NIPS 2017 dataset

One of the most common datasets used in evaluating adversarial transferability is the [NIPS 2017 Adversarial Learning challenges](https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set) dataset. The dataset contains 1000 images from the ImageNet validation set and is widely used in current state-of-the-art transferable adversarial attack research.

torchattack provides the [`NIPSLoader`][torchattack.eval.NIPSLoader] to load this dataset.

Provided that we have downloaded the dataset under `datasets/nips2017` with the following file structure.

```tree
datasets/nips2017
├── images
│   ├── 000b7d55b6184b08.png
│   ├── 001b5e6f1b89fd3b.png
│   ├── 00312c7e7196baf4.png
│   ├── 00c3cd597f1ee96f.png
│   ├── ...
│   └── fff35cdcce3cde43.png
├── categories.csv
└── images.csv
```

We can load the dataset like so.

```python hl_lines="3 9"
import torch
from torchattack import AttackModel, FGSM
from torchattack.eval import NIPSLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttackModel.from_pretrained(model_name='resnet50', device=device)
transform, normalize = model.transform, model.normalize

dataloader = NIPSLoader(root='datasets/nips2017', transform=transform, batch_size=16)

attack = FGSM(model, normalize, device)
```

!!! tip "Only the transform is applied to the dataloader, ==the normalize is left out==."
    Note that the earlier _separated transform and normalize function_ here. Only the transform function is passed to the dataloader,

And wrap the dataloader in the progress bar of your choice, [such as rich](https://rich.readthedocs.io/en/stable/progress.html).

```python
from rich.progress import track

dataloader = track(dataloader, description='Evaluating attack')
```

## Running the attack

When iterated over, the [`NIPSLoader`][torchattack.eval.NIPSLoader] returns a tuple of `(x, y, fname)`, where

- `x` is the batch of images,
- `y` is the batch of labels,
- and `fname` is the batch of filenames useful for saving the generated adversarial examples if needed.

We can now run the attack by iterating over the dataset and attacking batches of input samples.

```python
for x, y, fname in dataloader:
    x, y = x.to(device), y.to(device)
    x_adv = attack(x, y)
```

## Evaluating the attack

How would we know if the attack was successful?

We can evaluate the attack's **==fooling rate==**, by comparing the model's accuracy on clean samples and their associated adversarial examples. Fortunately, torchattack also provides a [`FoolingRateMetric`][torchattack.eval.FoolingRateMetric] tracker to do just that.

```python hl_lines="3 8-11"
from torchattack.eval import FoolingRateMetric

frm = FoolingRateMetric()
for x, y, fname in dataloader:
    x, y = x.to(device), y.to(device)
    x_adv = attack(x, y)

    # Track fooling rate
    x_outs = model(normalize(x))
    adv_outs = model(normalize(x_adv))
    frm.update(y, x_outs, adv_outs)
```

Finally, we can acquire the key metrics with `frm.compute()`, which returns a tuple of

- `clean_accuracy`: the model's accuracy on clean samples,
- `adv_accuracy`: the model's accuracy on adversarial examples,
- `fooling_rate`: the fooling rate of the attack.

```python
clean_accuracy, adv_accuracy, fooling_rate = frm.compute()
```
