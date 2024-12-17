# Putting it all together

For transferable black-box attacks, we would often initialize more than one model, assigning the models other than the one being directly attacked as black-box victim models, to evaluate the transferability of the attack _(the attack's effectiveness under a black-box scenario where the target victim model's internal workings are unknown to the attacker)_.

## A full example

To put everything together, we show a full example that does the following.

1. Load the NIPS 2017 dataset.
2. Initialize a pretrained ResNet-50, as the white-box surrogate model, for creating adversarial examples.
3. Initialize another pretrained VGG-19, as the black-box victim model, to evaluate adversarial transferability.
4. Run the classic MI-FGSM attack, and demonstrate its performance.

```python title="examples/mifgsm_transfer.py"
--8<-- "examples/mifgsm_transfer.py"
```

```console
$ python examples/mifgsm_transfer.py
Evaluating attack ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:47
White-box (ResNet-50): 95.10% -> 0.10% (FR: 99.89%)
Black-box (VGG-19): 89.10% -> 61.00% (FR: 31.54%)
```

## Attack Runner

torchattack provides a simple command line runner at `torchattack.eval.runner`, and the function [`run_attack`][torchattack.eval.run_attack], to evaluate the transferability of attacks. The runner itself also acts as a full example to demonstrate how researchers like us can use torchattack to create and evaluate adversarial transferability.

An exhaustive example :octicons-arrow-right-24: run the [`PGD`](../attacks/pgd.md) attack,

- with an epsilon constraint of 16/255,
- on the ResNet-18 model as the white-box surrogate model,
- transferred to the VGG-11, DenseNet-121, and Inception-V3 models as the black-box victim models,
- on the NIPS 2017 dataset,
- with a maximum of 200 samples and a batch size of 4,

```console
$ python -m torchattack.eval.runner \
    --attack PGD \
    --eps 16/255 \
    --model-name resnet18 \
    --victim-model-names vgg11 densenet121 inception_v3 \
    --dataset-root datasets/nips2017 \
    --max-samples 200 \
    --batch-size 4
PGD(model=ResNet, device=cuda, normalize=Normalize, eps=0.063, alpha=None, steps=10, random_start=True, clip_min=0.000, clip_max=1.000, targeted=False, lossfn=CrossEntropyLoss())
Attacking ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:07
Surrogate (resnet18): cln_acc=81.50%, adv_acc=0.00% (fr=100.00%)
Victim (vgg11): cln_acc=76.50%, adv_acc=33.00% (fr=56.86%)
Victim (densenet121): cln_acc=87.00%, adv_acc=37.00% (fr=57.47%)
Victim (inception_v3): cln_acc=78.50%, adv_acc=56.00% (fr=28.66%)
```
