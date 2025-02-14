# Putting it all together

For transferable black-box attacks, we would often initialize more than one model, assigning the models other than the one being directly attacked as black-box victim models, to evaluate the transferability of the attack _(the attack's effectiveness under a black-box scenario where the target victim model's internal workings are unknown to the attacker)_.

## A full example

To put everything together, we show a full example that does the following.

1. Load the NIPS 2017 dataset.
2. Initialize a pretrained ResNet-50, as the white-box surrogate model, for creating adversarial examples.
3. Initialize two other pretrained VGG-11 and Inception-v3, as black-box victim models, to evaluate transferability.
4. Run the classic MI-FGSM attack, and demonstrate its performance.

```python title="examples/mifgsm_transfer.py"
--8<-- "examples/mifgsm_transfer.py"
```

```console
$ python examples/mifgsm_transfer.py
Evaluating white-box resnet50 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:27
White-box (resnet50): 95.10% -> 0.10% (FR: 99.89%)
Evaluating black-box vgg11 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
Black-box (vgg11): 80.30% -> 59.70% (FR: 25.65%)
Evaluating black-box inception_v3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:02
Black-box (inception_v3): 92.60% -> 75.80% (FR: 18.14%)
```

!!! tip "Create relative transform for victim models (New in v1.5.0)"
    Notice in our example how we dynamically created a **relative transform** for each victim model. We use [`AttackModel.create_relative_transform`][torchattack.AttackModel.create_relative_transform] such that our relative transform for the victim model does not introduce additional unnecessary transforms such as resize and cropping that may affect the transferability of the adversarial perturbation.

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
Victim (vgg11): cln_acc=77.00%, adv_acc=34.00% (fr=55.84%)
Victim (densenet121): cln_acc=87.00%, adv_acc=37.00% (fr=57.47%)
Victim (inception_v3): cln_acc=92.00%, adv_acc=70.00% (fr=23.91%)
```
