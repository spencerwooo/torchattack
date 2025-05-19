#!/usr/bin/env python
r"""CLI tool for evaluating adversarial attack transferability.

Evaluates the effectiveness of attacks against a surrogate model and optionally 
measures transfer rates to victim models. Supports all built-in attacks from
`torchattack` against `torchvision.models` and `timm` models.

Example CLI usage:
    ```console
    $ python -m torchattack.evaluate.runner --attack PGD --eps 16/255 \
        --model-name resnet18 --victim-model-names vgg11 densenet121 \
        --dataset-root datasets/nips2017 --max-samples 200
    ```
"""

from typing import Type

from torchattack.attack import Attack


def run_attack(
    attack: str | Type['Attack'],
    attack_args: dict | None = None,
    model_name: str = 'resnet50',
    victim_model_names: list[str] | None = None,
    dataset_root: str = 'datasets/nips2017',
    max_samples: int = 100,
    batch_size: int = 4,
    save_adv_batch: int = -1,
) -> None:
    """Helper function to run evaluation on attacks.

    Example:
        ```pycon
        >>> from torchattack import FGSM
        >>> args = {"eps": 8 / 255, "clip_min": 0.0, "clip_max": 1.0}
        >>> run_attack(attack=FGSM, attack_args=args)
        ```

    Args:
        attack: The name of the attack to run.
        attack_args: A dict of keyword arguments to pass to the attack. Defaults to None.
        model_name: The name of the white-box surrogate model to attack. Defaults to "resnet50".
        victim_model_names: The names of the black-box victim models to attack. Defaults to None.
        dataset_root: Root directory of the NIPS2017 dataset. Defaults to "datasets/nips2017".
        max_samples: Max number of samples used for the evaluation. Defaults to 100.
        batch_size: Batch size for the dataloader. Defaults to 4.
        save_adv_batch: Batch index for optionally saving a batch of adversarial examples
            to visualize. Set to -1 to disable. Defaults to -1.
    """

    import torch
    from rich import print
    from rich.progress import track

    from torchattack import AttackModel, create_attack
    from torchattack.evaluate.dataset import NIPSLoader
    from torchattack.evaluate.meter import FoolingRateMeter

    if attack_args is None:
        attack_args = {}
    is_targeted = attack_args.get('targeted', False)

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained(model_name).to(device)
    transform, normalize = model.transform, model.normalize

    # Set up dataloader
    dataloader = NIPSLoader(
        root=dataset_root,
        batch_size=batch_size,
        transform=transform,
        max_samples=max_samples,
        return_target_label=is_targeted,
    )
    dataloader = track(dataloader, description='Attacking')  # type: ignore

    # Set up attack and trackers
    frm = FoolingRateMeter(is_targeted)
    adversary = create_attack(attack, model, normalize, device, **attack_args)
    print(adversary)

    # Setup victim models if provided
    if victim_model_names:
        victims = [AttackModel.from_pretrained(vn) for vn in victim_model_names]
        victim_frms = [FoolingRateMeter(is_targeted) for _ in victim_model_names]

    # Run attack over the dataset
    for i, (x, y, _) in enumerate(dataloader):
        if is_targeted:
            yl, yt = y  # Unpack target labels from `y` if the attack is targeted
            x, y = x.to(device), (yl.to(device), yt.to(device))
        else:
            x, y = x.to(device), y.to(device)

        # Create adversarial examples. Pass target labels if the attack is targeted
        advs = adversary(x, y[1]) if is_targeted else adversary(x, y)

        # Track accuracy
        cln_outs = model(normalize(x))
        adv_outs = model(normalize(advs))
        frm.update(y, cln_outs, adv_outs)

        # Save one batch of adversarial examples if requested
        if i == save_adv_batch:
            from torchattack.evaluate import save_image_batch

            save_image_batch(advs, f'outputs_{adversary.attack_name}_b{i}')

        # Track transfer fooling rates if victim models are provided
        if victim_model_names:
            for _, (v, vfrm) in enumerate(zip(victims, victim_frms)):
                v.to(device)
                vtransform = v.create_relative_transform(model)
                v_cln_outs = v(v.normalize(vtransform(x)))
                v_adv_outs = v(v.normalize(vtransform(advs)))
                vfrm.update(y, v_cln_outs, v_adv_outs)

    # Print results
    cln_acc, adv_acc, fr = frm.compute()
    print(f'Surrogate ({model_name}): {cln_acc=:.2%}, {adv_acc=:.2%} ({fr=:.2%})')

    if victim_model_names:
        for v, vfrm in zip(victims, victim_frms):
            vcln_acc, vadv_acc, vfr = vfrm.compute()
            print(
                f'Victim ({v.model_name}): cln_acc={vcln_acc:.2%}, '
                f'adv_acc={vadv_acc:.2%} (fr={vfr:.2%})'
            )


if __name__ == '__main__':
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(description='CLI tool for evaluating adversarial attack transferability.')
    parser.add_argument('--attack', type=str, required=True, help='The name of the attack to run.')
    parser.add_argument('--eps', type=str, default=None, help='The epsilon value for the attack. Do not pass for non-epsilon attacks.')
    parser.add_argument('--weights', type=str, default=None, help='Name of the generator weight. Do not pass for non-generative attacks.')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Path to the checkpoint file for the generator. Do not pass for non-generative attacks.')
    parser.add_argument('--targeted', action='store_true', help='Whether the attack is targeted.')
    parser.add_argument('--model-name', type=str, default='resnet50', help='The name of the white-box surrogate model to attack.')
    parser.add_argument('--victim-model-names', type=str, nargs='+', default=None, help='The names of the black-box victim models to attack.')
    parser.add_argument('--dataset-root', type=str, default='datasets/nips2017', help='Root directory of the NIPS2017 dataset.')
    parser.add_argument('--max-samples', type=int, default=None, help='Max number of samples used for the evaluation.')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for the dataloader.')
    parser.add_argument('--save-adv-batch', type=int, default=-1, help='Batch index for optionally saving a batch of adversarial examples to visualize. Set to -1 to disable.')
    args = parser.parse_args()
    # fmt: on

    attack_args: dict[str, str | int | None] = {}
    if args.eps:
        args.eps = eval(args.eps)
        attack_args['eps'] = args.eps
    if args.weights:
        attack_args['weights'] = args.weights
    if args.checkpoint_path:
        attack_args['checkpoint_path'] = args.checkpoint_path
    if args.targeted:
        attack_args['targeted'] = args.targeted

    run_attack(
        attack=args.attack,
        attack_args=attack_args,
        model_name=args.model_name,
        victim_model_names=args.victim_model_names,
        dataset_root=args.dataset_root,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        save_adv_batch=args.save_adv_batch,
    )
