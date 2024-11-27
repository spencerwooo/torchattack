from typing import Any


def run_attack(
    attack: Any,
    attack_args: dict | None = None,
    model_name: str = 'resnet50',
    victim_model_names: list[str] | None = None,
    dataset_root: str = 'datasets/nips2017',
    max_samples: int = 100,
    batch_size: int = 4,
) -> None:
    """Helper function to run evaluation on attacks.

    Example:
        >>> from torchattack import FGSM
        >>> args = {"eps": 8 / 255, "clip_min": 0.0, "clip_max": 1.0}
        >>> run_attack(attack=FGSM, attack_args=args)

    Note:
        For generative attacks, the model_name argument that defines the white-box
        surrogate model is not required. You can, however, manually provide a model name
        to load according to your designated weight for the generator, to recreate a
        white-box evaluation scenario, such as using VGG-19 (`model_name='vgg19'`) for
        BIA's VGG-19 generator weight (`BIAWeights.VGG19`).

    Args:
        attack: The attack class to initialize, either by name or class instance.
        attack_args: A dict of keyword arguments passed to the attack class.
        model_name: The surrogate model to attack. Defaults to "resnet50".
        victim_model_names: A list of the victim black-box models to attack. Defaults to None.
        dataset_root: Root directory of the dataset. Defaults to "datasets/nips2017".
        max_samples: Max number of samples to attack. Defaults to 100.
        batch_size: Batch size for the dataloader. Defaults to 16.
    """

    import torch
    from rich import print
    from rich.progress import track

    from torchattack import AttackModel, create_attack
    from torchattack.eval.dataset import NIPSLoader
    from torchattack.eval.metric import FoolingRateMetric

    if attack_args is None:
        attack_args = {}

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttackModel.from_pretrained(model_name, device)
    transform, normalize = model.transform, model.normalize

    # Set up dataloader
    dataloader = NIPSLoader(
        root=dataset_root,
        batch_size=batch_size,
        transform=transform,
        max_samples=max_samples,
    )
    dataloader = track(dataloader, description='Attacking')  # type: ignore

    # Set up attack and trackers
    frm = FoolingRateMetric()
    attacker = create_attack(attack, model, normalize, device, attack_args=attack_args)
    print(attacker)

    # Setup victim models if provided
    if victim_model_names:
        victims = [AttackModel.from_pretrained(n, device) for n in victim_model_names]
        victim_frms = [FoolingRateMetric() for _ in victim_model_names]

    # Run attack over the dataset (100 images by default)
    for _i, (x, y, _) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Adversarial images are created here
        advs = attacker(x, y)

        # Track accuracy
        cln_outs = model(normalize(x))
        adv_outs = model(normalize(advs))
        frm.update(y, cln_outs, adv_outs)

        # *Save first batch of adversarial examples
        # if _i == 0:
        #     import torchvision as tv

        #     saved_imgs = advs.detach().cpu().mul(255).to(torch.uint8)
        #     img_grid = tv.utils.make_grid(saved_imgs, nrow=4)
        #     tv.io.write_png(img_grid, 'adv_batch_0.png')

        # Track transfer fooling rates if victim models are provided
        if victim_model_names:
            for _, (vmodel, vfrm) in enumerate(zip(victims, victim_frms)):
                v_cln_outs = vmodel(vmodel.normalize(x))
                v_adv_outs = vmodel(vmodel.normalize(advs))
                vfrm.update(y, v_cln_outs, v_adv_outs)

    # Print results
    cln_acc, adv_acc, fr = frm.compute()
    print(f'Surrogate ({model_name}): {cln_acc=:.2%}, {adv_acc=:.2%} ({fr=:.2%})')

    if victim_model_names:
        for vmodel, vfrm in zip(victims, victim_frms):
            vcln_acc, vadv_acc, vfr = vfrm.compute()
            print(
                f'Victim ({vmodel.model_name}): cln_acc={vcln_acc:.2%}, '
                f'adv_acc={vadv_acc:.2%} (fr={vfr:.2%})'
            )


if __name__ == '__main__':
    import argparse

    import torchattack

    parser = argparse.ArgumentParser(description='Run an attack on a model.')
    parser.add_argument('--attack', type=str, required=True)
    parser.add_argument('--eps', type=float, default=16)
    parser.add_argument('--weights', type=str, default='DEFAULT')
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--victim-model-names', type=str, nargs='+', default=None)
    parser.add_argument('--dataset-root', type=str, default='datasets/nips2017')
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()

    attack_args = {}
    args.eps = args.eps / 255
    if args.attack not in torchattack.NON_EPS_ATTACKS:  # type: ignore
        attack_args['eps'] = args.eps
    if args.attack in torchattack.GENERATIVE_ATTACKS:  # type: ignore
        attack_args['weights'] = args.weights
        attack_args['checkpoint_path'] = args.checkpoint_path

    run_attack(
        attack=args.attack,
        attack_args=attack_args,
        model_name=args.model_name,
        victim_model_names=args.victim_model_names,
        dataset_root=args.dataset_root,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
