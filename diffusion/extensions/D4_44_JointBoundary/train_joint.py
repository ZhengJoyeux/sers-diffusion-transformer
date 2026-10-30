"""D4.44 joint boundary residual image head."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class JointHead(nn.Module):
    def __init__(
        self, linear, height, patch_width,
        radius=4, hidden=128
    ):
        super().__init__()
        if 2*radius > patch_width:
            raise ValueError('Boundary radius exceeds patch')
        self.weight = linear.weight
        self.bias = linear.bias
        self.height = height
        self.patch_width = patch_width
        self.radius = radius
        self.pairs = None

        with torch.random.fork_rng(devices=[]):
            self.joint = nn.Sequential(
                nn.Linear(2*linear.in_features, hidden),
                nn.SiLU(),
                nn.Linear(hidden, height*2*radius),
            )
            nn.init.zeros_(self.joint[-1].weight)
            nn.init.zeros_(self.joint[-1].bias)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        b, n, _ = base.shape
        if self.pairs is None:
            raise RuntimeError(
                'Joint head requires adjacent valid-token mask'
            )

        delta = self.joint(torch.cat(
            (x[:, :-1], x[:, 1:]), dim=-1
        ))
        delta = delta.reshape(
            b, n-1, self.height, 2*self.radius
        )
        delta = delta*self.pairs[..., None, None]
        self.last_delta_rms = (
            delta.detach().square().mean().sqrt()
        )

        correction = torch.zeros_like(base).reshape(
            b, n, self.height, self.patch_width
        )
        correction[:, :-1, :, -self.radius:] = (
            delta[..., :self.radius]
        )
        correction[:, 1:, :, :self.radius] = (
            delta[..., self.radius:]
        )
        return base+correction.reshape_as(base)


def joint_class(original):
    class JointDiT(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.final_linear = JointHead(
                self.final_linear,
                self.codec.height, self.patch_size,
            )

        def forward(
            self, image, timestep, *,
            valid_mask=None, **kwargs
        ):
            if valid_mask is None:
                raise ValueError('Pixel mask is required')
            b, _, h, w = valid_mask.shape
            tokens = valid_mask.reshape(
                b, 1, h, w//self.patch_size,
                self.patch_size,
            ).bool().any(dim=(1, 2, 4))
            self.final_linear.pairs = (
                tokens[:, :-1] & tokens[:, 1:]
            )
            try:
                return super().forward(
                    image, timestep,
                    valid_mask=valid_mask, **kwargs,
                )
            finally:
                self.final_linear.pairs = None

    return JointDiT


def install_joint(project, cfg, warm_start=False):
    import runtime

    runtime.FIELDS = (
        *runtime.FIELDS, 'joint_boundary_rms'
    )
    runtime.install(project, cfg)

    from src import dit_curve2d, curve_image_diffusion
    cls = curve_image_diffusion.CurveImageGaussianDiffusion2D
    original_losses = cls.p_losses

    def losses(self, *args, **kwargs):
        total = original_losses(self, *args, **kwargs)
        if total.ndim == 0:
            self._latest_loss_components[
                'joint_boundary_rms'
            ] = getattr(
                self.model.final_linear,
                'last_delta_rms', total.detach()*0,
            )
        return total

    cls.p_losses = losses
    if cfg['d4_44']['arm'] != 'joint':
        return

    dit_curve2d.CurveImageDiT2D = joint_class(
        dit_curve2d.CurveImageDiT2D
    )
    original_load = cls.load_state_dict

    def load(self, state_dict, strict=True, assign=False):
        fresh = self.state_dict()
        extra = {
            k for k in fresh
            if '.final_linear.joint.' in k
        }
        present = extra.intersection(state_dict)
        if warm_start and extra and not present:
            final = self.model.final_linear.joint[-1]
            if (
                torch.count_nonzero(final.weight)
                or torch.count_nonzero(final.bias)
            ):
                raise RuntimeError(
                    'Warm transfer requires a fresh zero residual head'
                )
            state_dict = {
                **state_dict,
                **{k: fresh[k] for k in extra},
            }
        return original_load(
            self, state_dict, strict=True, assign=assign
        )

    cls.load_state_dict = load


def main():
    import run
    import runtime
    import yaml

    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        'phase', choices=[
            'prepare', 'check', 'train', 'resume',
            'generate', 'evaluate',
        ],
    )
    p.add_argument(
        '--arm', choices=['control', 'joint'],
        required=True,
    )
    p.add_argument('--root', type=Path, required=True)
    p.add_argument(
        '--initializer', type=Path, required=True
    )
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--epoch', type=int)
    p.add_argument('--condition', default='DEL-H_water')
    p.add_argument('--output')
    p.add_argument('--generated')
    a = p.parse_args()
    torch.set_num_threads(2)

    s = run.settings()
    s.update(
        experiment=str(
            a.root.expanduser().resolve()/a.arm
        ),
        checkpoint=str(
            a.initializer.expanduser().resolve()
        ),
        epochs=a.epochs,
    )
    run.settings = lambda: s
    path = Path(s['experiment'])/'runtime_config.yaml'
    protocol = dict(
        version='d4_44_joint_head_v1',
        arm=a.arm,
        epochs=a.epochs,
        initializer=s['checkpoint'],
        radius=4,
        hidden=128,
        losses=(
            'original D4.38 losses; no D4.41 '
            'width or curvature addition'
        ),
    )

    if a.phase == 'prepare':
        if path.parent.exists():
            if (
                not path.is_file()
                or yaml.safe_load(path.read_text()).get(
                    'd4_44'
                ) != protocol
            ):
                raise RuntimeError(
                    'Existing experiment differs or is incomplete; '
                    'preserved: '+str(path.parent)
                )
            print('PREPARE REUSED:', path.parent)
            return

        run.prepare(a)
        cfg = yaml.safe_load(path.read_text())
        cfg.pop('d4_41', None)
        cfg['d4_44'] = protocol
        path.write_text(yaml.safe_dump(
            cfg, allow_unicode=True, sort_keys=False
        ))
        (path.parent/'joint_protocol.json').write_text(
            json.dumps(protocol, indent=2)
        )
        return

    if yaml.safe_load(path.read_text()).get(
        'd4_44'
    ) != protocol:
        raise ValueError(
            'Prepared protocol differs from arguments'
        )

    base_install = runtime.install

    def installed(project, cfg):
        runtime.install = base_install
        try:
            install_joint(
                project, cfg,
                warm_start=a.phase in (
                    'check', 'train', 'resume'
                ),
            )
        finally:
            runtime.install = installed

    runtime.install = installed
    a.checkpoint = None

    if a.phase in ('check', 'train', 'resume'):
        return run.train_or_check(a)
    if a.phase == 'generate':
        return run.generate(a)
    return run.evaluate(a)


if __name__ == '__main__':
    main()
