"""Export the ridge_0 sampling protocol with the original QC gates."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

import run
from core import ROOT, verify


def main():
    args = run.parser().parse_args(["generate", *sys.argv[1:]])
    verify(ROOT, "PACKAGE_SHA256.json")
    torch.set_num_threads(2)
    original_load, original_dump = run.load, run.dump
    protocol_path = Path(args.output).expanduser().resolve() / "protocol.json"

    def load_ridge0(options):
        bundle, model, diffusion = original_load(options)
        codec = bundle.residual
        temperature = codec.configuration["decode_temperature"]
        sigma = codec.configuration["sigma_pixels"]
        rows = torch.arange(codec.height, device=bundle.device, dtype=torch.float32)
        rows = rows.view(1, 1, -1, 1)

        def render_ridge(_model, _args, kwargs, image):
            probability = torch.softmax(image.float() / temperature, dim=2)
            center = (probability * rows).sum(dim=2, keepdim=True)
            ridge = torch.exp(-0.5 * ((rows - center) / sigma).square())
            return (2 * ridge - 1) * kwargs["valid_mask"]

        model.register_forward_hook(render_ridge, with_kwargs=True)
        return bundle, model, diffusion

    def dump_protocol(path, data):
        if Path(path) == protocol_path:
            data.update(output_layer="ridge_0", finetuning_updates=0)
        return original_dump(path, data)

    with patch.object(run, "load", load_ridge0), patch.object(run, "dump", dump_protocol):
        run.generate(args)


if __name__ == "__main__":
    main()
