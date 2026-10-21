"""Save actual sampled image tensors, separately from numerical guard outputs."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def save_curve_batch(diffusion, directory, start_index):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    diagnostics = dict(diffusion.last_curve_diagnostics or {})
    diagnostics['batch_start_index'] = int(start_index)
    diagnostics['stage'] = '2D sampler output before downstream numeric calibration/delivery'
    with (directory/'column_diagnostics.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(diagnostics, ensure_ascii=False)+'\n')
    (directory/'curve_coordinate_state.json').write_text(json.dumps({k:v for k,v in diffusion.codec.state.items() if k!='bridge'},ensure_ascii=False,indent=2))
    remaining = diffusion.curve_configuration['save_images_per_condition']-start_index
    images = diffusion.last_curve_images
    if remaining <= 0 or images is None:
        return
    values = images[:remaining].numpy().astype(np.float32)
    for i, tensor in enumerate(values):
        index = start_index+i+1
        np.savez_compressed(directory/f'raw_diffusion_{index:03d}.npz', image=tensor,
                            coordinate_json=json.dumps({k:v for k,v in diffusion.codec.state.items() if k!='bridge'}))
        # PNG is display only; floating-point npz is authoritative.
        pixels = np.rint((1-np.clip((tensor[0]+1)/2, 0, 1))*255).astype(np.uint8)
        Image.fromarray(pixels).save(directory/f'raw_diffusion_{index:03d}.png')
    (directory/'README.txt').write_text(
        'raw_diffusion_*: actual 2D sampler output, before downstream numerical guards.\n'
        'PNG: display only; NPZ: original float32 image plus fixed coordinate state.\n'
        'These images must not be described as post-calibration final exported spectra.\n'
        'Invalid measurement columns/padding are masked; see curve state and source axis.\n', encoding='utf-8')
