"""Load the retained T3.33 image recognizer with its exact embedded T3.32 baseline."""
from pathlib import Path
import sys
from types import SimpleNamespace

EXTENSIONS = Path(__file__).resolve().parent / 'extensions'


def load_checkpoint(path, device='cpu'):
    """Return (frozen eval model, original PlotScale, checkpoint metadata).

    Input tensors: image, image_valid_columns, raw_intensity, measured_mask.
    Output: cp [B,3], op/logits [B,3,2], plus ternary correction diagnostics.
    Pesticide order is DEL, CHL, TEB. No dataset or training cache is required.
    """
    correction = EXTENSIONS / 'T3_33_Ternary_Correction'
    image = EXTENSIONS / 'T3_32_Image_Recognition'
    sys.path.insert(0, str(correction))
    from baseline_bridge import configure
    _, hashes = configure(image)
    from run_correction import load_correction
    model, scale, state = load_correction(Path(path), SimpleNamespace(device=device, t332_hashes=hashes))
    return model.requires_grad_(False).eval(), scale, state
