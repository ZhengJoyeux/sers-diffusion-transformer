"""Read-only real-data coordinate/roundtrip preflight."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


@torch.no_grad()
def audit_curve_inputs(diffusion, *, full, bases, masks, training_indices,
                       validation_indices, directory):
    codec=diffusion.codec
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    report=dict(state={k:v for k,v in codec.state.items() if k!='bridge'},splits={})
    device=diffusion.betas.device
    pixel_step=(codec.y_max-codec.y_min)/(codec.height-1)
    for name,indices in [('training',training_indices),('validation',validation_indices)]:
        indices=np.asarray(indices,dtype=np.int64)
        # Verify all held-out inputs against the fixed axis, without fitting it.
        valid=np.asarray(masks)[indices].astype(bool)
        for field,array in [('full',full),('base',bases)]:
            values=np.asarray(array)[indices][valid]
            if not np.isfinite(values).all() or values.min()<codec.y_min or values.max()>codec.y_max:
                raise ValueError(f'{name} {field} outside fixed train-only curve bounds; no axis refit allowed')
        # Probe representative extremes and both axis lengths, plus spread rows.
        candidates=list(indices[np.linspace(0,len(indices)-1,min(len(indices),8)).astype(int)])
        amplitude=np.where(np.asarray(masks)[indices]>0,np.asarray(full)[indices],-np.inf)
        candidates+=[int(indices[np.argmax(amplitude.max(1))]),int(indices[np.argmin(np.asarray(masks)[indices].sum(1))])]
        candidates=[int(value) for value in dict.fromkeys(candidates)]
        errors=[]
        for i,row in enumerate(candidates):
            values=torch.tensor(np.asarray(full)[row:row+1],device=device)[:,None]
            mask=torch.tensor(np.asarray(masks)[row:row+1],device=device)[:,None]
            image=codec.render(values,mask)
            restored=codec.decode(image,mask)
            error=(restored-values).abs()[mask.bool()]
            errors.append(float(error.max()))
            if i<4:
                pixels=np.rint((1-np.clip((image[0,0].cpu().numpy()+1)/2,0,1))*255).astype(np.uint8)
                Image.fromarray(pixels).save(directory/f'{name}_{row}_fixed_curve.png')
                np.savez_compressed(directory/f'{name}_{row}_roundtrip.npz',original=values.cpu().numpy(),
                    restored=restored.cpu().numpy(),valid_mask=mask.cpu().numpy())
        maximum=max(errors)
        report['splits'][name]=dict(number_of_axis_checked_spectra=len(indices),
            roundtrip_checked_indices=candidates,max_abs_error=maximum,
            max_error_in_pixels=maximum/pixel_step)
        if maximum>pixel_step*0.25:
            raise RuntimeError(f'{name} curve roundtrip error exceeds 0.25 pixel; see {directory}')
    (directory/'curve_roundtrip_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('CURVE ROUNDTRIP AUDIT PASSED:',directory/'curve_roundtrip_audit.json')
    return report
