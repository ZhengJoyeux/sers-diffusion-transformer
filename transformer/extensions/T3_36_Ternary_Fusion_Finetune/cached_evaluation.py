"""Existing T3.33 frame construction and global-loss evaluation, unchanged."""
import torch
import pandas as pd
KEYS=("features","cp","op","logits","queries")

def prediction_frame(cache,cp,op,decoding):
    from ordinal_core import decode_ordinal_numpy
    cp=cp.cpu().numpy();op=op.cpu().numpy()
    grades=decode_ordinal_numpy(op,mode=decoding)
    truth=cache['levels'].numpy().astype(int)
    rows=[]
    for i,meta in enumerate(cache['metadata']):
        row=dict(meta,mixture_count=int((truth[i]>0).sum()),ternary_route=bool((cp[i]>=.5).all()))
        for j,drug in enumerate(('DEL','CHL','TEB')):
            row.update({'true_'+drug:int(truth[i,j]),'ordinal_'+drug:int(grades[i,j]),
                'pred_'+drug:int(grades[i,j]) if cp[i,j]>=.5 else 0,
                'presence_probability_'+drug:float(cp[i,j]),'P_ge_M_'+drug:float(op[i,j,0]),
                'P_ge_H_'+drug:float(op[i,j,1])})
        rows.append(row)
    return pd.DataFrame(rows)

@torch.no_grad()
def evaluate_cached(model,cache,args,runtime):
    from correction_model import protection_check
    from ordinal_core import present_only_corn_ordinal_loss
    model.eval();output=[];raw_logits=[]
    for start in range(0,len(cache['metadata']),args.batch_size):
        batch={k:cache[k][start:start+args.batch_size].to(args.device) for k in KEYS}
        predicted=model.forward_cached(batch)
        protection_check(predicted,batch)
        output.append(predicted['op'].cpu())
        raw_logits.append(predicted['logits'].cpu())
    op=torch.cat(output)
    frame=prediction_frame(cache,cache['cp'],op,args.ordinal_decoding)
    result=runtime.metrics(frame)
    labels=cache['levels'];presence=cache['presence']
    corn=present_only_corn_ordinal_loss(torch.cat(raw_logits),labels,presence)
    classification=torch.nn.functional.binary_cross_entropy(cache['cp'],presence)
    result.update(loss_total=float(.5*classification+.5*corn),loss_class=float(classification),
        loss_corn=float(corn),evaluation_split=cache['split'],
        ordinal_decoding=args.ordinal_decoding,max_image_clipped_fraction=cache['max_image_clipped_fraction'],
        loss_definition='0.5 presence BCE + 0.5 all-drug conditional CORN; correction training has additional regularization',
        output_protection='PASS')
    return result,frame
