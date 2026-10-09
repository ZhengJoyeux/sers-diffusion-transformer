"""Reload a self-contained T3.23 checkpoint and evaluate real validation only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd().resolve()))
import torch
from Paths_T3_23 import build_datasets
from data_t323 import audit_selection
from engine_t323 import load_checkpoint,cache_dataset,evaluate_cache


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=32)
    args=p.parse_args();device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    datasets=build_datasets(include_generated_train=True,maximum_generated_per_condition=48)
    selection=audit_selection(datasets)
    model,payload=load_checkpoint(args.checkpoint,device)
    cache=cache_dataset(model,datasets[1],args.batch_size,args.num_workers,device)
    frame,metrics,loss=evaluate_cache(model,cache,device)
    args.output_directory.mkdir(parents=True,exist_ok=True)
    frame.to_csv(args.output_directory/"validation_predictions.csv",index=False)
    (args.output_directory/"validation_metrics.json").write_text(json.dumps(dict(metrics=metrics,corn_loss=loss,
        checkpoint=str(args.checkpoint.resolve()),accepted=payload["accepted"],epoch=payload["epoch"],
        reference_refits=0,normalization_refits=0,test_predictions=0,selection_audit=selection),ensure_ascii=False,indent=2))
    print("T3.23 CHECKPOINT VALIDATION: PASS; train-fitted buffers restored; test not evaluated",flush=True)


if __name__=="__main__":main()
