"""Isolated check/smoke/train/evaluate/restore entry; originals never overwritten."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from prepare_t324 import PACKAGE,DEFAULT_BASELINE,DEFAULT_PREVIOUS,DEFAULT_PROTECTED,prepare,assert_unchanged,file_hash


def check_package():
    for name,expected in json.loads((PACKAGE/"package_manifest.json").read_text()).items():
        if file_hash(PACKAGE/name)!=expected:raise RuntimeError(f"Package file changed: {name}")


def command(command,cwd,logfile=None):
    env=os.environ.copy();env["PYTHONPATH"]=str(PACKAGE)+os.pathsep+env.get("PYTHONPATH","")
    env.setdefault("MPLBACKEND","Agg")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):env.setdefault(key,"4")
    print("Running:"," ".join(map(str,command)),flush=True)
    if logfile is None:subprocess.run(command,cwd=cwd,env=env,check=True);return
    with Path(logfile).open("w",encoding="utf-8") as log:
        proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in proc.stdout:print(line,end="",flush=True);log.write(line);log.flush()
            code=proc.wait()
            if code:raise subprocess.CalledProcessError(code,command)
        except BaseException:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("check","smoke","train","evaluate","restore"))
    p.add_argument("--project",default="/home/wqzheng/project_transformer")
    p.add_argument("--encoder-package",default="/home/wqzheng/T3_17_encoder_upgrade")
    p.add_argument("--baseline-run",default=DEFAULT_BASELINE)
    p.add_argument("--previous-run",default=DEFAULT_PREVIOUS)
    p.add_argument("--protected-run",default=DEFAULT_PROTECTED)
    p.add_argument("--variant",choices=("both","concentration_update","residual_control"),default="both")
    p.add_argument("--device",default="cuda")
    p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--cache-batch-size",type=int,default=32)
    p.add_argument("--checkpoint",type=Path)
    args=p.parse_args();check_package()
    state=prepare(args.project,args.encoder_package,args.baseline_run,args.previous_run,args.protected_run)
    known=json.loads((PACKAGE/"expected_reference_state.json").read_text())
    if state["baseline_checkpoint_sha256"]!=known["baseline_checkpoint_sha256"]:
        raise RuntimeError("Baseline checkpoint differs from the diagnosed T3.18 anchor")
    weighted=Path(known["weighted_checkpoint_path"])
    weighted_exists=weighted.is_file()
    if weighted_exists and file_hash(weighted)!=known["weighted_checkpoint_sha256"]:
        raise RuntimeError("Existing T3.20 checkpoint differs from diagnosis")
    runtime=PACKAGE/"runtime_project"
    try:
        if args.mode=="restore":
            print("Original code/data/anchor checkpoint are intact; no reverting required.")
            print("Original checkpoint:",state["baseline_checkpoint"])
            return
        if args.mode=="check":
            command([sys.executable,"-m","unittest","discover","-s",str(PACKAGE),"-p","test*t324.py","-v"],runtime)
            print("T3.24 ARTIFICIAL MODEL CHECK: PASS; no server optimizer updates",flush=True)
            return
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")+f"_{os.getpid()}"
        out=Path(state["project"])/"outputs"/f"t3_24_concentration_update_{args.mode}_{stamp}";out.mkdir(exist_ok=False)
        (out/"source_state.json").write_text(json.dumps(state,ensure_ascii=False,indent=2))
        if args.mode=="evaluate":
            if args.checkpoint is None:raise ValueError("evaluate requires --checkpoint")
            cmd=[sys.executable,"-u",str(PACKAGE/"Inference_T3_24.py"),"--checkpoint",str(args.checkpoint),"--output-directory",str(out),
                 "--device",args.device,"--num-workers",str(args.num_workers),"--batch-size",str(args.cache_batch_size)]
        else:
            cmd=[sys.executable,"-u",str(PACKAGE/"T3_24_Training.py"),"--output-directory",str(out),"--variant",args.variant,
                 "--device",args.device,"--num-workers",str(args.num_workers),"--cache-batch-size",str(args.cache_batch_size)]
            if args.mode=="smoke":cmd.append("--smoke")
        command(cmd,runtime,out/"console.log")
        (PACKAGE/("last_"+args.mode+"_path.txt")).write_text(str(out)+"\n")
        archive=out/"T3_24_concentration_update_reports.zip"
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path!=archive and "checkpoints" not in path.relative_to(out).parts and path.suffix!=".pt":
                    z.write(path,path.relative_to(out))
        print(f"Upload this report: {archive}",flush=True)
    finally:
        assert_unchanged(state,check_data=True)
        if weighted_exists and (not weighted.is_file() or file_hash(weighted)!=known["weighted_checkpoint_sha256"]):
            raise RuntimeError("Existing T3.20 checkpoint changed")


if __name__=="__main__":main()
