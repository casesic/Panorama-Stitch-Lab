"""Create an input-only value-pose-offset corpus from a recorded pipeline run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--capture",type=Path,required=True);parser.add_argument("--pipeline-fixture",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for camera in ("cam2","cam4"):
        shutil.copyfile(a.pipeline_fixture/f"model_{camera}_selected_posedirs.f32",a.output/f"{camera}_selected_posedirs.f32")
    counters={}; accepted={}; rows=[]
    with (a.capture/"value_pose_offset_call_manifest.csv").open() as stream:
        for source in csv.DictReader(stream):
            stage=source["stage"];camera=source["camera"];caller="objective" if stage.endswith("objective") else "linearization"
            key=(stage,camera);call=counters.get(key,0);counters[key]=call+1
            state_key=(stage.split("_")[0],caller);state=accepted.get(state_key,0);accepted[state_key]=state+1
            src=a.capture/source["feature_file"];data=src.read_bytes();name=source["feature_file"];(a.output/name).write_bytes(data)
            rows.append({"call_index":source["call_index"],"stage":stage,"caller":caller,"camera":camera,"trial_index":call-1,"accepted_state_index":state,"state_hash":hashlib.sha256(data).hexdigest(),"N":source["N"],"M":source["M"],"K":source["K"],"feature_file":name})
    with (a.output/"call_manifest.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    (a.output/"fixture_policy.txt").write_text("REFERENCE_LEAKAGE=NONE\ncontents=selected_posedirs,pose_feature,call_metadata\n")
    print(f"VALUE_POSE_OFFSET_CALL_FIXTURE_WRITTEN calls={len(rows)}")

if __name__=="__main__":main()
