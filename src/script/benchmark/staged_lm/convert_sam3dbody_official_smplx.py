#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from lib_pano.human3d.sam3dbody_official_conversion import convert_recon_to_smplx_official

def main():
 p=argparse.ArgumentParser();p.add_argument('--frames',required=True);p.add_argument('--recon_dir',required=True);p.add_argument('--output_dir',required=True)
 p.add_argument('--conversion_dir',default='tools/MHR/tools/mhr_smpl_conversion');p.add_argument('--model_path',required=True);p.add_argument('--mapping_path',required=True)
 p.add_argument('--device',default='cuda:0');p.add_argument('--cams',default='cam2,cam4');p.add_argument('--summary_json',required=True);a=p.parse_args()
 records=[]
 for frame in [int(x) for x in a.frames.split(',')]:
  for cam in a.cams.split(','):
   print(f'[OFFICIAL_CONVERSION] frame={frame} cam={cam}')
   r=convert_recon_to_smplx_official(Path(a.recon_dir)/cam/f'{frame:05d}.npz',Path(a.output_dir)/cam/f'{frame:05d}.npz',
      conversion_dir=a.conversion_dir,model_path=a.model_path,mapping_path=a.mapping_path,device=a.device)
   records.append({**r.__dict__,'output_path':str(r.output_path)})
 path=Path(a.summary_json);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'records':records},indent=2));print(f'[SAVE] {path}')
if __name__=='__main__':main()
