"""Append a versioned calibration from existing actual native evidence only."""
from pathlib import Path
import argparse,json
from physics_native_bind import validate_and_build_library,sha,read,write_new


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--parser-json',type=Path,required=True);parser.add_argument('--compound-json',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--proof',type=Path,required=True)
    parser.add_argument('--prior-proof',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    library=validate_and_build_library(args.run,args.parser_json,args.compound_json,args.library)
    prior=read(args.prior_proof);proof=dict(prior,native_library_sha256=sha(args.library),native_library_basename=args.library.name,
        frame_matches=library['frame_matches'],source_files=library['source_files'])
    snapshot=args.library.parent/('physics_native_bind_verifier_'+args.library.stem.rsplit('_',1)[-1]+'.py')
    if snapshot.exists():raise FileExistsError(snapshot)
    snapshot.write_bytes((Path(__file__).parent/'physics_native_bind.py').read_bytes())
    proof.update(verifier_source_snapshot=snapshot.resolve().relative_to(root).as_posix(),verifier_source_sha256=sha(snapshot))
    # This revision changes only display diagnostics and verifier provenance.
    # The actual native part geometry and fixed witness remain byte-identical.
    old=read(args.prior_proof.parent/args.prior_proof.name)
    if library['action_witness']['native_part_geometry_sha256']!=old['witness']['native_part_geometry_sha256']:
        raise ValueError('Witness changed; requires a separate reviewed contract')
    proof['revision_reason']='Use explicit USD display diagnostic transform; raw native frame matching and native parts unchanged'
    write_new(args.proof,proof)
    print(json.dumps(dict(library=str(args.library),library_sha256=sha(args.library),proof=str(args.proof),proof_sha256=sha(args.proof),
        max_display_rotation_error=max(x['display_usd_rotation_error'] for x in library['frame_matches']))))


if __name__=='__main__':main()
