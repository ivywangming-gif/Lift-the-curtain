"""Run only in the explicitly reserved window with no active Isaac process."""
from argparse import Namespace
import json
from pathlib import Path
from train_models import run, OUT, ROOT

methods=json.loads((OUT/'phase_a_valid/frozen_config.json').read_text(encoding='utf-8'))['methods']
run(Namespace(manifest=str(OUT/'dataset_phase_c/manifest.json'),run_name='phase_c',
              epochs=60,seeds=[4070,4071,4072],limit_methods=','.join(methods+['Instantaneous']),sides=True))
run(Namespace(manifest=str(OUT/'dataset/manifest.json'),run_name='phase_a_instantaneous',
              epochs=60,seeds=[4070,4071,4072],limit_methods='Instantaneous',sides=False))
print('OFFLINE_COMPLETION_FINISHED',flush=True)
