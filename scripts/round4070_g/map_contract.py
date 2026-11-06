"""Synthetic two-layer map contract; does not claim camera or navigation deployment."""
from dataclasses import dataclass
import json
from pathlib import Path
import unittest


@dataclass(frozen=True)
class Candidate:
    identifier: str
    cells: tuple
    layer: int
    observed_at: float
    observation_source: str
    action_id: str


def update(grid, candidate, decision, *, action_id, evidence_time,
           navigation_validated=False, previous_layer_passed_at=None):
    if candidate.layer not in [1, 2]:
        raise ValueError('Only two layers are in scope')
    if decision not in ['PASSABLE', 'BLOCKED', 'UNCERTAIN']:
        raise ValueError('UNKNOWN truth is not a model output')
    if action_id != candidate.action_id:
        raise ValueError('Evidence for a different followup action cannot clear this candidate')
    if evidence_time < candidate.observed_at:
        raise ValueError('Judgment precedes candidate observation')
    if candidate.layer == 2:
        if (previous_layer_passed_at is None or candidate.observed_at <= previous_layer_passed_at
                or candidate.observation_source not in ['fresh_rgbd', 'fresh_oracle_geometry']):
            raise ValueError('Layer two requires new geometry after layer one')
    result = [list(row) for row in grid]
    for r, c in candidate.cells:
        if result[r][c] != 2:
            raise ValueError('Only the identified uncertain candidate can be updated')
    if navigation_validated:
        for r, c in candidate.cells:
            result[r][c] = {'PASSABLE': 0, 'BLOCKED': 1, 'UNCERTAIN': 2}[decision]
    return result


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.grid = [[1,1,1,1,1], [0,2,2,2,0], [1,1,1,1,1]]
        self.first = Candidate('front', ((1,1),), 1, 1., 'oracle_geometry', 'frozen_action')

    def test_shadow_and_locality(self):
        args = dict(action_id='frozen_action', evidence_time=3.)
        self.assertEqual(update(self.grid,self.first,'PASSABLE',**args), self.grid)
        changed = update(self.grid,self.first,'PASSABLE',navigation_validated=True,**args)
        self.assertEqual(changed[1], [0,0,2,2,0])
        self.assertEqual(self.grid[1], [0,2,2,2,0])

    def test_second_layer_requires_new_observation(self):
        second = Candidate('back',((1,3),),2,2.,'oracle_geometry','frozen_action')
        with self.assertRaises(ValueError):
            update(self.grid,second,'BLOCKED',action_id='frozen_action',evidence_time=5.,previous_layer_passed_at=3.)
        second = Candidate('back',((1,3),),2,4.,'fresh_oracle_geometry','frozen_action')
        changed = update(self.grid,second,'BLOCKED',action_id='frozen_action',evidence_time=5.,previous_layer_passed_at=3.,navigation_validated=True)
        self.assertEqual(changed[1],[0,2,2,1,0])

    def test_action_and_layer_limits(self):
        with self.assertRaises(ValueError):
            update(self.grid,self.first,'PASSABLE',action_id='different_action',evidence_time=3.)
        third = Candidate('third',((1,3),),3,4.,'fresh_oracle_geometry','frozen_action')
        with self.assertRaises(ValueError):
            update(self.grid,third,'PASSABLE',action_id='frozen_action',evidence_time=5.)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    root = Path(__file__).resolve().parents[2]
    out = root/'outputs/round4070_comprehensive_traversability/map_contract.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    grid = [[1,1,1,1],[0,2,2,0],[1,1,1,1]]
    c = Candidate('candidate_front',((1,1),),1,1.,'oracle_geometry','frozen_action')
    out.write_text(json.dumps(dict(source='synthetic_oracle_only',tests_run=result.testsRun,
        tests_passed=result.wasSuccessful(),input=grid,
        shadow=update(grid,c,'PASSABLE',action_id='frozen_action',evidence_time=3.),
        oracle_verified=update(grid,c,'PASSABLE',action_id='frozen_action',evidence_time=3.,navigation_validated=True),
        real_rgbd_or_grounding_dino_executed=False),indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful(): raise SystemExit(1)
