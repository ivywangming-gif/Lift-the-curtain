"""Synthetic/oracle integration contract, not a visual navigation experiment."""
from dataclasses import dataclass
from enum import Enum
import json

class Judgment(str, Enum):
    PASSABLE='PASSABLE'
    BLOCKED='BLOCKED'
    UNCERTAIN='UNCERTAIN'

@dataclass(frozen=True)
class Evidence:
    candidate_id: str
    decision: Judgment
    robot_profile: str = 'G1_frozen_FALCON'
    action: str = 'single_nominal_0.10m_probe'
    exploratory: bool = True

def candidate_update(grid, cell, evidence, validated_for_navigation=False):
    """Update only this candidate. Research models cannot autonomously clear it."""
    result=[list(row) for row in grid]
    r,c=cell
    if grid[r][c]!=2: raise ValueError('Evidence must address an uncertain candidate')
    if evidence.exploratory and not validated_for_navigation:
        return result
    result[r][c]={Judgment.PASSABLE:0,Judgment.BLOCKED:1,Judgment.UNCERTAIN:2}[evidence.decision]
    return result

if __name__=='__main__':
    example=[[1,1,1,1],[0,2,2,0],[1,1,1,1]]
    e=Evidence('candidate_A',Judgment.PASSABLE)
    print(json.dumps({'input':example,'shadow_output':candidate_update(example,(1,1),e),
                      'oracle_validated_example':candidate_update(example,(1,1),e,True),
                      'behind_cell_unchanged':True,'source':'synthetic_oracle_only'}))
