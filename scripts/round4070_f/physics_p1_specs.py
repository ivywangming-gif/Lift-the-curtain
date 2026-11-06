"""Explicit F physics candidates; names are offline construction metadata only."""
from dataclasses import dataclass, asdict, replace
from lift_curtain.envs.task_scale_scene import FormalTaskSceneConfig


@dataclass(frozen=True)
class Sheet:
    name: str
    center_y_m: float = 0.0
    width_m: float = 1.2
    columns: int = 25
    bottom_anchor_mode: str = 'none'
    center_x_offset_m: float = 0.0

    def curtain_config(self, formal):
        return replace(formal.curtain_config(), width_m=self.width_m,
                       columns=self.columns, bottom_anchor_mode=self.bottom_anchor_mode,
                       translation_m=(formal.curtain_x_m+self.center_x_offset_m, self.center_y_m, 0.0))


@dataclass(frozen=True)
class Scene:
    name: str
    sheets: tuple
    mechanism: str

    def to_dict(self):
        return asdict(self)


def scene_specs():
    return {
        'SLIDING_TOP_TRACK': Scene('SLIDING_TOP_TRACK', (Sheet('single'),), 'dynamic support constrained by world-Y prismatic joint; uncalibrated rail friction'),
        'PLEATED_SINGLE': Scene('PLEATED_SINGLE', (Sheet('single'),), '15 mm sinusoidal rest geometry; top row remains planar'),
        'NET_MESH': Scene('NET_MESH', (Sheet('single'),), 'checkerboard interior holes in surface mesh; approximate mesh, not calibrated mosquito net'),
        'PVC_STRIPS_PROXY': Scene('PVC_STRIPS_PROXY', tuple(Sheet('strip'+str(i), (i-1)*0.4, 0.38, 9) for i in range(3)), 'three separated deformable sheets; real PVC constitutive model NOT available'),
        'SPLIT_CENTER': Scene('SPLIT_CENTER', (Sheet('right', -0.30, 0.58, 13), Sheet('left', 0.30, 0.58, 13)), 'two independent sheets with 20 mm center gap; differs from overlapping E scene'),
        'ONE_SIDE_CONSTRAINED': Scene('ONE_SIDE_CONSTRAINED', (Sheet('right', -0.30, 0.58, 13), Sheet('left', 0.30, 0.58, 13, 'right')), 'split scene with left outer bottom constrained'),
    }


def scene_by_name(name):
    return scene_specs()[name]
