"""Auditable local extensions to frozen E surface builder for physics-only tests.

The original file is read only. No extension is used in the G1 controller.
"""
from pathlib import Path
import sys
_base = Path(__file__).resolve().parents[1] / 'round4070_fixed_probe_benchmark' / 'structural_curtain_asset.py'
_source = _base.read_text(encoding='utf-8')
_scene = sys.argv[sys.argv.index('--scene') + 1] if '--scene' in sys.argv else ''

if _scene == 'PLEATED_SINGLE':
    _old = 'point = (float(x), 0.0, float(z))'
    assert _source.count(_old) == 1
    _source = _source.replace(_old, 'point = (float(x), 0.015 * math.sin(4*math.pi*col/(config.columns-1)) * math.sin(math.pi*row/(config.rows-1)), float(z))')
elif _scene == 'NET_MESH':
    _old = 'faces.extend(((i, j, l), (i, l, k)))'
    assert _source.count(_old) == 1
    _source = _source.replace(_old, 'if row in (0, config.rows-2) or col in (0, config.columns-2) or (row+col)%2 == 0:\n                faces.extend(((i, j, l), (i, l, k)))')
elif _scene == 'SLIDING_TOP_TRACK':
    _old = 'support_paths.append(path)'
    assert _source.count(_old) == 1
    _source = _source.replace(_old, '''if name == "Support":
            UsdPhysics.RigidBodyAPI.Apply(support.GetPrim())
            UsdPhysics.MassAPI.Apply(support.GetPrim()).CreateMassAttr(0.2)
            rail = UsdPhysics.PrismaticJoint.Define(stage, path + "_Rail")
            rail.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            rail.CreateAxisAttr("Y")
            rail.CreateLocalPos0Attr(Gf.Vec3f(*world_center))
            rail.CreateLocalRot1Attr(Gf.Quatf(math.cos(yaw/2), Gf.Vec3f(0, 0, -math.sin(yaw/2))))
            rail.CreateLowerLimitAttr(-0.30)
            rail.CreateUpperLimitAttr(0.30)
        support_paths.append(path)''')

exec(compile(_source, str(_base), 'exec'), globals())
