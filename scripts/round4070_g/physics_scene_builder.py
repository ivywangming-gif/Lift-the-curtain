"""Shared G scene construction for physics gates and frozen G1 episodes.

All mesh/constraint data and this bundle are evaluation-only. Legacy geometries
retain the exact E builder, support dimensions and attachment construction.
"""
from pathlib import Path
from dataclasses import asdict, replace
import importlib.util
import math
import sys
from types import SimpleNamespace
import numpy as np
from physics_scenes import expanded_sheets, fixed_indices

REPO=Path(__file__).resolve().parents[2]
_path=REPO/'scripts/round4070_fixed_probe_benchmark/structural_curtain_asset.py'
_source=_path.read_text(encoding='utf-8')
_source=_source.replace('point = (float(x), 0.0, float(z))','point = _geometry_point(x, z, row, col, config)')
_source=_source.replace('faces.extend(((i, j, l), (i, l, k)))','if _keep_cell(row, col, config):\n                faces.extend(((i, j, l), (i, l, k)))')
_source=_source.replace('static_friction=0.5,','static_friction=_active["physics"]["static_friction"],').replace('dynamic_friction=0.4,','dynamic_friction=_active["physics"]["dynamic_friction"],')
_legacy=importlib.util.module_from_spec(importlib.util.spec_from_loader('_round_g_surface',loader=None))
sys.modules[_legacy.__name__]=_legacy
exec(compile(_source,str(_path),'exec'),_legacy.__dict__)


def _point(x,z,row,col,config):
    g=_legacy._active['geometry']
    y=g['pleat_amplitude_m']*math.sin(2*math.pi*g['pleat_waves']*col/(config.columns-1))*math.sin(math.pi*row/(config.rows-1))
    return float(x),float(y),float(z)


def _keep(row,col,config):
    period=_legacy._active['geometry']['net_hole_period']
    return period==0 or row in (0,config.rows-2) or col in (0,config.columns-2) or (row%period!=0 or col%period!=0)


_legacy._geometry_point=_point
_legacy._keep_cell=_keep


def array(value):
    if hasattr(value,'detach'):return value.detach().cpu().numpy()
    if hasattr(value,'numpy'):return np.asarray(value.numpy())
    return np.asarray(value)


def rotation(q):
    w,x,y,z=np.asarray(q,float)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class AggregateDeformable:
    def __init__(self,parts):self.parts=parts;self.counts=[]
    def get_nodal_positions(self):
        data=[p.get_nodal_positions() for p in self.parts]
        self.counts=[array(x[0]).shape[1] for x in data]
        # E callers use simulation and rest; middle output remains explicitly
        # geometry state, never a robot deployment observation.
        return tuple(None if data[0][k] is None else np.concatenate([array(x[k]) for x in data],axis=1) for k in range(3))
    def get_nodal_velocities(self):return np.concatenate([array(p.get_nodal_velocities()) for p in self.parts],axis=1)
    def set_nodal_velocities(self,v):
        if not self.counts:self.get_nodal_positions()
        off=0
        for p,n in zip(self.parts,self.counts):p.set_nodal_velocities(np.asarray(v)[:,off:off+n]);off+=n


class Bundle:
    def __init__(self,stage,spec,infos,configs,faces,constraints,bodies,wall):
        self.stage,self.spec,self.infos,self.configs,self.faces=stage,spec,infos,configs,faces
        self.constraints,self.dynamic_bodies=constraints,bodies
        self.scene_kind={'rigid_board':'rigid','empty':'empty'}.get(spec['geometry']['kind'],'curtain')
        self.is_control=spec['geometry']['kind']=='empty'
        self.evaluation_metadata=dict(scene_id=spec['scene_id'],physical_config_id=spec.get('physical_config_id'),material_fidelity=spec['material_fidelity'],real_material_validated=False,wall=wall,candidate_plane_x=spec['geometry']['center_x_m'],initial_gap_m=None if not wall else wall['initial_gap_m'],label_scope=spec['scope'],out_of_probe_range=spec['scope']!='NEAR_FIELD_ACTION_TRAVERSABILITY')
        self.evaluation_metadata.update(constraints=constraints,is_control=self.is_control,rigid_candidate=self.scene_kind=='rigid')
        self.offsets=[];off=0;fixed=[];bottom=[];tops=[];rest=[];paths=[]
        for info in infos:
            self.offsets.append(off);fixed += [off+i for i in info.requested_fixed_vertex_indices];bottom += [off+i for i in info.requested_bottom_anchor_indices];tops += [off+i for i in info.top_row_indices];rest+=list(info.rest_points);paths+=list(info.attachment_paths);off+=len(info.rest_points)
        self.aggregate_info=None if not infos else replace(infos[0],rest_points=tuple(rest),top_row_indices=tuple(tops),requested_fixed_vertex_indices=tuple(fixed),requested_bottom_anchor_indices=tuple(bottom),attachment_paths=tuple(paths),constraint_objects=tuple(constraints))
        edges=[]
        for offset,face in zip(self.offsets,faces):
            e=np.concatenate([face[:,[0,1]],face[:,[1,2]],face[:,[2,0]]]);edges.append(np.unique(np.sort(e,axis=1),axis=0)+offset)
        self.edges=np.concatenate(edges) if edges else np.zeros((0,2),int)
        self.initial_support_poses={}
    def make_deformable(self):
        from isaacsim.core.experimental.prims import DeformablePrim
        if not self.infos:return None
        for path,body in self.dynamic_bodies.items():
            body.update(.005);self.initial_support_poses[path]=(array(body.data.root_pos_w)[0].copy(),array(body.data.root_quat_w)[0].copy())
        return AggregateDeformable([DeformablePrim(i.prim_path,deformable_type='surface') for i in self.infos])
    def max_neighbor_stretch(self,current,rest):
        if not len(self.edges):return 1.
        e=self.edges;return float((np.linalg.norm(current[e[:,0]]-current[e[:,1]],axis=1)/np.maximum(np.linalg.norm(rest[e[:,0]]-rest[e[:,1]],axis=1),1e-9)).max())
    def constraint_drifts(self,current,initial):
        top=anchor=0.
        for c in self.constraints:
            ids=np.asarray(c['global_vertex_indices'],int)
            if not len(ids):continue
            path=c['support_path']
            if path in self.dynamic_bodies:
                body=self.dynamic_bodies[path];body.update(.005)
                p,q=array(body.data.root_pos_w)[0],array(body.data.root_quat_w)[0]
                if path not in self.initial_support_poses:self.initial_support_poses[path]=(p.copy(),q.copy())
                p0,q0=self.initial_support_poses[path]
                delta=(current[ids]-p)@rotation(q)-(initial[ids]-p0)@rotation(q0)
            else:delta=current[ids]-initial[ids]
            drift=float(np.linalg.norm(delta,axis=1).max())
            if c['region']=='top':top=max(top,drift)
            else:anchor=max(anchor,drift)
        return top,anchor
    def support_metrics(self):
        rows=[]
        for path,body in self.dynamic_bodies.items():
            body.update(.005);p,q=array(body.data.root_pos_w)[0],array(body.data.root_quat_w)[0];p0,q0=self.initial_support_poses.get(path,(p,q))
            kind=next(c['mechanism'] for c in self.constraints if c['support_path']==path)
            rows.append(dict(path=path,mechanism=kind,position_w=p.tolist(),quaternion_wxyz=q.tolist(),dy_m=float(p[1]-p0[1]),transverse_drift_m=float(np.linalg.norm((p-p0)[[0,2]])),rotation_error_rad=float(2*np.arccos(np.clip(abs(np.dot(q,q0)),0,1)))))
        return rows


def serialize_bundle(bundle, *, door=None, prim_root='/World/Bimanual'):
    """Serialize the path/topology data needed to bind a saved USD again.

    Runtime tensor handles are deliberately excluded. A later Kit process
    must recreate those handles from the reopened, hash-pinned stage.
    """
    return dict(
        schema='round4070-g.saved-scene-binding.v1',
        scene_id=bundle.spec['scene_id'],
        physical_config_id=bundle.spec['physical_config_id'],
        prim_root=prim_root,
        infos=[asdict(info) for info in bundle.infos],
        constraints=bundle.constraints,
        wall=bundle.evaluation_metadata.get('wall'),
        dynamic_body_paths=sorted(bundle.dynamic_bodies),
        door=door,
    )


def bind_existing_scene(stage, spec, metadata):
    """Recreate Python runtime handles from an already reopened native USD."""
    from pxr import UsdGeom
    from isaaclab.assets import RigidObject, RigidObjectCfg

    if metadata.get('schema') != 'round4070-g.saved-scene-binding.v1':
        raise ValueError('Unsupported saved scene binding metadata')
    if metadata.get('scene_id') != spec.get('scene_id'):
        raise ValueError('Saved scene binding scene_id mismatch')
    if metadata.get('physical_config_id') != spec.get('physical_config_id'):
        raise ValueError('Saved scene binding physical_config_id mismatch')
    infos=[];faces=[]
    for row in metadata.get('infos', []):
        info=_legacy.SurfaceSheetInfo(**row)
        prim=stage.GetPrimAtPath(info.prim_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise RuntimeError('Saved deformable mesh is missing: '+info.prim_path)
        raw=UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()
        if raw is None or len(raw)%3:
            raise RuntimeError('Saved deformable mesh triangles are invalid: '+info.prim_path)
        infos.append(info);faces.append(np.asarray(raw,int).reshape(-1,3))
    bodies={}
    for path in metadata.get('dynamic_body_paths', []):
        if not stage.GetPrimAtPath(path):
            raise RuntimeError('Saved dynamic support is missing: '+path)
        bodies[path]=RigidObject(RigidObjectCfg(prim_path=path,spawn=None))
    return Bundle(stage,spec,infos,[],faces,list(metadata.get('constraints', [])),bodies,metadata.get('wall'))


def create_scene(stage,spec,prim_root='/World/GScene'):
    from pxr import Gf,Sdf,UsdGeom,UsdPhysics
    from omni.physx.scripts import deformableUtils
    from omni.physx import get_physx_attachment_private_interface
    from isaaclab.assets import RigidObject,RigidObjectCfg
    import isaaclab.sim as sim_utils
    infos=[];configs=[];faces=[];constraints=[];bodies={};offset=0
    legacy_map={'TOP_EDGE':('A0_TOP_EDGE_ONLY','none'),'TOP_PLUS_LEFT':('A0_TOP_EDGE_ONLY','right'),'TOP_PLUS_RIGHT':('A0_TOP_EDGE_ONLY','left'),'TOP_PLUS_BOTH':('A1_TOP_EDGE_BOTTOM_TWO_CORNERS','both'),'FOUR_CORNERS':('A2_FOUR_CORNERS_ONLY','none'),'PERIMETER':('A3_FULL_PERIMETER_FIXED','none')}
    if spec['geometry']['kind'] not in ('rigid_board','empty'):
      for sheet in expanded_sheets(spec):
        g,p,t=sheet['geometry'],sheet['physics'],sheet['topology'];_legacy._active=sheet
        use_legacy=t['kind'] in legacy_map and t['suspension']=='STATIC'
        topology,bottom=legacy_map[t['kind']] if use_legacy else ('A0_TOP_EDGE_ONLY','none')
        spacing=min(g['width_m']/(g['columns']-1),g['height_m']/(g['rows']-1));small=spacing<=.04
        cfg=_legacy.CurtainConfig(width_m=g['width_m'],height_m=g['height_m'],columns=g['columns'],rows=g['rows'],support_z_m=g['bottom_z_m']+g['height_m'],density_kg_m3=p['density_kg_m3'],surface_thickness_m=p['thickness_m'],stretch_stiffness=p['stretch_stiffness'],shear_stiffness=p['shear_stiffness'],bend_stiffness=p['bend_stiffness'],damping=p['damping'],translation_m=(g['center_x_m'],g['center_y_m'],0.),yaw_rad=math.pi/2,structural_topology=topology,bottom_anchor_mode=bottom,attachment_overlap_offset_m=.005 if small else .01,bottom_anchor_size_m=.006 if small else .02)
        mesh_path=f'{prim_root}/{sheet["name"]}/Sheet'
        info=_legacy.create_surface_sheet(stage,mesh_path,cfg,create_attachment=use_legacy)
        points=np.asarray(info.rest_points);requested=fixed_indices(g,t);attachment_paths=list(info.attachment_paths);support_paths=[]
        def attachment_indices(path):
            ids=[]
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(path+'/'):
                    attr=prim.GetAttribute('omniphysics:vtxIndicesSrc0')
                    if attr:ids+=list(attr.Get() or [])
            return sorted(set(map(int,ids)))
        if use_legacy:
            for c in info.constraint_objects:
                ids=attachment_indices(c['attachment_path'])
                constraints.append(dict(c,global_vertex_indices=[offset+i for i in ids],local_vertex_indices=ids,world_positions=points[ids].tolist(),region='top' if all(i<g['columns'] for i in ids) else 'edge',mechanism='WORLD_FIXED'))
        def add_support(ids,name,mechanism='WORLD_FIXED',mass=.03):
            selected=points[ids];lo=selected.min(axis=0);hi=selected.max(axis=0);center=(lo+hi)/2;size=hi-lo+.008
            path=mesh_path+'_'+name;prim=UsdGeom.Cube.Define(stage,path);prim.CreateSizeAttr(1.)
            xf=UsdGeom.XformCommonAPI(prim);xf.SetTranslate(Gf.Vec3d(*center));xf.SetScale(Gf.Vec3f(*size));UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            if mechanism!='WORLD_FIXED':
                UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim());UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(float(mass))
                if mechanism=='PRISMATIC_RAIL':
                    joint=UsdPhysics.PrismaticJoint.Define(stage,path+'_Rail');joint.CreateBody1Rel().SetTargets([Sdf.Path(path)]);joint.CreateAxisAttr('Y');joint.CreateLocalPos0Attr(Gf.Vec3f(*center));joint.CreateLowerLimitAttr(-t['rail_limit_m']);joint.CreateUpperLimitAttr(t['rail_limit_m'])
            ap=path+'_Attachment'
            if not deformableUtils.create_auto_deformable_attachment(stage,Sdf.Path(ap),Sdf.Path(mesh_path),Sdf.Path(path)):raise RuntimeError('Could not create '+ap)
            a=stage.GetPrimAtPath(ap)
            for key,value in [('enableDeformableVertexAttachments',True),('deformableVertexOverlapOffset',.005),('enableRigidSurfaceAttachments',True)]:a.GetAttribute('physxAutoDeformableAttachment:'+key).Set(value)
            if not get_physx_attachment_private_interface().update_auto_deformable_attachment(ap):raise RuntimeError('Could not refresh '+ap)
            actual=attachment_indices(ap)
            if set(actual)!=set(ids):raise RuntimeError(f'Attachment mismatch {name}: expected {ids}, got {actual}')
            attachment_paths.append(ap);support_paths.append(path)
            constraints.append(dict(attachment_path=ap,support_path=path,global_vertex_indices=[offset+i for i in ids],local_vertex_indices=ids,world_positions=selected.tolist(),region='top' if all(i<g['columns'] for i in ids) else 'edge',mechanism=mechanism,declared_mass_kg=mass if mechanism!='WORLD_FIXED' else None))
            if mechanism!='WORLD_FIXED':bodies[path]=RigidObject(RigidObjectCfg(prim_path=path,spawn=None))
        if not use_legacy:
            if t['suspension']=='WHOLE_RAIL':add_support(list(range(g['columns'])),'TopRail','PRISMATIC_RAIL',.2)
            elif t['suspension']=='MULTI_RINGS':
                for i in requested:
                    fixed=t.get('fixed_outer_ring')=='robot_left' and i==g['columns']-1
                    add_support([i],f'Ring_{i:03d}','WORLD_FIXED' if fixed else 'PRISMATIC_RAIL')
            elif t['kind'] in ['LEFT_EDGE','RIGHT_EDGE','PARTIAL_TOP']:add_support(requested,'SelectedEdge')
            else:
                for i in requested:add_support([i],f'Anchor_{i:04d}')
            info=replace(info,support_path=support_paths[0] if support_paths else None,attachment_path=attachment_paths[0] if attachment_paths else None,attachment_paths=tuple(attachment_paths),requested_fixed_vertex_indices=tuple(requested),requested_bottom_anchor_indices=tuple(i for i in requested if i>=(g['rows']-1)*g['columns']),constraint_objects=tuple(c for c in constraints if c['attachment_path'].startswith(mesh_path)))
        if p['hem_mass_kg']>0:
            for i in [(g['rows']-1)*g['columns'],(g['rows']-1)*g['columns']+g['columns']//2,g['rows']*g['columns']-1]:add_support([i],f'HemMass_{i}','FREE_HEM_MASS',p['hem_mass_kg']/3)
            info=replace(info,attachment_paths=tuple(attachment_paths),constraint_objects=tuple(c for c in constraints if c['attachment_path'].startswith(mesh_path)))
        face=np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path)).GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
        infos.append(info);configs.append(sheet);faces.append(face);offset+=len(points)
    wall=None
    if spec['wall'] or spec['geometry']['kind']=='rigid_board':
        w=spec['wall'] or dict(initial_gap_m=0.,width_m=1.2,height_m=2.,thickness_m=.04,center_y_m=0.,bottom_z_m=0.,fixed=True)
        center_x=spec['geometry']['center_x_m']+w['initial_gap_m']+(w['thickness_m']/2 if spec['wall'] else 0.)
        center=(center_x,w['center_y_m'],w['bottom_z_m']+w['height_m']/2);size=(w['thickness_m'],w['width_m'],w['height_m'])
        cuboid=sim_utils.CuboidCfg(size=size,collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005,rest_offset=0.))
        cuboid.func(prim_root+'/RigidBarrier',cuboid,translation=center)
        wall=dict(center_xyz=center,size_xyz=size,fixed=True,initial_gap_m=w['initial_gap_m'],candidate_plane_x=spec['geometry']['center_x_m'])
    return Bundle(stage,spec,infos,configs,faces,constraints,bodies,wall)
