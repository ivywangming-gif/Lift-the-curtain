"""Physics-only collision and moving-attachment checks; never model inputs."""
import numpy as np


def rotation_wxyz(q):
    w,x,y,z=np.asarray(q,dtype=float)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def triangles(config, scene):
    faces=[]
    for row in range(config.rows-1):
        for col in range(config.columns-1):
            if scene=='NET_MESH' and row not in (0,config.rows-2) and col not in (0,config.columns-2) and (row+col)%2:
                continue
            i=row*config.columns+col;j=i+1;k=i+config.columns;l=k+1
            faces.extend([(i,j,l),(i,l,k)])
    return np.asarray(faces,dtype=int)


def triangle_box_overlap(tri, center, half=.055):
    """Triangle/interior-box SAT; 5 mm tolerance on the 120 mm probe."""
    v=tri-np.asarray(center)[None,None,:]
    keep=((v.max(axis=1)>-half)&(v.min(axis=1)<half)).all(axis=1)
    edges=np.roll(v,-1,axis=1)-v
    axes=[np.cross(edges[:,0],edges[:,1])]
    for j in range(3):
        for unit in np.eye(3):axes.append(np.cross(edges[:,j],unit))
    for axis in axes:
        projection=(v*axis[:,None,:]).sum(axis=2)
        radius=half*np.abs(axis).sum(axis=1)
        nonzero=np.linalg.norm(axis,axis=1)>1e-12
        keep &= ~nonzero | ((projection.max(axis=1)>-radius)&(projection.min(axis=1)<radius))
    return keep


def audit_snapshots(scene, configs, snapshots, velocities, probe_centers, rail_rows, actual_faces=None, sample_dt=.1):
    result={'scope':f'{sample_dt} s discrete geometry samples; not continuous collision certification', 'real_material_validated':False}
    checks={}
    if probe_centers:
        counts=[]
        for t,center in enumerate(probe_centers):
            count=0
            for name,config in configs.items():
                face=actual_faces[name] if actual_faces is not None else triangles(config,scene)
                count+=int(triangle_box_overlap(np.asarray(snapshots[name][t])[face],center).sum())
            counts.append(count)
        final_speed=max(float(np.linalg.norm(v[-1],axis=-1).max()) for v in velocities.values())
        result.update(probe_triangle_interior_max=max(counts),released_probe_triangle_interiors=counts[-1],final_node_speed_mps=final_speed)
        checks.update(sampled_probe_nonpenetration=max(counts)==0,release_geometry_clear=counts[-1]==0,release_speed_bounded=final_speed<5.)
    if rail_rows:
        result['rail_samples']=rail_rows
        checks.update(rail_y_limits=max(abs(r['dy_m']) for r in rail_rows)<=.305,
                      rail_transverse_fixed=max(r['transverse_drift_m'] for r in rail_rows)<=.01,
                      attachment_relative_to_support=max(r['relative_anchor_drift_m'] for r in rail_rows)<=.01,
                      rail_rotation_locked=max(r['rotation_error_rad'] for r in rail_rows)<=.01)
    result['checks']=checks
    return result
