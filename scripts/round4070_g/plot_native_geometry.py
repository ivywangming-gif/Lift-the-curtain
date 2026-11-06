"""Plot actual native collider evidence; this is not a G1 task rollout."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from collision_clearance import _rotations

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'research/round4070_comprehensive_traversability'
OUT=ROOT/'outputs/round4070_comprehensive_traversability'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def points(part):
    if part['type']=='convex':return np.asarray(part['vertices_body_m'])
    center=np.asarray(part['center_body_m']);radius=part['radius_m']
    if part['type']=='sphere':
        a,b=np.meshgrid(np.linspace(0,2*np.pi,40),np.linspace(0,np.pi,21))
        return center+radius*np.c_[np.cos(a).ravel()*np.sin(b).ravel(),np.sin(a).ravel()*np.sin(b).ravel(),np.cos(b).ravel()]
    axis=np.asarray(part['axis_body']);helper=np.eye(3)[np.argmin(np.abs(axis))]
    first=np.cross(axis,helper);first/=np.linalg.norm(first);second=np.cross(axis,first)
    angle=np.linspace(0,2*np.pi,65)
    circle=radius*(np.cos(angle)[:,None]*first+np.sin(angle)[:,None]*second)
    return np.concatenate([center+circle+s*part['half_height_m']*axis for s in [-1,1]])


def main():
    action=json.loads((OUT/'g1_action_contract_v2.json').read_text())
    library_path=ROOT/action['witness']['native_library_path']
    if sha(library_path)!=action['witness']['native_library_sha256']:
        raise ValueError('Action-sidecar native library hash differs')
    library=json.loads(library_path.read_text())
    source_name=next(n for n in library['source_files'] if n.endswith('/native_link_pose_samples.json'))
    source=ROOT/source_name
    if sha(source)!=library['source_files'][source_name]:raise ValueError('Actual native pose source changed')
    recorded=json.loads(source.read_text());sample=recorded['samples'][-1]
    if recorded['body_names']!=library['body_names']:raise ValueError('Native body order differs')
    pose=np.asarray(sample['body_link_pose_w']);rotation,_=_rotations(pose[:,3:],1e-4)
    world=[]
    for part in library['parts']:
        index=part['body_index'];world.append((part,points(part)@rotation[index].T+pose[index,:3]))
    witness=library['action_witness'];index=witness['body_index']
    point=np.asarray(witness['point_body_m'])@rotation[index].T+pose[index,:3]
    fig,axes=plt.subplots(1,2,figsize=(6.4,6.4),layout='constrained')
    colors={'EXACT_NATIVE':'#2878a5','NATIVE_VERTEX_CONSERVATIVE_ENVELOPE':'#e49b31'}
    for ax,indices,name in [(axes[0],(1,2),'Front: Y / Z'),(axes[1],(0,2),'Side: X / Z')]:
        for part,vertices in world:
            projected=vertices[:,indices];hull=ConvexHull(projected)
            ax.add_patch(Polygon(projected[hull.vertices],closed=True,facecolor=colors[part['evidence_mode']],
                edgecolor=colors[part['evidence_mode']],alpha=.26,linewidth=.65))
        ax.scatter(*point[list(indices)],color='#b3252d',s=32,zorder=10,label='Exact-shape witness')
        ax.autoscale_view();ax.set_aspect('equal');ax.grid(alpha=.2)
        ax.set_xlabel(name+' (m)');ax.set_ylabel('World Z (m)')
    fig.suptitle('Actual G1 native collision geometry\nUncontrolled diagnostic at simulation origin',fontsize=11)
    fig.legend(handles=[Polygon([[0,0],[1,0],[1,1]],color=colors['EXACT_NATIVE'],alpha=.5,label='27 exact native shapes'),
        Polygon([[0,0],[1,0],[1,1]],color=colors['NATIVE_VERTEX_CONSERVATIVE_ENVELOPE'],alpha=.5,label='12 conservative native-vertex envelopes'),
        Line2D([0],[0],color='#b3252d',marker='o',linestyle='none',label='Exact witness: 51.9 mm native interior radius')],
        loc='outside lower center',ncols=1,fontsize=8)
    target=DOC/'NATIVE_GEOMETRY_AUDIT.png';fig.savefig(target,dpi=180);plt.close(fig)
    evidence=dict(scope='DIAGNOSTIC_NATIVE_GEOMETRY_VISUALIZATION_NOT_G1_TASK_RESULT',
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [library_path,source,Path(__file__)]},
        recorded_simulation_time_s=sample['simulation_time_s'],shape_count=len(world),
        visualization_only='Projected silhouettes; spheres/cylinders tessellated only for this figure. The independent auditor uses actual analytic primitives.',
        image_sha256=sha(target))
    (DOC/'NATIVE_GEOMETRY_AUDIT.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(target)


if __name__=='__main__':main()
