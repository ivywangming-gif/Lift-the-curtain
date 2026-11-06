"""Round G physical configuration registry. Metadata is never policy input."""
from copy import deepcopy
import hashlib
import json

SCHEMA='round4070-g.scene.v1'


def base(scene_id, name):
    return dict(schema=SCHEMA,scene_id=scene_id,family=scene_id[0],name=name,
                material_fidelity='UNCALIBRATED_SURFACE_DEFORMABLE_PROXY',real_material_validated=False,
                geometry=dict(kind='surface',width_m=1.2,height_m=2.0,bottom_z_m=.02,center_x_m=.85,center_y_m=0.,columns=25,rows=33,pleat_amplitude_m=0.,pleat_waves=2,net_hole_period=0,net_hole_size=1),
                physics=dict(density_kg_m3=250.,thickness_m=.002,stretch_stiffness=500.,shear_stiffness=250.,bend_stiffness=.02,damping=.05,static_friction=.5,dynamic_friction=.4,hem_mass_kg=0.),
                topology=dict(kind='TOP_EDGE',suspension='STATIC',rail_limit_m=.30,ring_stride=3,local_fixed_vertex_indices=None),
                sheets=[],wall=None,scope='NEAR_FIELD_ACTION_TRAVERSABILITY')


def scene_specs():
    result={}
    def add(sid,name,geometry=None,physics=None,topology=None,**kw):
        s=base(sid,name)
        for k,v in [('geometry',geometry),('physics',physics),('topology',topology)]:
            if v:s[k].update(v)
        s.update(kw);result[sid]=s;return s
    add('A01','nominal top-fixed free bottom')
    add('A02','short 1.80m',geometry={'height_m':1.8})
    add('A03','long 2.10m',geometry={'height_m':2.1})
    add('A04','narrow 1.00m',geometry={'width_m':1.0,'columns':21})
    add('A05','raised lower edge',geometry={'height_m':1.8,'bottom_z_m':.20})
    add('A06','flat 1.95m',geometry={'height_m':1.95})
    add('A07','gentle physical pleats',geometry={'pleat_amplitude_m':.015})
    add('A08','larger physical pleats',geometry={'pleat_amplitude_m':.045,'pleat_waves':3})
    for sid,name,kind in [
        ('B01','entire top edge','TOP_EDGE'),('B02','top plus robot-left bottom','TOP_PLUS_LEFT'),
        ('B03','top plus robot-right bottom','TOP_PLUS_RIGHT'),('B04','top plus both bottom','TOP_PLUS_BOTH'),
        ('B05','only four corners','FOUR_CORNERS'),('B06','three corners robot-left bottom free','THREE_FREE_LEFT'),
        ('B07','three corners robot-right bottom free','THREE_FREE_RIGHT'),('B08','robot-left edge only','LEFT_EDGE'),
        ('B09','robot-right edge only','RIGHT_EDGE'),('B10','full perimeter','PERIMETER'),('B11','partial central top edge','PARTIAL_TOP')]:
        add(sid,name,topology={'kind':kind})
    add('C01','light single-factor density',physics={'density_kg_m3':125.})
    add('C02','nominal density control')
    add('C03','heavy free lower edge',physics={'density_kg_m3':375.})
    add('C04','heavy robot-right corner fixed',physics={'density_kg_m3':375.},topology={'kind':'TOP_PLUS_RIGHT'})
    add('C05','higher density free',physics={'density_kg_m3':562.5})
    add('C06','physical weighted lower hem',physics={'hem_mass_kg':.20})
    add('C07','single-factor increased damping',physics={'damping':.10})
    add('C08','single-factor increased bending resistance',physics={'bend_stiffness':.06})
    add('D01','fixed full rigid board',geometry={'kind':'rigid_board'},material_fidelity='RIGID_GEOMETRY_WOOD_MATERIAL_UNCALIBRATED')
    for sid,gap in [('D02',.02),('D03',.05),('D04',.08)]:
        add(sid,f'cloth with rear plate gap {gap}',wall=dict(initial_gap_m=gap,width_m=1.2,height_m=2.,thickness_m=.04,center_y_m=0.,bottom_z_m=0.,fixed=True))
    add('D05','matched cloth no rear plate')
    add('D06','partial rear plate',wall=dict(initial_gap_m=.05,width_m=.45,height_m=2.,thickness_m=.04,center_y_m=.28,bottom_z_m=0.,fixed=True))
    add('D07','rear plate beyond current probe',wall=dict(initial_gap_m=.30,width_m=1.2,height_m=2.,thickness_m=.04,center_y_m=0.,bottom_z_m=0.,fixed=True),scope='DEFER_VISUAL_REOBSERVATION')
    add('E01','fully fixed upper edge')
    add('E02','whole top support on prismatic rail',topology={'suspension':'WHOLE_RAIL'})
    add('E03','independent movable suspension points',topology={'suspension':'MULTI_RINGS','ring_stride':3})
    add('E04','one side opens on independent suspension',topology={'suspension':'MULTI_RINGS','ring_stride':3,'fixed_outer_ring':'robot_left'})
    add('E05','split center with independent moving top supports',sheets=[dict(center_y_m=-.30,width_m=.58,columns=13,suspension='WHOLE_RAIL'),dict(center_y_m=.30,width_m=.58,columns=13,suspension='WHOLE_RAIL')])
    add('F01','nominal fabric proxy')
    add('F02','light open mesh proxy',geometry={'net_hole_period':2},physics={'density_kg_m3':125.},material_fidelity='UNCALIBRATED_PERFORATED_SURFACE_NET_PROXY')
    add('F03','denser mesh proxy',geometry={'net_hole_period':3},material_fidelity='UNCALIBRATED_PERFORATED_SURFACE_NET_PROXY')
    add('F04','independent 200mm strips proxy',sheets=[dict(center_y_m=(i-2.5)*.22,width_m=.20,columns=7) for i in range(6)],material_fidelity='UNCALIBRATED_SURFACE_PVC_PROXY')
    add('F05','overlapping 200mm strips proxy',sheets=[dict(center_y_m=(i-3)*.17,width_m=.20,columns=7,center_x_offset_m=(i%2)*.025) for i in range(7)],material_fidelity='UNCALIBRATED_SURFACE_PVC_PROXY')
    add('F06','wide thick overlapping strips proxy',physics={'thickness_m':.003},sheets=[dict(center_y_m=(i-2)*.23,width_m=.28,columns=9,center_x_offset_m=(i%2)*.030) for i in range(5)],material_fidelity='UNCALIBRATED_SURFACE_PVC_PROXY')
    split=[dict(center_y_m=-.30,width_m=.58,columns=13),dict(center_y_m=.30,width_m=.58,columns=13)]
    add('F07','two independent center-opening fabric sheets',sheets=deepcopy(split))
    s=deepcopy(split);s[0]['topology']='TOP_PLUS_RIGHT';add('F08','left sheet free right sheet constrained',sheets=s)
    s=deepcopy(split);s[1]['topology']='TOP_PLUS_LEFT';add('F09','right sheet free left sheet constrained',sheets=s)
    assert len(result)==48
    for s in result.values():s['physical_config_id']=physical_hash(s)
    return result


def physical_hash(spec):
    payload={k:spec[k] for k in ['geometry','physics','topology','sheets','wall']}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]


def expanded_sheets(spec):
    result=[]
    for i,override in enumerate(spec['sheets'] or [{}]):
        g=deepcopy(spec['geometry']);p=deepcopy(spec['physics']);t=deepcopy(spec['topology'])
        for k,v in override.items():
            if k in g:g[k]=v
            elif k in p:p[k]=v
            elif k=='topology':t['kind']=v
            elif k in t:t[k]=v
        g['center_x_m'] += override.get('center_x_offset_m',0.)
        result.append(dict(name=f'sheet_{i:02d}',geometry=g,physics=p,topology=t))
    return result


def fixed_indices(g,t):
    cols,rows=g['columns'],g['rows'];top=list(range(cols));bottom=list(range((rows-1)*cols,rows*cols))
    # Local x grows toward world +Y, i.e. robot-left when facing +X.
    left=[r*cols+cols-1 for r in range(rows)];right=[r*cols for r in range(rows)]
    corners=[0,cols-1,(rows-1)*cols,rows*cols-1]
    options={'TOP_EDGE':top,'TOP_PLUS_LEFT':top+[bottom[-1]],'TOP_PLUS_RIGHT':top+[bottom[0]],'TOP_PLUS_BOTH':top+[bottom[0],bottom[-1]],'FOUR_CORNERS':corners,'THREE_FREE_LEFT':corners[:-1],'THREE_FREE_RIGHT':[corners[0],corners[1],corners[3]],'LEFT_EDGE':left,'RIGHT_EDGE':right,'PERIMETER':top+bottom+left+right,'PARTIAL_TOP':top[cols//3:2*cols//3+1]}
    if t.get('local_fixed_vertex_indices') is not None:return sorted(set(t['local_fixed_vertex_indices']))
    if t['suspension']=='MULTI_RINGS':return sorted(set(top[::t['ring_stride']]+[top[-1]]))
    return sorted(set(options[t['kind']]))
