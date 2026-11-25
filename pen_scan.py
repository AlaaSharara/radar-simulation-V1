import numpy as np, collections
from radarsim.pipeline import SimulationPipeline
from radarsim.motion import generate_motion_sequence
p=SimulationPipeline(root_dir='.')
objs=[(o.kind,np.array(o.box.center_m),np.array(o.box.half_extents_m)) for o in p.scene.objects]
d=p.scene.doors()[0]; v=np.array(d.panel_surface.vertices_m); A=v[0][:2]; B=v[1][:2]
def sd_box(pt,c,h):
    q=np.abs(pt-c)-h; return np.linalg.norm(np.maximum(q,0))+min(max(q),0)
def d_seg(pt,a,b):
    ab=b-a; t=np.clip(np.dot(pt-a,ab)/np.dot(ab,ab),0,1); return np.linalg.norm(pt-(a+t*ab))
res=collections.Counter(); worst={}
SKIP={'chair','sofa'}   # sitting on them is intentional contact
for seed in range(40):
    sq=generate_motion_sequence(p.scene,p.project.motion,seed=seed)
    for f in sq.frames[::4]:
        for c in f.body_capsules:
            a=np.array(c.a_m); b=np.array(c.b_m)
            for t in (0,0.5,1):
                pt=a+(b-a)*t
                for kind,cen,hx in objs:
                    if kind in SKIP: continue
                    dd=sd_box(pt,cen,hx)-c.radius_m
                    if dd<-0.04 and not (kind=='bed' and f.action_label in('sit','stand','lie_down_intentionally','intentional_lie_down','sleep','sleep_and_turn','roll_over_while_lying')):
                        res[(kind,f.action_label)]+=1; worst[(kind,f.action_label)]=min(worst.get((kind,f.action_label),0),round(dd,2))
                if pt[2]<2.1:
                    dl=d_seg(pt[:2],A,B)-c.radius_m-0.02
                    if dl<-0.04: res[('door_leaf',f.action_label)]+=1; worst[('door_leaf',f.action_label)]=min(worst.get(('door_leaf',f.action_label),0),round(dl,2))
print('penetrations deeper than 4 cm (excluding intended bed/chair/sofa contact):')
for k,vv in sorted(res.items(),key=lambda x:-x[1])[:10]: print(' ',k,vv,'worst',worst[k])
if not res: print('  none')
