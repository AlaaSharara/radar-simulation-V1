import numpy as np
from radarsim.pipeline import SimulationPipeline
from radarsim.motion import generate_motion_sequence
p = SimulationPipeline(root_dir='.')
d, falls = [], 0
for seed in range(50):
    s = generate_motion_sequence(p.scene, p.project.motion, seed=seed)
    d.append(len(s.frames) / s.sample_rate_hz)
    falls += len(s.fall_events)
print('min %.1f  mean %.1f  max %.1f  falls %d' % (min(d), np.mean(d), max(d), falls))
# smoothness
mx=[];ma=[]
for seed in range(20):
    s = generate_motion_sequence(p.scene, p.project.motion, seed=seed)
    xy=np.array([[f.root_position_m[0],f.root_position_m[1]] for f in s.frames])
    v=np.linalg.norm(np.diff(xy,axis=0),axis=1)*100
    mx.append(v.max()); ma.append(np.abs(np.diff(v)).max()*100)
print('max speed %.2f m/s  max accel %.1f m/s^2'%(max(mx),max(ma)))
