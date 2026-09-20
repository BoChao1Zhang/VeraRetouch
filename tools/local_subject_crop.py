"""Large, subject-oriented context crops; outside controls stay exactly zero."""
import numpy as np
from tools.preview_three_part_local import window_sum, select_window


def context_windows(states,masks,step,subject_priority=True):
    h,w=states.shape[1:3];mask=masks[step-1]
    subject=masks[-1]>0 if subject_priority else np.ones((h,w),dtype=bool)
    delta=np.abs(states[step]-states[step-1]).mean(-1)*100
    gy,gx=np.gradient(states[step-1].mean(-1));texture=np.abs(gx)+np.abs(gy)
    inside=None;outside=None
    # Preserve the full-frame aspect ratio for six truly equal image panels.
    for fraction in [.18,.14,.10,.075]:
        cw=max(2,round(w*np.sqrt(fraction)));ch=max(2,round(h*np.sqrt(fraction)))
        subject_fraction=window_sum(subject,cw,ch)/(cw*ch)
        active_fraction=window_sum(mask>0,cw,ch)/(cw*ch)
        valid=(subject_fraction>=.8)&(active_fraction>=.5)
        inside=select_window(delta*subject,valid,cw,ch)
        if inside:
            x,y=inside['box'][:2]
            inside.update(subject_fraction=float(subject_fraction[y,x]),
                          active_fraction=float(active_fraction[y,x]),area_fraction=cw*ch/(w*h))
            break
    for fraction in [.18,.14,.10,.075,.04,.02,.01,.005]:
        cw=max(2,round(w*np.sqrt(fraction)));ch=max(2,round(h*np.sqrt(fraction)))
        outside=select_window(texture,window_sum(mask!=0,cw,ch)==0,cw,ch)
        if outside:
            outside.update(area_fraction=cw*ch/(w*h))
            break
    if not inside or not outside:return None
    x0,y0,x1,y1=outside['box']
    outside_change=float(delta[y0:y1,x0:x1].max())
    assert outside_change==0
    return dict(step=step,eligible=True,inside=inside,outside=outside,outside_max_change=outside_change)
