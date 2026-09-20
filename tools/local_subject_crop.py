"""Large, subject-oriented context crops; outside controls stay exactly zero."""
import numpy as np
import threading
import hashlib
from pathlib import Path
from tools.preview_three_part_local import window_sum, select_window

_face_models=threading.local()
FACE_MODEL=Path('/home/bc/data/models/opencv_yunet/face_detection_yunet_2023mar.onnx')
FACE_MODEL_SHA256='8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
# Official MIT-licensed OpenCV Zoo artifact; used only for figure crop location.
FACE_MODEL_URL='https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'

def detect_faces(rgb):
    """Locate faces for cropping only; no identification or attribute inference."""
    import cv2
    cv2.setNumThreads(1)
    if not hasattr(_face_models,'detector'):
        if not FACE_MODEL.is_file():raise FileNotFoundError(f'OpenCV Zoo YuNet weights are required for face-aware crops: {FACE_MODEL}')
        assert hashlib.sha256(FACE_MODEL.read_bytes()).hexdigest()==FACE_MODEL_SHA256
        _face_models.detector=cv2.FaceDetectorYN.create(str(FACE_MODEL),'',(320,320),.9,.3,5000)
    h,w=rgb.shape[:2];scale=min(1.,640/max(h,w))
    small=cv2.resize(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8),(round(w*scale),round(h*scale)))
    _face_models.detector.setInputSize((small.shape[1],small.shape[0]))
    _,faces=_face_models.detector.detect(cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
    if faces is None:return []
    boxes=[]
    for detection in faces:
        x,y,fw,fh=detection[:4]
        box=[max(0,round(x/scale)),max(0,round(y/scale)),min(w,round((x+fw)/scale)),min(h,round((y+fh)/scale))]
        if box[2]>box[0] and box[3]>box[1]:boxes.append(box)
    return boxes


def context_windows(states,masks,step,subject_priority=True,allow_low_support=False,face_boxes=None):
    h,w=states.shape[1:3];mask=masks[step-1]
    subject=masks[-1]>0 if subject_priority else np.ones((h,w),dtype=bool)
    delta=np.abs(states[step]-states[step-1]).mean(-1)*100
    faces=detect_faces(states[-1]) if face_boxes is None else face_boxes
    gy,gx=np.gradient(states[step-1].mean(-1));texture=np.abs(gx)+np.abs(gy)
    inside=None;outside=None;fallback=None
    # Preserve the full-frame aspect ratio for six truly equal image panels.
    fractions=[.18,.14,.10,.075]
    if any(f[2]-f[0]>.42*w or f[3]-f[1]>.42*h for f in faces):fractions=[.36,.25]+fractions
    for fraction in fractions:
        cw=max(2,round(w*np.sqrt(fraction)));ch=max(2,round(h*np.sqrt(fraction)))
        subject_fraction=window_sum(subject,cw,ch)/(cw*ch)
        active_fraction=window_sum(mask>0,cw,ch)/(cw*ch)
        valid=(subject_fraction>=.8)&(active_fraction>=.5)
        # Prefer a visibly affected face with surrounding context. If none is
        # detected/edited, bias gently upward rather than choosing tiny texture.
        totals=window_sum(delta*subject,cw,ch)/(cw*ch)
        yy,xx=np.indices(valid.shape);cy=yy+ch/2
        scores=totals*(1+.35*np.exp(-((cy-.30*h)/(.23*h))**2))
        face_valid=np.zeros_like(valid)
        best=float(totals[valid].max()) if valid.any() else 0.
        for fx0,fy0,fx1,fy1 in faces:
            if fx1-fx0>cw or fy1-fy0>ch:continue
            if float(delta[fy0:fy1,fx0:fx1].mean())<max(.15,.03*best):continue
            if float((mask[fy0:fy1,fx0:fx1]>0).mean())<.5:continue
            fcx=(fx0+fx1)/2;fcy=(fy0+fy1)/2
            face_valid|=((xx<=fx0)&(xx+cw>=fx1)&(yy<=fy0)&(yy+ch>=fy1)&
                         (fcx>=xx+.25*cw)&(fcx<=xx+.75*cw)&
                         (fcy>=yy+.25*ch)&(fcy<=yy+.60*ch))
        focus='detected_face' if (valid&face_valid).any() else 'upper_edit_region'
        usable=valid&face_valid if focus=='detected_face' else valid
        if usable.any():
            y,x=np.unravel_index(np.where(usable,scores,-np.inf).argmax(),scores.shape)
            inside=dict(box=[int(x),int(y),int(x+cw),int(y+ch)],score=float(totals[y,x]),focus=focus,detected_faces=faces)
        else:inside=None
        if inside:
            x,y=inside['box'][:2]
            inside.update(subject_fraction=float(subject_fraction[y,x]),
                          active_fraction=float(active_fraction[y,x]),area_fraction=cw*ch/(w*h))
            if focus=='detected_face' or not faces:break
            if fallback is None:fallback=inside
    if inside is None or inside.get('focus')!='detected_face':inside=fallback or inside
    for fraction in [.18,.14,.10,.075,.04,.02,.01,.005]:
        cw=max(2,round(w*np.sqrt(fraction)));ch=max(2,round(h*np.sqrt(fraction)))
        outside=select_window(texture,window_sum(mask!=0,cw,ch)==0,cw,ch)
        if outside:
            outside.update(area_fraction=cw*ch/(w*h))
            break
    outside_kind='zero'
    if outside is None and allow_low_support:
        cw=max(2,round(w*np.sqrt(.18)));ch=max(2,round(h*np.sqrt(.18)))
        outside=select_window(-mask,np.ones((h-ch+1,w-cw+1),dtype=bool),cw,ch)
        outside.update(area_fraction=cw*ch/(w*h))
        outside_kind='low_support'
    if not inside or not outside:return None
    x0,y0,x1,y1=outside['box']
    outside_change=float(delta[y0:y1,x0:x1].max())
    if outside_kind=='zero':assert outside_change==0
    outside['mean_support']=float(mask[y0:y1,x0:x1].mean())
    return dict(step=step,eligible=True,inside=inside,outside=outside,outside_max_change=outside_change,
                outside_kind=outside_kind)
