import argparse, math
from collections import defaultdict
import pandas as pd
import trimesh, yaml
from shapely.geometry import LineString
from shapely.ops import unary_union

def target_polygon(mesh,z):
    s=mesh.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
    if s is None:return None
    try:p,_=s.to_2D()
    except Exception:p,_=s.to_planar()
    g=None
    for q in list(p.polygons_full):
        g=q if g is None else g.symmetric_difference(q)
    return g

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--trajectory",required=True);a.add_argument("--stl",required=True);a.add_argument("--config",required=True)
    x=a.parse_args()
    with open(x.config,encoding="utf-8") as f:c=yaml.safe_load(f)
    mesh=trimesh.load(x.stl,force="mesh")
    df=pd.read_csv(x.trajectory);df["mode"]=df["mode"].astype(str).str.strip().str.upper()
    h=float(c["process"]["layer_height_mm"]);base=float(c["process"]["build_plane_z_mm"])
    n=int(round((mesh.bounds[1][2]-base)/h));zs=[base+(i+.5)*h for i in range(n)]
    width=float(c["process"]["bead_width_mm"]);res=int(c["shape_validation"].get("polygon_buffer_resolution",8))
    lines=defaultdict(list);rc=defaultdict(int);length=0.;dtime=0.
    for rid,g in df.groupby("robot_id"):
        g=g.sort_values("time_s").reset_index(drop=True)
        for i in range(len(g)-1):
            r0,r1=g.iloc[i],g.iloc[i+1]
            # Validator semantics: row i mode applies to interval i -> i+1
            if r0["mode"]!="D":continue
            p0=(float(r0.x_mm),float(r0.y_mm));p1=(float(r1.x_mm),float(r1.y_mm))
            L=math.dist(p0,p1)
            if L<1e-9:continue
            z=(float(r0.z_mm)+float(r1.z_mm))/2
            li=min(range(n),key=lambda k:abs(zs[k]-z))
            if abs(z-zs[li])<=0.25:
                lines[li].append(LineString([p0,p1]))
            rc[str(rid)]+=1;length+=L;dtime+=max(0,float(r1.time_s)-float(r0.time_s))
    print("="*90);print("ACTUAL TRAJECTORY D-PATH SHAPE AUDIT");print("="*90)
    print(f"CSV rows: {len(df):,} | D intervals: {sum(rc.values()):,} | by robot: {dict(rc)}")
    print(f"Total D XY length: {length:,.2f} mm | D interval time: {dtime:,.2f} s")
    TT=TD=TI=TO=0.;failed=0
    for li,z in enumerate(zs):
        t=target_polygon(mesh,z); ls=lines.get(li,[])
        if t is None or not ls:
            if t is not None:TT+=t.area
            failed+=1;print(f"Layer {li}: D segments={len(ls)} | coverage=0.00% | IoU=0.00% | FAIL");continue
        d=unary_union(ls).buffer(width/2,resolution=res)
        ta=t.area;da=d.area;inter=t.intersection(d).area;out=d.difference(t).area;u=t.union(d).area
        cov=inter/ta;over=out/ta;iou=inter/u
        ok=iou>=float(c["shape_validation"]["minimum_layer_iou"]);failed+=int(not ok)
        print(f"Layer {li}: D segments={len(ls):5d} | coverage={cov*100:6.2f}% | overfill={over*100:6.2f}% | IoU={iou*100:6.2f}% | {'PASS' if ok else 'FAIL'}")
        TT+=ta;TD+=da;TI+=inter;TO+=out
    cov=TI/TT;over=TO/TT;u=TT+TD-TI;iou=TI/u if u else 0
    print("="*90);print(f"Coverage: {cov*100:.2f}% | Overfill: {over*100:.2f}% | IoU: {iou*100:.2f}%")
    print(f"Failed layers: {failed}/{n} ({failed/n*100:.2f}%)")
    print("IMPORTANT: row i mode is treated as interval i -> i+1.")
if __name__=="__main__":main()
