import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- load & normalize ----------

def load_mask_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "mask" in cols:
        return df[cols["mask"]].to_numpy(dtype=np.float32).reshape(-1)
    if df.shape[1] == 2 and "token_idx" in cols:
        other = [c for c in df.columns if c.lower() != "token_idx"][0]
        return df[other].to_numpy(dtype=np.float32).reshape(-1)
    if df.shape[1] == 1:
        return df.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1)
    raise ValueError(f"{path.name}: cannot infer mask column from {list(df.columns)}")

def norm_basic(x: np.ndarray, how: str, p_lo: float, p_hi: float) -> np.ndarray:
    if x.size == 0: return x.astype(np.float32)
    if how == "none": return x.astype(np.float32)
    if how == "minmax":
        lo, hi = np.min(x), np.max(x)
    elif how == "percentile":
        lo, hi = np.percentile(x, [p_lo, p_hi])
    else:
        raise ValueError(how)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1).astype(np.float32)

def final_normalize(vec: np.ndarray, how: str, p_lo: float, p_hi: float, gamma: float) -> np.ndarray:
    v = norm_basic(vec, how, p_lo, p_hi)
    if gamma != 1.0: v = np.power(v, float(gamma))
    return v.astype(np.float32)

# ---------- fusion helpers ----------

def fuse_topk_mean(mats: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, mats.shape[0]))
    part = np.partition(mats, -k, axis=0)[-k:]
    return part.mean(axis=0)

def fuse_mean(mats: np.ndarray) -> np.ndarray:
    return mats.mean(axis=0)

def fuse_max(mats: np.ndarray) -> np.ndarray:
    return mats.max(axis=0)

# ---------- map & token ----------

def id2aa_true_map() -> Dict[int, str]:
    return {
        0:"CLS", 1:"PAD", 2:"EOS", 3:"UNK",
        4:"L",5:"A",6:"G",7:"V",8:"S",9:"E",10:"R",11:"T",12:"I",
        13:"D",14:"P",15:"K",16:"Q",17:"N",18:"F",19:"Y",20:"M",
        21:"H",22:"W",23:"C",24:"X",25:"B",26:"U",27:"Z",28:"O",29:".",30:"-"
    }

def tokens_to_aa1(ids: np.ndarray, id2aa: Dict[int,str]) -> np.ndarray:
    return np.array([id2aa.get(int(t), "UNK") for t in ids], dtype=object)

# ---------- mask discovery ----------

def build_exact_sequence(mask_dir: Path, pdbid: str) -> List[Path]:
    pid = pdbid.lower()
    def first(pat: str) -> Optional[Path]:
        hits = sorted(Path(mask_dir).glob(pat))
        return hits[0] if hits else None
    seq=[]
    def add(p: Optional[Path], label:str):
        if p is None: print(f"[WARN] {pdbid}: missing {label}")
        else: seq.append(p)
    add(first(f"mc_model.cg_model.core_model.attn_dym.0__{pid}.csv"),"core_model.attn_dym.0")
    add(first(f"mc_model.cg_model.core_model.ffn_dym.0__{pid}.csv"),"core_model.ffn_dym.0")
    add(first(f"mc_model.cg_model.fc_dym__{pid}.csv"),"cg_model.fc_dym")
    for i in range(12):
        add(first(f"mc_model.cg_model.gnn_dym.{i}__{pid}.csv"),f"cg_model.gnn_dym.{i}")
        add(first(f"mc_model.cg_model.ffn_dym.{i}__{pid}.csv"),f"cg_model.ffn_dym.{i}")
    for i in range(6):
        add(first(f"mc_model.row_dym.{i}__{pid}.csv"),f"row_dym.{i}")
        add(first(f"mc_model.ffn_dym.{i}__{pid}.csv"),f"ffn_dym.{i}")
    return seq

# ---------- main ----------

def main():
    ap=argparse.ArgumentParser(description="Fuse DyM masks per structure and separate antibody vs antigen averages.")
    ap.add_argument("--seq-npz",required=True,type=Path)
    ap.add_argument("--mask-dir",required=True,type=Path)
    ap.add_argument("--fusion",choices=["topk_mean","mean","max"],default="topk_mean")
    ap.add_argument("--k-top",type=int,default=5)
    ap.add_argument("--norm",choices=["none","minmax","percentile"],default="percentile")
    ap.add_argument("--p-lo",type=float,default=5)
    ap.add_argument("--p-hi",type=float,default=99)
    ap.add_argument("--final-norm",choices=["none","minmax","percentile"],default="percentile")
    ap.add_argument("--final-p-lo",type=float,default=5)
    ap.add_argument("--final-p-hi",type=float,default=99)
    ap.add_argument("--gamma",type=float,default=1.0)
    ap.add_argument("--plot-out",required=True,type=Path)
    ap.add_argument("--csv-out",type=Path,default=None)
    args=ap.parse_args()

    z=np.load(args.seq_npz,allow_pickle=True)
    id2aa=id2aa_true_map()
    canon=["L","A","G","V","S","E","R","T","I","D","P","K","Q","N","F","Y","M","H","W","C"]

    all_aa_antibody=[]
    all_val_antibody=[]
    all_aa_antigen=[]
    all_val_antigen=[]
    used_pdb=0

    for pdbid in z.files:
        arr=z[pdbid]
        tokens=arr[0]
        aa1=tokens_to_aa1(tokens,id2aa)
        eocs=np.where(tokens==24)[0]
        if len(eocs)<2:
            print(f"[WARN] {pdbid}: cannot find EOCs (need ≥2); skip.")
            continue
        Ls,Lh,Lg=eocs[0],eocs[1],len(tokens)
        chain_spans={"Lchain":(0,Ls),"Hchain":(Ls+1,Lh),"Antigen":(Lh+1,Lg)}
        mask_paths=build_exact_sequence(args.mask_dir,pdbid)
        mats=[]
        for p in mask_paths:
            s=load_mask_csv(p)
            if s.shape[0]!=tokens.shape[0]:
                print(f"[WARN] {pdbid}: {p.name} mismatch; skip.")
                continue
            v=s.copy()
            mask=(tokens>=4)&(tokens<=23)
            v[mask]=norm_basic(v[mask],args.norm,args.p_lo,args.p_hi)
            mats.append(v)
        if not mats: continue
        M=np.stack(mats,axis=0)
        if args.fusion=="topk_mean": fused=fuse_topk_mean(M,args.k_top)
        elif args.fusion=="mean": fused=fuse_mean(M)
        else: fused=fuse_max(M)
        fused=final_normalize(fused,args.final_norm,args.final_p_lo,args.final_p_hi,args.gamma)
        used_pdb+=1
        # collect by group
        for cname,(start,end) in chain_spans.items():
            is_res=np.isin(aa1[start:end],canon)
            aa_seg=aa1[start:end][is_res]
            val_seg=fused[start:end][is_res]
            if "Antigen" in cname:
                all_aa_antigen.append(aa_seg)
                all_val_antigen.append(val_seg)
            else:
                all_aa_antibody.append(aa_seg)
                all_val_antibody.append(val_seg)

    def avg_df(aa_list,val_list):
        aa=np.concatenate(aa_list)
        vals=np.concatenate(val_list).astype(float)
        df=pd.DataFrame({"aa":aa,"score":vals})
        df=df.groupby("aa",as_index=False)["score"].mean()
        df["order"]=df["aa"].apply(lambda x: canon.index(x) if x in canon else 999)
        return df.sort_values("order")

    df_ab=avg_df(all_aa_antibody,all_val_antibody)
    df_ag=avg_df(all_aa_antigen,all_val_antigen)
    if args.csv_out:
        out=pd.concat([
            df_ab.assign(group="Antibody"),
            df_ag.assign(group="Antigen")
        ])
        out.to_csv(args.csv_out,index=False)
        print(f"[OK] CSV saved → {args.csv_out}")

    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    axs[0].bar(df_ab["aa"],df_ab["score"],color="skyblue")
    axs[0].set_title("Antibody (L+H)")
    axs[1].bar(df_ag["aa"],df_ag["score"],color="salmon")
    axs[1].set_title("Antigen")
    for ax in axs:
        ax.set_xlabel("Residue")
        ax.set_ylabel("Avg fused mask")
    plt.tight_layout()
    args.plot_out.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(args.plot_out,dpi=200)
    plt.close()
    print(f"[OK] Processed {used_pdb} PDBs. Plot → {args.plot_out}")

if __name__=="__main__":
    main()

