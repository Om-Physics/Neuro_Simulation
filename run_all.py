import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats, signal
from scipy.ndimage import uniform_filter1d

FIGDIR = Path(__file__).parent / "figures"
FIGDIR.mkdir(exist_ok=True)

# ── Global plot style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "savefig.dpi": 200,
})

C = {   # palette — colour-blind friendly
    "V":"#2c7bb6","m":"#d7191c","h":"#1a9641","n":"#ff7f00",
    "E":"#d73027","I":"#4575b4","LFP":"#4d4d4d","w":"#984ea3",
    "AMPA":"#d7191c","NMDA":"#2c7bb6","GABAA":"#1a9641","GABAB":"#ff7f00",
}

def _tag(ax, lbl, x=-0.16, y=1.06):
    ax.text(x, y, lbl, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="right")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  HODGKIN-HUXLEY MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("━" * 60)
print("1 / 8  Hodgkin-Huxley neuron …")

# ── biophysical parameters (original 1952 squid axon, 6.3 °C) ────────────
C_m  = 1.0;    g_Na = 120.0; g_K  = 36.0;  g_L  = 0.3
E_Na = 55.0;   E_K  = -77.0; E_L  = -54.387
V0   = -65.0

def alpha_m(V): dV=V+40.; return (abs(dV)<1e-7) and 1.0 or 0.1*dV/(1-np.exp(-dV/10.))
def beta_m(V):  return 4.0*np.exp(-(V+65.)/18.)
def alpha_h(V): return 0.07*np.exp(-(V+65.)/20.)
def beta_h(V):  return 1./(1.+np.exp(-(V+35.)/10.))
def alpha_n(V): dV=V+55.; return (abs(dV)<1e-7) and 0.1 or 0.01*dV/(1-np.exp(-dV/10.))
def beta_n(V):  return 0.125*np.exp(-(V+65.)/80.)

def ss(V):
    am,bm=alpha_m(V),beta_m(V); ah,bh=alpha_h(V),beta_h(V); an,bn=alpha_n(V),beta_n(V)
    return am/(am+bm), ah/(ah+bh), an/(an+bn)

def hh_deriv(state, I_ext):
    V,m,h,n = state
    INa = g_Na*m**3*h*(V-E_Na)
    IK  = g_K *n**4  *(V-E_K)
    IL  = g_L        *(V-E_L)
    return np.array([(I_ext-INa-IK-IL)/C_m,
                     alpha_m(V)*(1-m)-beta_m(V)*m,
                     alpha_h(V)*(1-h)-beta_h(V)*h,
                     alpha_n(V)*(1-n)-beta_n(V)*n])

def hh_simulate(I_ext_arr, dt=0.025):
    m0,h0,n0 = ss(V0)
    state = np.array([V0,m0,h0,n0])
    n = len(I_ext_arr)
    V_r=np.zeros(n); m_r=np.zeros(n); h_r=np.zeros(n); n_r=np.zeros(n)
    spk=[]
    above=False
    for i,I in enumerate(I_ext_arr):
        k1=hh_deriv(state,I); k2=hh_deriv(state+.5*dt*k1,I)
        k3=hh_deriv(state+.5*dt*k2,I); k4=hh_deriv(state+dt*k3,I)
        state += (dt/6.)*(k1+2*k2+2*k3+k4)
        state[1:]=np.clip(state[1:],0,1)
        V_r[i]=state[0]; m_r[i]=state[1]; h_r[i]=state[2]; n_r[i]=state[3]
        if state[0]>=-55. and not above: spk.append(i*dt); above=True
        elif state[0]<-55.: above=False
    INa=g_Na*m_r**3*h_r*(V_r-E_Na)
    IK =g_K *n_r**4   *(V_r-E_K)
    IL =g_L            *(V_r-E_L)
    return dict(V=V_r,m=m_r,h=h_r,n=n_r,INa=INa,IK=IK,IL=IL,spikes=np.array(spk))

dt=0.025; T=150.0
t=np.arange(0,T,dt)
I_arr=np.zeros(len(t)); I_arr[int(10/dt):int(140/dt)]=10.0
hh=hh_simulate(I_arr, dt=dt)
hh["t"]=t; hh["I"]=I_arr

# f-I curve for HH
def hh_fi(I_vals, T_ms=800, dt=0.025):
    out=[]
    for Iv in I_vals:
        nn=int(T_ms/dt); Ia=np.full(nn,Iv)
        res=hh_simulate(Ia,dt); out.append(len(res["spikes"])/(T_ms*1e-3))
    return np.array(out)

I_range_hh=np.linspace(0,20,40)
print("   computing HH f-I curve (40 points)…")
fI_hh=hh_fi(I_range_hh)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  LEAKY INTEGRATE-AND-FIRE
# ═══════════════════════════════════════════════════════════════════════════
print("2 / 8  Leaky Integrate-and-Fire …")

def lif_simulate(I_ext_arr, dt=0.1,
                 Cm=200.,gL=10.,EL=-70.,Vthr=-55.,Vreset=-70.,Vpeak=30.,tref=2.):
    V=EL; ref=0.; spk=[]
    n=len(I_ext_arr); V_r=np.zeros(n)
    for i,I in enumerate(I_ext_arr):
        if ref>0: ref-=dt; V=Vreset
        else:
            dV=(-(V-EL)*gL+I)/Cm
            V+=dV*dt
            if V>=Vthr: V=Vpeak; spk.append(i*dt); ref=tref
        V_r[i]=V
    return dict(V=V_r, spikes=np.array(spk))

def lif_fi(I_vals, T_ms=1000, dt=0.1):
    out=[]
    for Iv in I_vals:
        nn=int(T_ms/dt); Ia=np.full(nn,Iv)
        res=lif_simulate(Ia,dt=dt); out.append(len(res["spikes"])/(T_ms*1e-3))
    return np.array(out)

I_range_lif=np.linspace(0,600,40)
fI_lif=lif_fi(I_range_lif)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  AdEx MODEL — multiple firing patterns
# ═══════════════════════════════════════════════════════════════════════════
print("3 / 8  AdEx firing patterns …")

def adex_simulate(I_ext_arr, dt=0.1,
                  Cm=200.,gL=10.,EL=-70.,VT=-55.,Vthr=-30.,Vreset=-58.,Vpeak=30.,
                  dT=2.,tref=2.,a=4.,b=80.5,tauw=144.):
    V=EL; w=0.; ref=0.; spk=[]
    n=len(I_ext_arr); V_r=np.zeros(n); w_r=np.zeros(n)
    tref_steps=max(1,int(tref/dt))
    ref_ctr_a=0
    for i,I in enumerate(I_ext_arr):
        if ref_ctr_a>0:
            ref_ctr_a-=1; V=Vreset
        else:
            exp_t=gL*dT*np.exp(np.clip((V-VT)/dT,-10,10))
            dV=(-(V-EL)*gL+exp_t-w+I)/Cm
            dw=(a*(V-EL)-w)/tauw
            V+=dV*dt; w+=dw*dt
            if V>=Vthr:
                V_r[i]=Vpeak; w_r[i]=w   # record peak
                w+=b; spk.append(i*dt)
                V=Vreset; ref_ctr_a=tref_steps
                continue
        V_r[i]=V; w_r[i]=w
    return dict(V=V_r,w=w_r,spikes=np.array(spk))

# Preset params from Naud et al. 2008 (Table 1)
ADEX_PRESETS = {
    "RS":  dict(a=4.,  b=80.5,  tauw=144., Vreset=-58.),   # Regular Spiking
    "IB":  dict(a=4.,  b=80.5,  tauw=16.,  Vreset=-58.),   # Intrinsic Bursting
    "CH":  dict(a=4.,  b=80.5,  tauw=5.,   Vreset=-58.),   # Chattering
    "FS":  dict(a=0.,  b=0.,    tauw=40.,  Vreset=-58.),   # Fast Spiking
    "LTS": dict(a=8.,  b=200.,  tauw=200., Vreset=-58.),   # Low-Threshold Spiking
}

T_adex=600.; dt_adex=0.1; n_adex=int(T_adex/dt_adex)
I_adex=np.zeros(n_adex); I_adex[int(100/dt_adex):int(550/dt_adex)]=500.
adex_results={}
for name,kw in ADEX_PRESETS.items():
    r=adex_simulate(I_adex,dt=dt_adex,**kw)
    r["t"]=np.arange(n_adex)*dt_adex
    spk=r["spikes"]
    isi=np.diff(spk) if len(spk)>1 else np.array([])
    r["rate"]=len(spk)/(T_adex*1e-3)
    r["cv"]  =float(np.std(isi)/np.mean(isi)) if len(isi)>1 else np.nan
    adex_results[name]=r

# AdEx f-I (RS preset)
def adex_fi(I_vals, T_ms=800, dt=0.1, **kw):
    out=[]
    for Iv in I_vals:
        nn=int(T_ms/dt); Ia=np.full(nn,Iv)
        res=adex_simulate(Ia,dt=dt,**kw); out.append(len(res["spikes"])/(T_ms*1e-3))
    return np.array(out)

I_range_adex=np.linspace(0,600,40)
fI_adex_RS=adex_fi(I_range_adex,**ADEX_PRESETS["RS"])
fI_adex_FS=adex_fi(I_range_adex,**ADEX_PRESETS["FS"])


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SYNAPSE KINETICS
# ═══════════════════════════════════════════════════════════════════════════
print("4 / 8  Synapse kinetics …")

def run_synapse(alpha, beta, T_max, t_pulse, g_max, T_ms=300., dt=0.05, V_post=-65.):
    n=int(T_ms/dt); t=np.arange(n)*dt; g=np.zeros(n); r=0.; t_trans=1e9
    spike_ms=10.  # single pre-spike at 10 ms
    for i in range(n):
        ti=t[i]
        if abs(ti-spike_ms)<dt: t_trans=0.
        else: t_trans+=dt
        T=T_max if t_trans<t_pulse else 0.
        dr=alpha*T*(1-r)-beta*r; r=np.clip(r+dr*dt,0,1); g[i]=g_max*r
    return dict(t=t, g=g)

# GABA-B needs 4-state G-protein model
def run_gabab(g_max=0.5, T_ms=600., dt=0.05):
    K1,K2,K3,K4,Kd,n_hill=0.52,0.0013,0.098,0.033,100.,4.
    T_max,t_pulse=0.5,0.3
    n=int(T_ms/dt); t=np.arange(n)*dt; g=np.zeros(n); R=0.; G=0.; t_trans=1e9
    for i in range(n):
        ti=t[i]
        if abs(ti-10.)<dt: t_trans=0.
        else: t_trans+=dt
        T=T_max if t_trans<t_pulse else 0.
        dR=K1*T*(1-R)-K2*R; dG=K3*R-K4*G
        R=np.clip(R+dR*dt,0,1); G=max(0.,G+dG*dt)
        g[i]=g_max*(G**n_hill)/(G**n_hill+Kd)
    return dict(t=t, g=g)

syn_data = {
    "AMPA":   run_synapse(1.1, 0.19, 1.0, 1.0, 0.5),
    "NMDA":   run_synapse(0.072,0.0066,1.0,1.0,0.5, T_ms=500.),
    "GABA-A": run_synapse(5.0, 0.18, 1.0, 1.0, 0.5),
    "GABA-B": run_gabab(),
}


# ═══════════════════════════════════════════════════════════════════════════
# 5.  STDP WEIGHT DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════
print("5 / 8  STDP weight evolution …")

rng_stdp=np.random.default_rng(42)
A_plus,A_minus=0.01,0.0105; tau_plus,tau_minus=20.,20.
mu=1.0; w_min,w_max=0.,1.

def run_stdp_population(n_syn=100, T_s=200., dt_ms=0.5, rate=20.):
    """100 independent synapses, Poisson pre/post at `rate` Hz."""
    n_steps=int(T_s*1e3/dt_ms)
    p_spike=rate*dt_ms*1e-3
    W=np.full(n_syn,.5); W_hist=np.zeros((n_steps,n_syn))
    x_pre=np.zeros(n_syn); x_post=np.zeros(n_syn)
    for i in range(n_steps):
        x_pre *=np.exp(-dt_ms/tau_plus); x_post*=np.exp(-dt_ms/tau_minus)
        pre_spk =rng_stdp.random(n_syn)<p_spike
        post_spk=rng_stdp.random(n_syn)<p_spike
        x_pre[pre_spk]+=1.; x_post[post_spk]+=1.
        dw=np.zeros(n_syn)
        dw+=A_plus *(w_max-W)**mu*x_pre *post_spk
        dw-=A_minus*(W-w_min)**mu*x_post*pre_spk
        W=np.clip(W+dw,w_min,w_max); W_hist[i]=W
    return W_hist

print("   running 100-synapse STDP (200 s)…")
W_hist=run_stdp_population()
# Single representative synapse trace
w_trace=W_hist[:,0]
w_final_dist=W_hist[-1]   # final weight distribution across 100 synapses


# ═══════════════════════════════════════════════════════════════════════════
# 6.  E/I RECURRENT SPIKING NETWORK  (vectorised NumPy LIF)
# ═══════════════════════════════════════════════════════════════════════════
print("6 / 8  E/I network (200E + 50I, 600 ms) …")

rng_net=np.random.default_rng(7)
N_E,N_I=200,50; N=N_E+N_I
dt_n,T_n=0.2,600.     # ms
n_steps_n=int(T_n/dt_n)
t_net=np.arange(n_steps_n)*dt_n

Cm_n,gL_n,EL_n=200.,10.,-70.
Vthr_n,Vreset_n,Vpeak_n=-50.,-60.,30.
t_ref_steps=int(2./dt_n)
tau_s=3.0          # synaptic decay ms

# Connectivity
p_conn=0.12
conn=rng_net.random((N,N))<p_conn
np.fill_diagonal(conn,False)
W_exc=np.zeros((N,N))   # E→* AMPA weights (positive)
W_inh=np.zeros((N,N))   # I→* GABA weights (positive, sign handled via E_rev)
W_exc[:,:N_E][conn[:,:N_E]]=0.30    # nS
W_inh[:,N_E:][conn[:,N_E:]]=2.80    # nS  (reversal handles inhibition)

E_rev_exc =  0.0   # mV AMPA
E_rev_inh = -75.0  # mV GABA-A

V_net=np.full(N,EL_n)
g_exc=np.zeros(N); g_inh=np.zeros(N)
ref_ctr=np.zeros(N,dtype=int)
V_rec_net=np.zeros((N,n_steps_n),dtype=np.float32)
spike_list=[[] for _ in range(N)]

I_dc=280.; I_noise_sig=60.

fired_prev=np.zeros(N,dtype=bool)

for i in range(n_steps_n):
    decay=np.exp(-dt_n/tau_s)
    g_exc*=decay; g_inh*=decay
    # W[post,pre]: sum over pre → W @ fired_prev gives input per post-neuron
    g_exc+=W_exc @ fired_prev.astype(float)
    g_inh+=W_inh @ fired_prev.astype(float)

    I_syn=(g_exc*(V_net-E_rev_exc) + g_inh*(V_net-E_rev_inh))
    I_noise=rng_net.normal(0,I_noise_sig,N)
    I_total=I_dc+I_noise-I_syn

    in_ref=(ref_ctr>0)
    dV=(-(V_net-EL_n)*gL_n+I_total)/Cm_n
    V_net=np.where(in_ref,Vreset_n,V_net+dV*dt_n)
    ref_ctr=np.where(in_ref,ref_ctr-1,ref_ctr)

    fired=V_net>=Vthr_n
    V_net[fired]=Vpeak_n
    ref_ctr[fired]=t_ref_steps
    for nid in np.where(fired)[0]:
        spike_list[nid].append(t_net[i])
    fired_prev=fired
    V_rec_net[:,i]=V_net.astype(np.float32)

# population rates (spike count in 5-ms bins → Hz)
pop_E=np.zeros(n_steps_n); pop_I=np.zeros(n_steps_n)
for i in range(n_steps_n):
    pass   # computed below via PSTH

bin_ms=5.; bins=np.arange(0,T_n+bin_ms,bin_ms)
E_spk=np.concatenate([spike_list[i] for i in range(N_E)] or [[]])
I_spk=np.concatenate([spike_list[N_E+i] for i in range(N_I)] or [[]])
e_cnt,_=np.histogram(E_spk,bins=bins)
i_cnt,_=np.histogram(I_spk,bins=bins)
t_bins=0.5*(bins[:-1]+bins[1:])
e_rate=e_cnt/(N_E*bin_ms*1e-3); i_rate=i_cnt/(N_I*bin_ms*1e-3)
LFP=np.mean(V_rec_net[:N_E],axis=0)

total_e_spk=sum(len(spike_list[i]) for i in range(N_E))
total_i_spk=sum(len(spike_list[N_E+i]) for i in range(N_I))
print(f"   E: {total_e_spk} spikes ({total_e_spk/(N_E*T_n*1e-3):.1f} Hz mean)  "
      f"I: {total_i_spk} spikes ({total_i_spk/(N_I*T_n*1e-3):.1f} Hz mean)")


# ═══════════════════════════════════════════════════════════════════════════
# 7.  SPIKE TRAIN ANALYSIS (population ISI stats)
# ═══════════════════════════════════════════════════════════════════════════
print("7 / 8  Spike-train statistics …")

cv_all=[]
for i in range(N_E):
    st=np.array(spike_list[i])
    if len(st)>2:
        isi=np.diff(st)
        if np.mean(isi)>0:
            cv_all.append(np.std(isi)/np.mean(isi))
cv_pop=np.array(cv_all)

# Pool E-cell spikes for ISI distribution
pool_spk=E_spk
isi_pool=[]
for i in range(N_E):
    st=np.array(spike_list[i])
    if len(st)>1: isi_pool.extend(np.diff(st).tolist())
isi_pool=np.array(isi_pool)

# Fit gamma and exponential
if len(isi_pool)>10:
    bins_isi=np.linspace(0,np.percentile(isi_pool,99),50)
    gam_a,_,gam_sc=stats.gamma.fit(isi_pool,floc=0)
    exp_sc=np.mean(isi_pool)
    isi_fits=dict(gamma=dict(a=gam_a,scale=gam_sc),
                  exponential=dict(scale=exp_sc))
    best_fit="gamma"
else:
    bins_isi=np.array([0,1]); isi_fits={"gamma":{"a":1.,"scale":1.},
                                         "exponential":{"scale":1.}}; best_fit="N/A"

# PSD of LFP (Welch)
fs_lfp=1e3/dt_n   # Hz
f_psd,psd_lfp=signal.welch(LFP,fs=fs_lfp,nperseg=min(len(LFP),2048),scaling="density")
band_powers={}
for bname,(lo,hi) in [("delta",(0.5,4)),("theta",(4,8)),
                       ("alpha",(8,13)),("beta",(13,30)),("gamma",(30,100))]:
    mask=(f_psd>=lo)&(f_psd<=hi)
    band_powers[bname]=float(np.trapezoid(psd_lfp[mask],f_psd[mask])) if mask.any() else 0.
dom_f=float(f_psd[np.argmax(psd_lfp[(f_psd>1)&(f_psd<200)])+np.searchsorted(f_psd,1)])


# ═══════════════════════════════════════════════════════════════════════════
# 8.  DATABASE  (SQLite via SQLAlchemy)
# ═══════════════════════════════════════════════════════════════════════════
print("8 / 8  Persisting results to database …")

from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Session
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "compneuro.db"

class Base(DeclarativeBase): pass

class SimulationRecord(Base):
    __tablename__ = "simulation_records"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(120), nullable=False)
    model_type  = Column(String(60),  nullable=False)
    duration_ms = Column(Float)
    dt_ms       = Column(Float)
    n_spikes    = Column(Integer)
    firing_rate_hz = Column(Float)
    cv_isi      = Column(Float)
    notes       = Column(Text)
    params_json = Column(JSON)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class SpikeRecord(Base):
    __tablename__ = "spike_records"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    sim_id    = Column(Integer)
    neuron_id = Column(Integer)
    time_ms   = Column(Float)

class NetworkStateRecord(Base):
    __tablename__ = "network_state_records"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    sim_id        = Column(Integer)
    E_rate_hz     = Column(Float)
    I_rate_hz     = Column(Float)
    mean_cv_isi   = Column(Float)
    synchrony_chi = Column(Float)
    network_state = Column(String(10))
    dominant_freq_hz = Column(Float)
    n_E_spikes    = Column(Integer)
    n_I_spikes    = Column(Integer)

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base.metadata.create_all(engine)

with Session(engine) as sess:
    # HH record
    hh_cv = float(np.std(np.diff(hh["spikes"]))/np.mean(np.diff(hh["spikes"]))) \
            if len(hh["spikes"])>2 else None
    sess.add(SimulationRecord(
        name="HH_single_neuron", model_type="HodgkinHuxley",
        duration_ms=T, dt_ms=dt, n_spikes=len(hh["spikes"]),
        firing_rate_hz=len(hh["spikes"])/(T*1e-3), cv_isi=hh_cv,
        params_json=dict(C_m=C_m,g_Na=g_Na,g_K=g_K,g_L=g_L,
                         E_Na=E_Na,E_K=E_K,E_L=E_L,I_ext=10.0),
        notes="Canonical HH 1952 squid axon parameters at 6.3°C"
    ))
    # LIF record
    lif_demo=lif_simulate(np.full(int(500/.1),400.),dt=0.1)
    sess.add(SimulationRecord(
        name="LIF_single_neuron", model_type="LIF",
        duration_ms=500., dt_ms=0.1, n_spikes=len(lif_demo["spikes"]),
        firing_rate_hz=len(lif_demo["spikes"])/(500.*1e-3),
        params_json=dict(Cm=200.,gL=10.,EL=-70.,Vthr=-55.,Vreset=-70.,tref=2.),
    ))
    # AdEx records
    for preset,res in adex_results.items():
        sess.add(SimulationRecord(
            name=f"AdEx_{preset}", model_type="AdaptiveExponentialIF",
            duration_ms=T_adex, dt_ms=dt_adex,
            n_spikes=len(res["spikes"]), firing_rate_hz=res["rate"],
            cv_isi=float(res["cv"]) if not np.isnan(res["cv"]) else None,
            params_json=ADEX_PRESETS[preset],
            notes=f"AdEx preset: {preset}"
        ))
    # Network record
    mean_cv=float(np.mean(cv_pop)) if len(cv_pop) else None
    # synchrony χ
    bins_sync=np.arange(0,T_n+1.,1.)
    cnt_pop=np.array([np.histogram(spike_list[i],bins=bins_sync)[0] for i in range(N_E)])
    pop_rate_bin=cnt_pop.mean(axis=0)
    var_pop=np.var(pop_rate_bin)
    mean_var_single=np.mean([np.var(cnt_pop[i]) for i in range(N_E)])
    chi=float(np.sqrt(var_pop/max(mean_var_single,1e-12)))
    state="AI" if chi<0.15 and (mean_cv is None or mean_cv>0.5) else \
          "SR" if chi>0.5 else "SIf"
    net_sim=SimulationRecord(
        name="EI_network_LIF", model_type="SpikingNetwork",
        duration_ms=T_n, dt_ms=dt_n,
        n_spikes=total_e_spk+total_i_spk,
        firing_rate_hz=(total_e_spk+total_i_spk)/((N_E+N_I)*T_n*1e-3),
        cv_isi=mean_cv,
        params_json=dict(N_E=N_E,N_I=N_I,p_conn=p_conn,
                         W_EE=0.30,W_IE=-2.50,I_dc=I_dc,tau_s=tau_s),
        notes=f"State={state}, χ={chi:.3f}"
    )
    sess.add(net_sim); sess.flush()
    sess.add(NetworkStateRecord(
        sim_id=net_sim.id,
        E_rate_hz=total_e_spk/(N_E*T_n*1e-3),
        I_rate_hz=total_i_spk/(N_I*T_n*1e-3),
        mean_cv_isi=mean_cv, synchrony_chi=chi,
        network_state=state, dominant_freq_hz=dom_f,
        n_E_spikes=total_e_spk, n_I_spikes=total_i_spk,
    ))
    sess.commit()
print(f"   Database saved → {DB_PATH}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating figures …\n")

# ─── Fig 1: HH Action Potential Anatomy ──────────────────────────────────
print("  Fig 1 – HH AP anatomy")
fig,axes=plt.subplots(4,1,figsize=(9,10),sharex=True)
fig.suptitle("Hodgkin-Huxley Neuron — Action Potential Anatomy",
             fontsize=13,fontweight="bold",y=0.99)
t_=hh["t"]; V_=hh["V"]; spk=hh["spikes"]
t_end=min(spk[2]+20,t_[-1]) if len(spk)>=3 else t_[-1]
mask=t_<=t_end

ax=axes[0]
ax.plot(t_[mask],V_[mask],color=C["V"],lw=1.8)
ax.axhline(-65,color="gray",lw=0.8,ls="--",alpha=0.5,label="V_rest  –65 mV")
ax.axhline(-55,color="salmon",lw=0.8,ls=":",alpha=0.8,label="Threshold –55 mV")
[ax.axvline(s,color="#aaa",lw=0.5,alpha=0.4) for s in spk if s<=t_end]
ax.set_ylabel("V (mV)"); ax.set_ylim(-85,55); ax.legend(frameon=False,ncol=2,fontsize=8)
_tag(ax,"A")

ax=axes[1]
ax.plot(t_[mask],hh["m"][mask],color=C["m"],label="m  Na act.")
ax.plot(t_[mask],hh["h"][mask],color=C["h"],label="h  Na inact.")
ax.plot(t_[mask],hh["n"][mask],color=C["n"],label="n  K act.")
ax.set_ylabel("Gate value"); ax.set_ylim(-0.05,1.05)
ax.legend(frameon=False,ncol=3,fontsize=8); _tag(ax,"B")

ax=axes[2]
ax.plot(t_[mask],hh["m"][mask]**3*hh["h"][mask],color=C["m"],label=r"$m^3h$ Na conductance")
ax.plot(t_[mask],hh["n"][mask]**4,color=C["n"],label=r"$n^4$ K conductance")
ax.set_ylabel("Norm. conductance"); ax.set_ylim(-0.02,1.02)
ax.legend(frameon=False,ncol=2,fontsize=8); _tag(ax,"C")

ax=axes[3]
ax.plot(t_[mask],hh["INa"][mask],color=C["m"],label=r"$I_{Na}$")
ax.plot(t_[mask],hh["IK"][mask], color=C["n"],label=r"$I_K$")
ax.plot(t_[mask],hh["IL"][mask], color="#999",ls="--",label=r"$I_L$")
ax.axhline(0,color="k",lw=0.5)
ax.set_ylabel(r"Current (µA/cm²)"); ax.set_xlabel("Time (ms)")
ax.legend(frameon=False,ncol=3,fontsize=8); _tag(ax,"D")
fig.align_ylabels(axes)
plt.tight_layout(); fig.savefig(FIGDIR/"fig1_hh_action_potential.png"); plt.close(fig)

# ─── Fig 2: Phase Plane ───────────────────────────────────────────────────
print("  Fig 2 – Phase plane + gating curves")
V_r_pp=np.linspace(-80,50,500)
m_inf=np.array([alpha_m(v)/(alpha_m(v)+beta_m(v)) for v in V_r_pp])
h_inf=np.array([alpha_h(v)/(alpha_h(v)+beta_h(v)) for v in V_r_pp])
n_inf=np.array([alpha_n(v)/(alpha_n(v)+beta_n(v)) for v in V_r_pp])

fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Phase-Plane Analysis — Hodgkin-Huxley",fontsize=13,fontweight="bold")
ax=axes[0]
ax.plot(V_r_pp,n_inf,"b-",lw=2,label=r"$n_\infty(V)$ nullcline")
ax.plot(hh["V"],hh["n"],color="#555",lw=0.6,alpha=0.7,label="Limit cycle trajectory")
ax.set_xlabel("V (mV)"); ax.set_ylabel("K⁺ activation  n")
ax.set_xlim(-80,50); ax.set_ylim(0,1)
ax.legend(frameon=False); ax.set_title("V–n Phase Plane"); _tag(ax,"A")

ax=axes[1]
ax.plot(V_r_pp,m_inf,color=C["m"],lw=2,label=r"$m_\infty(V)$")
ax.plot(V_r_pp,h_inf,color=C["h"],lw=2,label=r"$h_\infty(V)$")
ax.plot(V_r_pp,n_inf,color=C["n"],lw=2,label=r"$n_\infty(V)$")
ax.set_xlabel("V (mV)"); ax.set_ylabel("Steady-state value")
ax.set_ylim(-0.02,1.02); ax.legend(frameon=False)
ax.set_title("Gating Variable Steady-States  (Boltzmann curves)"); _tag(ax,"B")
plt.tight_layout(); fig.savefig(FIGDIR/"fig2_phase_plane.png"); plt.close(fig)

# ─── Fig 3: f-I curves ────────────────────────────────────────────────────
print("  Fig 3 – f-I curves (HH / LIF / AdEx)")
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Frequency-Current (f-I) Curves — Model Comparison",fontsize=13,fontweight="bold")
ax=axes[0]
ax.plot(I_range_hh,  fI_hh,     color="#d7191c",lw=2,label="Hodgkin-Huxley")
ax.plot(I_range_lif, fI_lif,    color="#2c7bb6",lw=2,ls="--",label="LIF")
ax.plot(I_range_adex,fI_adex_RS,color="#1a9641",lw=2,ls="-.",label="AdEx-RS")
ax.plot(I_range_adex,fI_adex_FS,color="#ff7f00",lw=2,ls=":",label="AdEx-FS")
ax.set_xlabel(r"Input current"); ax.set_ylabel("Firing rate (Hz)")
ax.set_title("f-I Gain Curves"); ax.legend(frameon=False); _tag(ax,"A")

ax=axes[1]
for name,I_v,f_v,col,ls in [
    ("HH",         I_range_hh,  fI_hh,     "#d7191c","-"),
    ("LIF",        I_range_lif, fI_lif,    "#2c7bb6","--"),
    ("AdEx-RS",    I_range_adex,fI_adex_RS,"#1a9641","-."),
    ("AdEx-FS",    I_range_adex,fI_adex_FS,"#ff7f00",":")]:
    mask=f_v>1.
    if mask.sum()>2:
        ax.semilogy(I_v[mask],f_v[mask],color=col,ls=ls,lw=2,label=name)
ax.set_xlabel("Input current"); ax.set_ylabel("Firing rate (Hz, log)")
ax.set_title("Supra-threshold Gain (semi-log)"); ax.legend(frameon=False); _tag(ax,"B")
plt.tight_layout(); fig.savefig(FIGDIR/"fig3_fi_curves.png"); plt.close(fig)

# ─── Fig 4: AdEx firing patterns ─────────────────────────────────────────
print("  Fig 4 – AdEx firing patterns")
presets=list(adex_results.keys()); ncols=3; nrows=2
fig,axes=plt.subplots(nrows,ncols,figsize=(13,8))
axes=axes.flatten()
fig.suptitle("AdEx Model — Biologically Realistic Firing Patterns (Naud et al. 2008)",
             fontsize=13,fontweight="bold")
full_names={"RS":"Regular Spiking","IB":"Intrinsic Bursting",
            "CH":"Chattering","FS":"Fast Spiking","LTS":"Low-Threshold Spiking"}
for idx,preset in enumerate(presets):
    res=adex_results[preset]; ax=axes[idx]
    ax.plot(res["t"],res["V"],color=C["V"],lw=1.1)
    I_trace=np.zeros_like(res["t"])
    I_trace[int(100/dt_adex):int(550/dt_adex)]=500.
    ax2=ax.twinx(); ax2.fill_between(res["t"],I_trace,alpha=0.12,color="#aaa")
    ax2.set_ylabel("I (pA)",fontsize=7); ax2.set_ylim(-100,1500)
    ax2.tick_params(labelsize=7)
    cv_str=f", CV={res['cv']:.2f}" if not np.isnan(res['cv']) else ""
    ax.set_title(f"{full_names[preset]}\n{res['rate']:.1f} Hz{cv_str}")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("V (mV)"); ax.set_ylim(-85,50)
    _tag(ax,chr(65+idx))
for j in range(len(presets),len(axes)): axes[j].set_visible(False)
plt.tight_layout(); fig.savefig(FIGDIR/"fig4_adex_patterns.png"); plt.close(fig)

# ─── Fig 5: Synaptic conductances ─────────────────────────────────────────
print("  Fig 5 – Synaptic conductance kinetics")
fig,axes=plt.subplots(2,2,figsize=(11,8))
axes=axes.flatten()
fig.suptitle("Synaptic Conductance Kinetics — Single Presynaptic Spike at t=10 ms",
             fontsize=13,fontweight="bold")
descr={"AMPA":"Fast excitatory (Glu)\nτ_peak ≈ 1 ms, τ_decay ≈ 5 ms",
       "NMDA":"Slow excitatory (Glu) + Mg²⁺ block\nτ_decay ≈ 150 ms",
       "GABA-A":"Fast inhibitory (Cl⁻)\nτ_decay ≈ 5 ms",
       "GABA-B":"Slow inhibitory (K⁺, metabotropic)\nτ_decay ≈ 200 ms"}
for idx,(name,dat) in enumerate(syn_data.items()):
    ax=axes[idx]; col=list(C.values())[list(["AMPA","NMDA","GABA-A","GABA-B"]).index(name)]
    ax.plot(dat["t"],dat["g"]*1e3,color=col,lw=2)
    ax.axvline(10,color="k",lw=0.8,ls=":",alpha=0.5)
    ax.set_title(f"{name}\n{descr[name]}",fontsize=9)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Conductance (pS)")
    pk=float(np.max(dat["g"]))*1e3; tpk=float(dat["t"][np.argmax(dat["g"])])
    ax.annotate(f"Peak {pk:.0f} pS\nt={tpk:.1f} ms",
                xy=(tpk,pk),xytext=(tpk+dat["t"][-1]*0.12,pk*0.75),
                arrowprops=dict(arrowstyle="->",lw=0.8),fontsize=8)
    _tag(ax,chr(65+idx))
plt.tight_layout(); fig.savefig(FIGDIR/"fig5_synaptic_conductances.png"); plt.close(fig)

# ─── Fig 6: NMDA Mg2+ block ───────────────────────────────────────────────
print("  Fig 6 – NMDA Mg²⁺ block")
V_pp=np.linspace(-90,50,500)
Mg_concs=[0.,0.5,1.,2.]
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("NMDA Receptor — Voltage-Dependent Mg²⁺ Block",fontsize=13,fontweight="bold")
cmap=plt.cm.plasma
cols_mg=[cmap(i/(len(Mg_concs)-1)) for i in range(len(Mg_concs))]
ax=axes[0]
for Mg,col in zip(Mg_concs,cols_mg):
    B=1./(1.+np.exp(-0.062*V_pp)*Mg/3.57)
    ax.plot(V_pp,B,color=col,lw=2,label=f"[Mg²⁺]={Mg} mM")
ax.axvline(-65,color="gray",ls="--",lw=0.8,alpha=0.6)
ax.text(-63,0.05,"V_rest",fontsize=8,color="gray")
ax.set_xlabel("V (mV)"); ax.set_ylabel("Unblock factor B(V)")
ax.set_title(r"$B(V)=\frac{1}{1+e^{-0.062V}[\mathrm{Mg}]/3.57}$")
ax.legend(frameon=False); _tag(ax,"A")

ax=axes[1]
for Mg,col in zip(Mg_concs,cols_mg):
    B=1./(1.+np.exp(-0.062*V_pp)*Mg/3.57)
    I_NMDA=1.*B*(V_pp-0.)
    ax.plot(V_pp,I_NMDA,color=col,lw=2,label=f"[Mg²⁺]={Mg} mM")
ax.axhline(0,color="k",lw=0.5); ax.axvline(0,color="k",lw=0.5)
ax.set_xlabel("V (mV)"); ax.set_ylabel("I_NMDA (norm.)")
ax.set_title("NMDA I-V Relationship\n(negative-resistance region with Mg²⁺)")
ax.legend(frameon=False,fontsize=8); _tag(ax,"B")
plt.tight_layout(); fig.savefig(FIGDIR/"fig6_nmda_mg_block.png"); plt.close(fig)

# ─── Fig 7: STDP ──────────────────────────────────────────────────────────
print("  Fig 7 – STDP")
fig,axes=plt.subplots(1,3,figsize=(14,5))
fig.suptitle("Spike-Timing Dependent Plasticity (STDP) — Bi & Poo 1998 / Song et al. 2000",
             fontsize=13,fontweight="bold")
# Learning window
ax=axes[0]
dts=np.linspace(-100,100,600)
dw_ltp= A_plus *np.exp(-dts/tau_plus) *(dts>0)
dw_ltd=-A_minus*np.exp( dts/tau_minus)*(dts<0)
dw=dw_ltp+dw_ltd
ax.fill_between(dts,dw,0,where=dw>0,color="#d7191c",alpha=0.3,label="LTP")
ax.fill_between(dts,dw,0,where=dw<0,color="#2c7bb6",alpha=0.3,label="LTD")
ax.plot(dts,dw,"k-",lw=2)
ax.axhline(0,color="k",lw=0.5); ax.axvline(0,color="k",lw=0.5,ls="--",alpha=0.5)
ax.set_xlabel(r"$\Delta t = t_{post}-t_{pre}$ (ms)"); ax.set_ylabel(r"$\Delta w$")
ax.set_title("STDP Learning Window\n(asymmetric, multiplicative rule)")
ax.legend(frameon=False); _tag(ax,"A")

ax=axes[1]
n_show=min(len(w_trace),4000)
ax.plot(np.linspace(0,200,n_show),w_trace[:n_show],color="#1a9641",lw=1.,alpha=0.9)
ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("Synaptic weight w")
ax.set_title("Single-Synapse Weight Trajectory\n(Poisson pre & post, 20 Hz)")
ax.set_ylim(0,1); ax.axhline(0.5,color="gray",lw=0.7,ls="--",alpha=0.5); _tag(ax,"B")

ax=axes[2]
ax.hist(w_final_dist,bins=30,color="#ff7f00",edgecolor="white",alpha=0.85,density=True)
ax.axvline(w_min,color="k",lw=0.8,ls="--",alpha=0.6)
ax.axvline(w_max,color="k",lw=0.8,ls="--",alpha=0.6)
ax.set_xlabel("Final weight value"); ax.set_ylabel("Probability density")
ax.set_title(f"Steady-State Weight Distribution\n(N=100 synapses; μ={w_final_dist.mean():.2f})")
_tag(ax,"C")
plt.tight_layout(); fig.savefig(FIGDIR/"fig7_stdp.png"); plt.close(fig)

# ─── Fig 8: Network raster + rates + LFP ──────────────────────────────────
print("  Fig 8 – Network dynamics")
fig=plt.figure(figsize=(13,12))
gs=gridspec.GridSpec(4,2,figure=fig,hspace=0.5,wspace=0.32)
fig.suptitle(f"E/I Recurrent Spiking Network Dynamics  ({N_E}E + {N_I}I neurons)\n"
             f"Network state: {state}   χ={chi:.3f}   Mean CV-ISI={mean_cv:.2f}" if mean_cv else
             f"E/I Recurrent Spiking Network Dynamics",
             fontsize=12,fontweight="bold")

# (A) Raster
ax=fig.add_subplot(gs[0,:])
n_show_ras=min(N,250)
for nid in range(n_show_ras):
    st=np.array(spike_list[nid])
    col=C["E"] if nid<N_E else C["I"]
    if len(st): ax.scatter(st,np.full(len(st),nid),s=0.7,c=col,alpha=0.55,linewidths=0)
ax.axhline(N_E-0.5,color="k",lw=0.8,ls="--",alpha=0.3)
ax.set_ylabel("Neuron index"); ax.set_xlabel("Time (ms)")
ax.set_title(f"Spike Raster  (red = E cells, blue = I cells, showing first {n_show_ras})")
ax.set_xlim(0,T_n); _tag(ax,"A")

# (B) Population rates
ax=fig.add_subplot(gs[1,:])
sm=max(1,int(10./bin_ms))
ax.plot(t_bins,uniform_filter1d(e_rate,sm),color=C["E"],lw=1.6,label="E population")
ax.plot(t_bins,uniform_filter1d(i_rate,sm),color=C["I"],lw=1.6,ls="--",label="I population")
ax.set_ylabel("Rate (Hz)"); ax.set_xlabel("Time (ms)")
ax.set_title(f"Population Firing Rates (smoothed {int(sm*bin_ms)} ms window)")
ax.legend(frameon=False,ncol=2); ax.set_xlim(0,T_n); _tag(ax,"B")

# (C) LFP proxy
ax=fig.add_subplot(gs[2,:])
lfp_sm=uniform_filter1d(LFP,int(5./dt_n))
ax.plot(t_net,lfp_sm,color=C["LFP"],lw=0.8,alpha=0.9)
ax.set_ylabel("Mean E-cell V (mV)"); ax.set_xlabel("Time (ms)")
ax.set_title("Local Field Potential Proxy"); ax.set_xlim(0,T_n); _tag(ax,"C")

# (D) Per-neuron rate distribution
ax=fig.add_subplot(gs[3,0])
T_s_n=T_n*1e-3
rE=np.array([len(spike_list[i])/T_s_n for i in range(N_E)])
rI=np.array([len(spike_list[N_E+i])/T_s_n for i in range(N_I)])
ax.hist(rE,bins=20,color=C["E"],alpha=0.7,density=True,label=f"E  μ={rE.mean():.1f} Hz")
ax.hist(rI,bins=20,color=C["I"],alpha=0.7,density=True,label=f"I  μ={rI.mean():.1f} Hz")
ax.set_xlabel("Firing rate (Hz)"); ax.set_ylabel("Density")
ax.set_title("Single-neuron Rate Distribution"); ax.legend(frameon=False); _tag(ax,"D")

# (E) E-I cross-correlogram
ax=fig.add_subplot(gs[3,1])
lag_pts=min(int(200./bin_ms),len(e_rate)-1)
from scipy.signal import correlate as sig_corr
cc=sig_corr(e_rate-e_rate.mean(),i_rate-i_rate.mean(),mode="full")
ctr=len(cc)//2; lags_cc=np.arange(-lag_pts,lag_pts+1)*bin_ms
norm_cc=np.std(e_rate)*np.std(i_rate)*len(e_rate)+1e-12
ax.plot(lags_cc,cc[ctr-lag_pts:ctr+lag_pts+1]/norm_cc,color="#984ea3",lw=1.4)
ax.axvline(0,color="k",lw=0.5,ls="--",alpha=0.5); ax.axhline(0,color="k",lw=0.5)
ax.set_xlabel("Lag (ms)"); ax.set_ylabel("Norm. cross-corr.")
ax.set_title("E–I Population Rate Cross-Correlogram"); _tag(ax,"E")

fig.savefig(FIGDIR/"fig8_network_dynamics.png"); plt.close(fig)

# ─── Fig 9: ISI analysis ──────────────────────────────────────────────────
print("  Fig 9 – ISI distribution & CV-ISI histogram")
fig,axes=plt.subplots(1,3,figsize=(14,5))
fig.suptitle("Inter-Spike Interval Analysis — E-cell Population",fontsize=13,fontweight="bold")

ax=axes[0]
if len(isi_pool)>5:
    ax.hist(isi_pool,bins=bins_isi,density=True,color="#2c7bb6",alpha=0.75,
            edgecolor="white",label="Observed ISI")
    x_fit=np.linspace(0,np.percentile(isi_pool,99),300)
    ax.plot(x_fit,stats.gamma.pdf(x_fit,isi_fits["gamma"]["a"],
                                   scale=isi_fits["gamma"]["scale"]),
            "r-",lw=2,label=f"Gamma (a={isi_fits['gamma']['a']:.2f})")
    ax.plot(x_fit,stats.expon.pdf(x_fit,scale=isi_fits["exponential"]["scale"]),
            "g--",lw=1.5,label="Exponential (Poisson)")
ax.set_xlabel("ISI (ms)"); ax.set_ylabel("Probability density")
ax.set_title(f"ISI Distribution\n(pooled E-cells, best fit: {best_fit})")
ax.legend(frameon=False,fontsize=8); _tag(ax,"A")

ax=axes[1]
if len(isi_pool)>5:
    isi_s=np.sort(isi_pool); sv=1.-np.arange(len(isi_s))/len(isi_s)
    ax.semilogy(isi_s,sv,color="#2c7bb6",lw=2,label="Empirical")
    ax.semilogy(x_fit,1-stats.gamma.cdf(x_fit,isi_fits["gamma"]["a"],
                                          scale=isi_fits["gamma"]["scale"]),
                "r--",lw=1.5,label="Gamma fit")
    ax.semilogy(x_fit,np.exp(-x_fit/isi_fits["exponential"]["scale"]),
                "g:",lw=1.5,label="Exponential")
ax.set_xlabel("ISI (ms)"); ax.set_ylabel("P(ISI > x)  — survival fn.")
ax.set_title("ISI Survival Function\n(detects heavy tail / bursting)")
ax.legend(frameon=False,fontsize=8); _tag(ax,"B")

ax=axes[2]
cv_clean=cv_pop[np.isfinite(cv_pop)]
if len(cv_clean):
    ax.hist(cv_clean,bins=20,color="#ff7f00",edgecolor="white",alpha=0.85,density=True)
    ax.axvline(1.0,color="r",ls="--",lw=1.5,label="CV=1 (Poisson)")
    ax.axvline(cv_clean.mean(),color="k",lw=1.5,label=f"Mean={cv_clean.mean():.2f}")
ax.set_xlabel("CV-ISI"); ax.set_ylabel("Density")
ax.set_title(f"Population CV-ISI Distribution\nAI state when CV ≈ 1 (Brunel 2000)")
ax.legend(frameon=False); _tag(ax,"C")
plt.tight_layout(); fig.savefig(FIGDIR/"fig9_isi_analysis.png"); plt.close(fig)

# ─── Fig 10: Power Spectral Density ───────────────────────────────────────
print("  Fig 10 – LFP Power Spectral Density")
band_cols={"delta":"#4575b4","theta":"#74add1","alpha":"#abd9e9","beta":"#f46d43","gamma":"#d73027"}
band_ranges={"delta":(0.5,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,100)}
fig,axes=plt.subplots(1,2,figsize=(11,5))
fig.suptitle("Local Field Potential — Power Spectral Density (Welch)",
             fontsize=13,fontweight="bold")

ax=axes[0]
mask_f=(f_psd>=0.5)&(f_psd<=200)
ax.semilogy(f_psd[mask_f],psd_lfp[mask_f],color=C["LFP"],lw=1.5)
ax.axvline(dom_f,color="r",ls="--",lw=1,label=f"Dominant {dom_f:.1f} Hz")
for bname,(lo,hi) in band_ranges.items():
    bm=(f_psd>=lo)&(f_psd<=hi)
    if bm.any(): ax.fill_between(f_psd[bm],psd_lfp[bm],alpha=0.25,
                                  color=band_cols[bname],label=bname.capitalize())
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (mV²/Hz)")
ax.set_title("LFP Power Spectrum"); ax.legend(frameon=False,fontsize=8,ncol=2)
ax.set_xlim(0.5,200); _tag(ax,"A")

ax=axes[1]
names_b=list(band_powers.keys()); pows=[band_powers[n] for n in names_b]
bars=ax.bar(names_b,pows,color=[band_cols[n] for n in names_b],
            edgecolor="white",linewidth=0.8)
ax.set_ylabel("Band power (mV²)"); ax.set_title("Frequency Band Powers")
ax.set_xlabel("EEG Band (canonical ranges)")
for bar,val in zip(bars,pows):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.03,
            f"{val:.2e}",ha="center",va="bottom",fontsize=8)
_tag(ax,"B")
plt.tight_layout(); fig.savefig(FIGDIR/"fig10_psd.png"); plt.close(fig)


# ─── Summary figure (composite overview) ─────────────────────────────────
print("  Fig 11 – Summary composite")
fig=plt.figure(figsize=(16,12))
gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.55,wspace=0.38)
fig.suptitle("Computational Neuroscience — Neural Network Basics  |  Summary Overview",
             fontsize=14,fontweight="bold",y=1.01)

# AP waveform
ax=fig.add_subplot(gs[0,0])
mask_ap=hh["t"]<=t_end
ax.plot(hh["t"][mask_ap],hh["V"][mask_ap],color=C["V"],lw=1.5)
ax.axhline(-65,color="gray",lw=0.7,ls="--",alpha=0.5)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("V (mV)"); ax.set_title("HH Action Potential")
_tag(ax,"A")

# Phase plane
ax=fig.add_subplot(gs[0,1])
ax.plot(hh["V"],hh["n"],color="#555",lw=0.6,alpha=0.7)
ax.plot(V_r_pp,n_inf,"b-",lw=1.8,label=r"$n_\infty(V)$")
ax.set_xlabel("V (mV)"); ax.set_ylabel("n"); ax.set_title("Phase Plane (V–n)")
ax.legend(frameon=False,fontsize=8); _tag(ax,"B")

# f-I curves
ax=fig.add_subplot(gs[0,2])
ax.plot(I_range_hh,  fI_hh,     color="#d7191c",lw=2,label="HH")
ax.plot(I_range_lif, fI_lif,    color="#2c7bb6",lw=2,ls="--",label="LIF")
ax.plot(I_range_adex,fI_adex_RS,color="#1a9641",lw=2,ls="-.",label="AdEx")
ax.set_xlabel("Input current"); ax.set_ylabel("Rate (Hz)")
ax.set_title("f-I Curves"); ax.legend(frameon=False,fontsize=8); _tag(ax,"C")

# Synapse kinetics
ax=fig.add_subplot(gs[1,0])
for nm,col in zip(["AMPA","NMDA","GABA-A","GABA-B"],
                  [C["AMPA"],C["NMDA"],C["GABAA"],C["GABAB"]]):
    ax.plot(syn_data[nm]["t"],syn_data[nm]["g"]*1e3,color=col,lw=1.6,label=nm)
ax.axvline(10,color="k",lw=0.7,ls=":",alpha=0.5)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("Conductance (pS)")
ax.set_title("Synapse Kinetics"); ax.legend(frameon=False,fontsize=7,ncol=2); _tag(ax,"D")

# NMDA Mg block
ax=fig.add_subplot(gs[1,1])
for Mg,col in zip([0.,1.,2.],["#1a9641","#2c7bb6","#d7191c"]):
    B=1./(1.+np.exp(-0.062*V_pp)*Mg/3.57)
    ax.plot(V_pp,B,color=col,lw=1.8,label=f"{Mg} mM")
ax.set_xlabel("V (mV)"); ax.set_ylabel("Unblock B(V)")
ax.set_title("NMDA Mg²⁺ Block"); ax.legend(frameon=False,fontsize=8,title="[Mg²⁺]"); _tag(ax,"E")

# STDP window
ax=fig.add_subplot(gs[1,2])
ax.fill_between(dts,dw,0,where=dw>0,color="#d7191c",alpha=0.3)
ax.fill_between(dts,dw,0,where=dw<0,color="#2c7bb6",alpha=0.3)
ax.plot(dts,dw,"k-",lw=1.8)
ax.axhline(0,color="k",lw=0.5); ax.axvline(0,color="k",lw=0.5,ls="--",alpha=0.5)
ax.set_xlabel(r"$\Delta t$ (ms)"); ax.set_ylabel(r"$\Delta w$")
ax.set_title("STDP Learning Window"); _tag(ax,"F")

# Network raster (subset)
ax=fig.add_subplot(gs[2,0])
for nid in range(0,min(N,100),1):
    st=np.array(spike_list[nid])
    col=C["E"] if nid<N_E else C["I"]
    if len(st): ax.scatter(st,np.full(len(st),nid),s=0.5,c=col,alpha=0.5,linewidths=0)
ax.set_xlim(0,T_n); ax.set_ylabel("Neuron"); ax.set_xlabel("Time (ms)")
ax.set_title("Spike Raster (first 100)"); _tag(ax,"G")

# ISI distribution
ax=fig.add_subplot(gs[2,1])
if len(isi_pool)>5:
    ax.hist(isi_pool,bins=bins_isi,density=True,color="#2c7bb6",alpha=0.7,edgecolor="w")
    ax.plot(x_fit,stats.gamma.pdf(x_fit,isi_fits["gamma"]["a"],
                                   scale=isi_fits["gamma"]["scale"]),"r-",lw=2,label="Gamma fit")
ax.set_xlabel("ISI (ms)"); ax.set_ylabel("Density")
ax.set_title(f"ISI Distribution  (mean CV={cv_clean.mean():.2f})" if len(cv_clean) else "ISI")
ax.legend(frameon=False,fontsize=8); _tag(ax,"H")

# PSD
ax=fig.add_subplot(gs[2,2])
ax.semilogy(f_psd[mask_f],psd_lfp[mask_f],color=C["LFP"],lw=1.5)
for bname,(lo,hi) in band_ranges.items():
    bm=(f_psd>=lo)&(f_psd<=hi)
    if bm.any(): ax.fill_between(f_psd[bm],psd_lfp[bm],alpha=0.3,color=band_cols[bname])
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power")
ax.set_title("LFP Power Spectrum"); ax.set_xlim(0.5,150); _tag(ax,"I")

plt.tight_layout()
fig.savefig(FIGDIR/"fig11_summary_overview.png",dpi=200); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════════
figs = sorted(FIGDIR.glob("fig*.png"))
print("\n" + "━"*60)
print("✓  All simulations complete")
print(f"   Database  : {DB_PATH}")
print(f"   Figures ({len(figs)}) saved to: {FIGDIR}/")
for f in figs:
    kb=f.stat().st_size//1024
    print(f"     {f.name:<45} {kb:>5} KB")
print("━"*60)
print(f"\n  HH neuron      : {len(hh['spikes'])} spikes  |  "
      f"{len(hh['spikes'])/(T*1e-3):.1f} Hz  |  I_ext = 10 µA/cm²")
print(f"  AdEx RS        : {adex_results['RS']['rate']:.1f} Hz  |  "
      f"CV-ISI = {adex_results['RS']['cv']:.2f}")
print(f"  Network        : {state}  |  χ = {chi:.3f}  |  "
      f"E-rate = {total_e_spk/(N_E*T_n*1e-3):.1f} Hz  |  "
      f"I-rate = {total_i_spk/(N_I*T_n*1e-3):.1f} Hz")
print(f"  Dominant LFP freq : {dom_f:.1f} Hz")
print("━"*60)
