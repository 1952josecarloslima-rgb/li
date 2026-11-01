# -*- coding: utf-8 -*-
"""
Etapa 3. Dez testes de modelo falso ou espúrio sobre o modelo escolhido.
Usa apenas o conjunto de desenvolvimento (o teste final segue intocado).
Saídas: saidas/etapa3_checks.csv, saidas/etapa3_detalhes.txt, saidas/etapa3_*.csv, figuras fig_pdp.png, fig_shap.png
"""
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import partial_dependence
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

import comum as C

warnings.filterwarnings("ignore")
det = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    det.append(s)


b = C.carrega_base()
dev, teste = C.separa(b)
X = dev[C.FEATS].to_numpy(float)
y = dev["tml"].to_numpy(float)
g = dev["grupo"].to_numpy()
rk = pd.read_csv(C.SAIDA / "ranking.csv", index_col=0)
M = pd.read_csv(C.SAIDA / "cv_metricas_dobra.csv")
P = pd.read_csv(C.SAIDA / "cv_predicoes.csv")
vencedor, candidatos, melhor_numerico = C.escolhe_vencedor(rk)
if len(sys.argv) > 1:
    vencedor = sys.argv[1]
log(f"Melhor RMSE numérico: {melhor_numerico}; candidatos em empate: {candidatos}; ESCOLHIDO (mais simples): {vencedor}")
pipe, espaco = C.modelo_pronto(vencedor)
log("Hiperparâmetros fixados (moda das dobras):", C.hiperparametros_moda(vencedor))
dobras = C.dobras_agrupadas(dev, 5, 5)
rmse_cv = rk.loc[vencedor, "rmse_mean"]
checks = []


def cv_rmse(pipe_, Xm, ym, dobras_, busca=False, espaco_=None):
    """RMSE médio fora da dobra. busca=True refaz a busca de hiperparâmetros dentro da dobra (aninhado)."""
    out = []
    for rep, k, tr, te in dobras_:
        if busca and espaco_ is not None:
            mod, _ = C.ajusta_com_busca(pipe_, espaco_, Xm[tr], ym[tr], g[tr], C.SEMENTE + k, n_iter=10)
        else:
            mod = clone(pipe_).fit(Xm[tr], ym[tr])
        out.append(C.metricas(ym[te], mod.predict(Xm[te]))["rmse"])
    return float(np.mean(out)), float(np.std(out))


# 1. Permutação do alvo -------------------------------------------------------
log("\n[1] Teste de permutação do alvo")
rng = np.random.default_rng(C.SEMENTE)
dob1 = [d for d in dobras if d[0] == "rep0"]
perm = []
for i in range(30):
    yp = rng.permutation(y)
    perm.append(cv_rmse(pipe, X, yp, dob1, busca=True, espaco_=espaco)[0])
perm = np.array(perm)
p_perm = float((np.sum(perm <= rmse_cv) + 1) / (len(perm) + 1))
sd_y = y.std(ddof=1)
log(f"RMSE real (CV agrupada) = {rmse_cv:.3f}; RMSE com alvo permutado = {perm.mean():.3f} ± {perm.std():.3f} "
    f"(mín {perm.min():.3f}); desvio padrão do alvo = {sd_y:.3f}; p = {p_perm:.3f}")
v1 = "aprovado" if p_perm < 0.05 and perm.mean() >= sd_y * 0.95 else ("atenção" if p_perm < 0.1 else "reprovado")
checks.append(("1. Permutação do alvo", f"RMSE permutado {perm.mean():.2f} (nível de acaso) contra {rmse_cv:.2f} real; p = {p_perm:.3f}", v1))
pd.Series(perm).to_csv(C.SAIDA / "etapa3_permutacao.csv", index=False)

# 2. Ganho sobre as linhas de base --------------------------------------------
log("\n[2] Comparação com as linhas de base")
piv = M[M.esquema == "grupo"].pivot_table(index=["rep", "dobra"], columns="modelo", values="rmse")
n_tr, n_te = M[M.esquema == "grupo"].n_treino.mean(), M[M.esquema == "grupo"].n_teste.mean()
res2 = []
for base in ["media_historica", "mediana_por_produto", "linear_ppc"]:
    d = piv[vencedor] - piv[base]
    media, t, p = C.teste_t_corrigido(d.values, n_tr, n_te)
    res2.append((base, piv[base].mean(), media, p))
    log(f"  vs {base:<20}: RMSE base {piv[base].mean():.3f}; ganho {-media:.3f} pp; p (Nadeau-Bengio) = {p:.4f}")
ganho_media = -res2[0][2]
v2 = "aprovado" if res2[0][3] < 0.05 and res2[2][3] < 0.10 else ("atenção" if res2[0][3] < 0.05 else "reprovado")
checks.append(("2. Ganho sobre linhas de base", f"ganho de {ganho_media:.2f} pp de RMSE sobre a média (p = {res2[0][3]:.3f}); "
               f"sobre a regressão simples em PPC {-res2[2][2]:.2f} pp (p = {res2[2][3]:.3f})", v2))

# 3. Sobreajuste e curva de aprendizado ----------------------------------------
log("\n[3] Treino x validação e curva de aprendizado")
sub = M[(M.esquema == "grupo") & (M.modelo == vencedor)]
gap = sub.rmse.mean() - sub.rmse_treino.mean()
log(f"RMSE treino = {sub.rmse_treino.mean():.3f}; RMSE validação = {sub.rmse.mean():.3f}; razão = {sub.rmse.mean() / sub.rmse_treino.mean():.2f}")
fracs = [0.3, 0.45, 0.6, 0.8, 1.0]
curva = []
for f in fracs:
    tr_s, va_s = [], []
    for rep, k, tr, te in dobras:
        grupos_tr = np.unique(g[tr])
        rng_f = np.random.default_rng(C.SEMENTE + k)
        sel = rng_f.choice(grupos_tr, size=max(6, int(round(f * len(grupos_tr)))), replace=False)
        tr_f = tr[np.isin(g[tr], sel)]
        mod = clone(pipe).fit(X[tr_f], y[tr_f])
        tr_s.append(C.metricas(y[tr_f], mod.predict(X[tr_f]))["rmse"])
        va_s.append(C.metricas(y[te], mod.predict(X[te]))["rmse"])
    curva.append({"fracao": f, "n_treino_medio": int(round(f * n_tr)), "rmse_treino": np.mean(tr_s), "rmse_val": np.mean(va_s), "rmse_val_sd": np.std(va_s)})
    log(f"  fração {f:.2f}: treino {np.mean(tr_s):.3f} | validação {np.mean(va_s):.3f} ± {np.std(va_s):.3f}")
curva = pd.DataFrame(curva)
curva.to_csv(C.SAIDA / "etapa3_curva_aprendizado.csv", index=False)
inclinacao = curva.rmse_val.iloc[-2] - curva.rmse_val.iloc[-1]
v3 = "aprovado" if sub.rmse.mean() / sub.rmse_treino.mean() < 1.5 else ("atenção" if sub.rmse.mean() / sub.rmse_treino.mean() < 2.5 else "reprovado")
checks.append(("3. Sobreajuste (treino x validação)", f"RMSE treino {sub.rmse_treino.mean():.2f} contra validação {sub.rmse.mean():.2f} (razão {sub.rmse.mean() / sub.rmse_treino.mean():.2f}); "
               f"curva de aprendizado ainda cai {inclinacao:.3f} pp entre 80% e 100% dos dados", v3))

# 4. Multicolinearidade ---------------------------------------------------------
log("\n[4] Multicolinearidade: VIF e estabilidade dos coeficientes")
Xd = pd.DataFrame(X, columns=C.FEATS)
Xc = np.column_stack([np.ones(len(Xd)), Xd.values])
vif = pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(Xd.shape[1])], index=C.FEATS).round(2)
log("VIF:\n" + vif.to_string())
corr = Xd.corr().round(2)
corr.to_csv(C.SAIDA / "etapa3_correlacao.csv")
vif.to_csv(C.SAIDA / "etapa3_vif.csv")
coefs = []
for rep, k, tr, te in dobras:
    mod = clone(pipe).fit(X[tr], y[tr])
    est = mod.named_steps["m"]
    if hasattr(est, "coef_"):
        coefs.append(np.ravel(est.coef_))
if coefs:
    coefs = pd.DataFrame(coefs, columns=C.FEATS)
    est_coef = pd.DataFrame({"media": coefs.mean(), "sd": coefs.std(), "sinal_consistente_%": (np.sign(coefs) == np.sign(coefs.mean())).mean() * 100}).round(3)
    log("Coeficientes padronizados nas 25 dobras:\n" + est_coef.to_string())
    est_coef.to_csv(C.SAIDA / "etapa3_coeficientes_dobras.csv")
    instaveis = est_coef[(est_coef["sinal_consistente_%"] < 80) & (est_coef["media"].abs() > 0.02)].index.tolist()
    v4 = "aprovado" if vif.max() < 10 and not instaveis else ("atenção" if vif.max() < 30 else "reprovado")
    txt4 = f"VIF máximo {vif.max():.1f} ({vif.idxmax()}); coeficientes com sinal instável: {instaveis if instaveis else 'nenhum relevante'}"
else:
    v4 = "aprovado" if vif.max() < 10 else "atenção"
    txt4 = f"VIF máximo {vif.max():.1f} ({vif.idxmax()}); modelo sem coeficientes lineares"
checks.append(("4. Multicolinearidade", txt4, v4))

# 5. Variáveis proxy e confundidoras -------------------------------------------
log("\n[5] Proxy e confundidores: conjuntos alternativos de variáveis (busca aninhada em cada dobra)")
conjuntos = {
    "todas": C.FEATS,
    "sem_contexto (só química + blend)": C.NUMERICAS,
    "só_química": C.FEATS_QUIMICA,
    "só_contexto (produto, origem, desidratante)": C.BINARIAS,
    "todas + tempo (t_dias)": C.FEATS + ["t_dias"],
    "só_tempo": ["t_dias"],
}
res5 = {}
for nome, cols in conjuntos.items():
    Xm = dev[cols].to_numpy(float)
    if vencedor == "linear_ppc" or nome.startswith("só_tempo"):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        pp, esp = Pipeline([("escala", StandardScaler()), ("m", Ridge())]), {"m__alpha": C.loguniform(1e-2, 1e3)}
    else:
        pp, esp = C.catalogo()[vencedor][0], espaco
        pp = clone(pp)
        if "sel" in pp.named_steps:
            pp.set_params(sel__idx=[cols.index(c) for c in ["ppc"] if c in cols] or [0])
    m, s = cv_rmse(pp, Xm, y, dobras, busca=True, espaco_=esp)
    res5[nome] = (m, s)
    log(f"  {nome:<45} RMSE = {m:.3f} ± {s:.3f}")
pd.DataFrame(res5, index=["rmse", "sd"]).T.to_csv(C.SAIDA / "etapa3_proxy.csv")
r_tudo, r_quim, r_ctx, r_tempo, r_semctx = res5["todas"][0], res5["só_química"][0], res5["só_contexto (produto, origem, desidratante)"][0], res5["só_tempo"][0], res5["sem_contexto (só química + blend)"][0]
base_media = piv["media_historica"].mean()
frac_ctx = (base_media - r_ctx) / max(base_media - r_tudo, 1e-9)
v5 = "aprovado" if (r_quim <= r_tudo + 0.03 or frac_ctx < 0.5) else ("atenção" if frac_ctx < 0.8 else "reprovado")
checks.append(("5. Proxy e confundidores", f"RMSE todas {r_tudo:.2f}; só química {r_quim:.2f}; só contexto {r_ctx:.2f}; só tempo {r_tempo:.2f}; "
               f"contexto sozinho explica {frac_ctx * 100:.0f}% do ganho sobre a média", v5))

# 6. Tendência espúria -----------------------------------------------------------
log("\n[6] Autocorrelação do alvo e dos resíduos; modelo sem tendência")
ordem = np.argsort(dev["data"].values, kind="stable")
y_ord = y[ordem]
acf_y = acf(y_ord, nlags=5, fft=False)
dw_y = durbin_watson(y_ord - y_ord.mean())
lb_y = acorr_ljungbox(y_ord - y_ord.mean(), lags=[3], return_df=True)["lb_pvalue"].iloc[0]
oof = P[(P.modelo == vencedor) & (P.esquema == "grupo")].groupby("idx").agg(y=("y", "first"), pred=("pred", "mean")).sort_index()
res_oof = (oof["pred"] - oof["y"]).values[ordem]
acf_r = acf(res_oof, nlags=5, fft=False)
dw_r = durbin_watson(res_oof)
lb_r = acorr_ljungbox(res_oof, lags=[3], return_df=True)["lb_pvalue"].iloc[0]
log(f"ACF do TML (lags 1 a 3, ordem cronológica): {np.round(acf_y[1:4], 2)}; Durbin-Watson = {dw_y:.2f}; Ljung-Box p(3) = {lb_y:.3f}")
log(f"ACF dos resíduos fora da dobra: {np.round(acf_r[1:4], 2)}; Durbin-Watson = {dw_r:.2f}; Ljung-Box p(3) = {lb_r:.3f}")
# modelo em dados detrendizados: tendência linear estimada só no treino de cada dobra
t = dev["t_dias"].to_numpy(float)
det_rmse, det_rmse_puro = [], []
for rep, k, tr, te in dobras:
    beta = np.polyfit(t[tr], y[tr], 1)
    tend_tr, tend_te = np.polyval(beta, t[tr]), np.polyval(beta, t[te])
    mod = clone(pipe).fit(X[tr], y[tr] - tend_tr)
    det_rmse.append(C.metricas(y[te], mod.predict(X[te]) + tend_te)["rmse"])      # alvo detrendizado, tendência devolvida
    det_rmse_puro.append(C.metricas(y[te] - tend_te, mod.predict(X[te]))["rmse"])  # erro na componente sem tendência
log(f"RMSE com alvo detrendizado (tendência devolvida) = {np.mean(det_rmse):.3f}; erro na componente sem tendência = {np.mean(det_rmse_puro):.3f}; original = {rmse_cv:.3f}")
v6 = "aprovado" if (abs(np.mean(det_rmse) - rmse_cv) < 0.03 and dw_r > 1.5 and dw_r < 2.5) else ("atenção" if abs(np.mean(det_rmse) - rmse_cv) < 0.08 else "reprovado")
checks.append(("6. Tendência espúria", f"ACF1 do TML {acf_y[1]:.2f}, DW {dw_y:.2f}; ACF1 resíduos {acf_r[1]:.2f}, DW {dw_r:.2f}; "
               f"RMSE detrendizado {np.mean(det_rmse):.2f} contra {rmse_cv:.2f} original", v6))

# 7. Estabilidade temporal e deriva -------------------------------------------------
log("\n[7] Estabilidade temporal (walk-forward) e deriva das variáveis")
wf = M[(M.esquema == "temporal") & (M.modelo == vencedor)][["rep", "n_treino", "n_teste", "rmse", "vies", "pct_super_0p5"]]
wf_base = M[(M.esquema == "temporal") & (M.modelo == "media_historica")][["rep", "rmse"]].rename(columns={"rmse": "rmse_media_hist"})
wf = wf.merge(wf_base, on="rep")
log(wf.round(3).to_string(index=False))
wf.to_csv(C.SAIDA / "etapa3_walkforward.csv", index=False)
drift = []
for c in C.FEATS:
    a, bb = dev[c].to_numpy(float), teste[c].to_numpy(float)
    sdp = np.sqrt((a.var(ddof=1) + bb.var(ddof=1)) / 2) if len(bb) > 1 else a.std(ddof=1)
    smd = (bb.mean() - a.mean()) / sdp if sdp > 0 else 0
    ks = stats.ks_2samp(a, bb).pvalue if len(np.unique(np.r_[a, bb])) > 1 else 1.0
    drift.append({"variavel": c, "media_dev": a.mean(), "media_teste": bb.mean(), "smd": smd, "ks_p": ks})
drift = pd.DataFrame(drift).round(3)
log("Deriva desenvolvimento -> teste final (SMD = diferença padronizada de médias):\n" + drift.to_string(index=False))
drift.to_csv(C.SAIDA / "etapa3_drift.csv", index=False)
d25, d26 = dev[dev.ano == 2025], dev[dev.ano == 2026]
log(f"TML médio no desenvolvimento: 2025 = {d25.tml.mean():.2f} (n={len(d25)}), 2026 = {d26.tml.mean():.2f} (n={len(d26)}); teste final = {teste.tml.mean():.2f}")
pior_bloco = wf.rmse.max()
v7 = "aprovado" if (wf.rmse < wf.rmse_media_hist).mean() >= 0.75 and drift.smd.abs().max() < 0.8 and wf.pct_super_0p5.max() < 25 else ("atenção" if (wf.rmse < wf.rmse_media_hist).mean() >= 0.5 else "reprovado")
checks.append(("7. Estabilidade temporal e deriva", f"walk-forward: RMSE por bloco {wf.rmse.round(2).tolist()}, viés {wf.vies.round(2).tolist()}, superestimações > 0,5 pp {wf.pct_super_0p5.round(0).tolist()}%; vence a média histórica em {int((wf.rmse < wf.rmse_media_hist).sum())} de {len(wf)} blocos; "
               f"maior deriva dev->teste: {drift.loc[drift.smd.abs().idxmax(), 'variavel']} (SMD {drift.smd.abs().max():.2f})", v7))

# 8. Plausibilidade física: PDP e SHAP ------------------------------------------------
log("\n[8] Plausibilidade física (PDP e SHAP no modelo ajustado em todo o desenvolvimento)")
mod_dev = clone(pipe).fit(X, y)
esperado = {"ppc": "+", "densidade": "-", "sio2": "?", "p": "?", "pfuf": "?", "pm": "?", "pfsa": "+",
            "is_pfsa": "+", "is_ppc_alto": "+", "orig_navio": "?", "orig_patio": "?", "desidratante": "?"}
# pfuf: a literatura do IMSBC (TWG iron ore fines) não dá sentido único para o efeito de finos/ultrafinos no TML
# (superfície específica eleva retenção de água, mas finos também reduzem a porosidade do empacotamento); registrado como "?".
MIN_LOTES_PARA_JULGAR = 3  # indicadores com menos lotes ativos no treino são "indeterminados"
justif = {"ppc": "PPC alto = goethita/hidroxilas, poroso, retém mais água antes de fluir",
          "densidade": "menor densidade = mais goethita/porosidade = TML maior; densidade alta = hematita compacta",
          "sio2": "quartzo é grosso e não poroso (reduz TML) mas aqui marca o produto PF SA (sem expectativa única)",
          "p": "sem mecanismo físico direto; acompanha goethita (fraco +)",
          "pfuf": "ultrafino: superfície específica eleva retenção de água, mas finos reduzem porosidade; sem sinal único na literatura",
          "pm": "composição não informada",
          "pfsa": "PF SA tem TML médio maior (11,4 contra 11,1)",
          "is_pfsa": "PF SA: menor densidade e maior PPC", "is_ppc_alto": "PPC alto por definição",
          "orig_navio": "sem mecanismo físico: variável de contexto/amostragem", "orig_patio": "idem",
          "desidratante": "aditivo age na umidade, efeito sobre TML não estabelecido"}
pdp_rows = []
for j, c in enumerate(C.FEATS):
    grade = np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, 9))) if c in C.NUMERICAS else np.array([0.0, 1.0])
    pd_res = partial_dependence(mod_dev, X, [j], grid_resolution=9, kind="average", custom_values={j: grade})
    xs, ys = pd_res["grid_values"][0], pd_res["average"][0]
    incl = np.polyfit(xs, ys, 1)[0] if len(xs) > 1 and np.ptp(xs) > 0 else 0.0
    efeito = float(ys[-1] - ys[0]) if len(ys) > 1 else 0.0
    sinal = "+" if efeito > 0.02 else ("-" if efeito < -0.02 else "0")
    n_ativos = int((X[:, j] != 0).sum()) if c in C.BINARIAS or c in ("pfuf", "pm", "pfsa") else len(X)
    if n_ativos < MIN_LOTES_PARA_JULGAR:
        coer = f"indeterminado (n={n_ativos})"
    else:
        coer = "coerente" if esperado[c] in ("?", sinal) or sinal == "0" else "INCOERENTE"
    pdp_rows.append({"variavel": c, "efeito_pdp_pp": round(efeito, 3), "sinal_modelo": sinal, "sinal_esperado": esperado[c], "avaliacao": coer, "justificativa": justif[c]})
    for xx, yy in zip(xs, ys):
        pdp_rows[-1].setdefault("curva", []).append((float(xx), float(yy)))
pdp_df = pd.DataFrame(pdp_rows)
log(pdp_df.drop(columns="curva").to_string(index=False))
pdp_df.to_pickle(C.SAIDA / "etapa3_pdp.pkl")
pdp_df.drop(columns="curva").to_csv(C.SAIDA / "etapa3_pdp.csv", index=False)
# SHAP
try:
    expl = shap.Explainer(mod_dev.predict, X, seed=C.SEMENTE)
    sv = expl(X, max_evals=2000 if X.shape[1] * 2 + 1 < 2000 else "auto")
    shap_vals = sv.values
except Exception as e:  # fallback
    log("SHAP Explainer genérico falhou, usando KernelExplainer:", e)
    ke = shap.KernelExplainer(mod_dev.predict, X)
    shap_vals = ke.shap_values(X, nsamples=500)
shap_imp = pd.DataFrame({"variavel": C.FEATS, "shap_medio_abs": np.abs(shap_vals).mean(0)}).sort_values("shap_medio_abs", ascending=False)
shap_imp["direcao_corr_valor"] = [np.corrcoef(X[:, j], shap_vals[:, j])[0, 1] if X[:, j].std() > 0 else 0 for j in [C.FEATS.index(v) for v in shap_imp.variavel]]
log("Importância SHAP (média do |valor|, pp de TML):\n" + shap_imp.round(3).to_string(index=False))
shap_imp.to_csv(C.SAIDA / "etapa3_shap_importancia.csv", index=False)
np.save(C.SAIDA / "etapa3_shap_valores.npy", shap_vals)
incoer = pdp_df[pdp_df.avaliacao == "INCOERENTE"].variavel.tolist()
v8 = "aprovado" if not incoer else ("atenção" if len(incoer) <= 1 else "reprovado")
checks.append(("8. Plausibilidade física", f"sinais PDP: " + ", ".join(f"{r.variavel} {r.sinal_modelo}" for r in pdp_df.itertuples() if r.sinal_modelo != "0")
               + f"; incoerentes: {incoer if incoer else 'nenhum'}", v8))

# figuras PDP e SHAP
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), facecolor=C.COR["superficie"], sharey=True)
todas_y = np.concatenate([[yy for _, yy in r.curva] for r in pdp_df.itertuples()])
for ax, c in zip(axes.ravel(), C.NUMERICAS + ["is_pfsa"]):
    ax.set_ylim(todas_y.min() - 0.08, todas_y.max() + 0.05)
    C.eixo_virgula(ax)
    r = pdp_df[pdp_df.variavel == c].iloc[0]
    xs, ys = zip(*r.curva)
    ax.plot(xs, ys, color=C.COR["azul"], lw=2, marker="o", ms=4)
    ax.set_title(C.ROTULOS[c], fontsize=10, color=C.COR["texto"])
    ax.set_facecolor(C.COR["superficie"])
    ax.grid(color=C.COR["grade"], lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=C.COR["texto2"], labelsize=8)
    if c in C.NUMERICAS:
        ax.plot(X[:, C.FEATS.index(c)], np.full(len(X), todas_y.min() - 0.06), "|", color=C.COR["texto2"], alpha=0.5)
fig.suptitle(f"Dependência parcial do TML previsto ({vencedor}), conjunto de desenvolvimento (mesma escala vertical; traços = valores observados)", fontsize=11, color=C.COR["texto"])
fig.supylabel("TML previsto (%)", color=C.COR["texto2"])
fig.tight_layout()
fig.savefig(C.SAIDA / "fig_pdp.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=C.COR["superficie"])
ordem_imp = shap_imp.sort_values("shap_medio_abs")
ax.barh([C.ROTULOS[v] for v in ordem_imp.variavel], ordem_imp.shap_medio_abs, color=C.COR["azul"], height=0.6)
for i, (v, s) in enumerate(zip(ordem_imp.variavel, ordem_imp.shap_medio_abs)):
    ax.text(s + 0.003, i, f"{s:.3f}".replace(".", ","), va="center", fontsize=8, color=C.COR["texto2"])
ax.set_xlabel("Média do |valor SHAP| (pontos percentuais de TML)", color=C.COR["texto2"])
C.eixo_virgula(ax, y=False)
ax.set_title(f"Importância das variáveis por SHAP ({vencedor})", color=C.COR["texto"])
ax.set_facecolor(C.COR["superficie"])
ax.grid(axis="x", color=C.COR["grade"], lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=C.COR["texto2"])
fig.tight_layout()
fig.savefig(C.SAIDA / "fig_shap.png", dpi=150)
plt.close(fig)

# 9. Resíduos ------------------------------------------------------------------------------
log("\n[9] Análise de resíduos fora da dobra (média das 5 repetições por lote)")
res = oof["pred"] - oof["y"]
sh_p = stats.shapiro(res).pvalue
bp_p = het_breuschpagan(res.values, np.column_stack([np.ones(len(res)), oof["pred"].values]))[1]
log(f"Shapiro-Wilk p = {sh_p:.3f} (normalidade); Breusch-Pagan p = {bp_p:.3f} (heterocedasticidade vs previsto)")
dd = dev.copy()
dd["res"] = res.values
dd["faixa_tml"] = pd.cut(dd.tml, [0, 10.9, 11.3, 20], labels=["TML < 10,9", "10,9 a 11,3", "> 11,3"])
for chave in ["faixa_tml", "produto", "origem", "ano"]:
    tab = dd.groupby(chave, observed=True)["res"].agg(n="size", vies="mean", rmse=lambda s: np.sqrt(np.mean(s ** 2))).round(3)
    log(f"Por {chave}:\n" + tab.to_string())
    tab.to_csv(C.SAIDA / f"etapa3_residuos_{chave}.csv")
faixa = dd.groupby("faixa_tml", observed=True)["res"].mean()
v9 = "atenção" if (faixa.abs().max() > 0.2 or bp_p < 0.05) else "aprovado"
if faixa.abs().max() > 0.4:
    v9 = "reprovado"
checks.append(("9. Resíduos", f"Shapiro p = {sh_p:.2f}; Breusch-Pagan p = {bp_p:.2f}; viés por faixa de TML: "
               + ", ".join(f"{k}: {v:+.2f}" for k, v in faixa.items()), v9))
dd[["linha_excel", "data", "material", "produto", "origem", "tml", "res"]].assign(pred=oof["pred"].values).to_csv(C.SAIDA / "etapa3_residuos_lote.csv", index=False)

# 10. Domínio de aplicação ---------------------------------------------------------------------
log("\n[10] Domínio de aplicação")
dom = pd.DataFrame({"min": dev[C.NUMERICAS].min(), "p05": dev[C.NUMERICAS].quantile(0.05), "p95": dev[C.NUMERICAS].quantile(0.95),
                    "max": dev[C.NUMERICAS].max()}).round(3)
log(dom.to_string())
dom.to_csv(C.SAIDA / "etapa3_dominio.csv")
fora = []
for c in C.NUMERICAS:
    m = (teste[c] < dev[c].min()) | (teste[c] > dev[c].max())
    if m.any():
        fora.append(f"{c}: {teste.loc[m, 'material'].str.slice(0, 30).tolist()}")
log("Lotes do teste final fora da faixa de treino:", fora if fora else "nenhum")
cobertura_prod = dev.produto.value_counts().to_dict()
v10 = "atenção"  # sempre: base pequena, PF SA e PPC alto sub-representados, mineralogia ausente
checks.append(("10. Domínio de aplicação", f"PFN com SiO2 {dom.loc['sio2', 'min']:.2f} a {dom.loc['sio2', 'max']:.2f}, PPC {dom.loc['ppc', 'min']:.2f} a {dom.loc['ppc', 'max']:.2f}, "
               f"densidade {dom.loc['densidade', 'min']:.3f} a {dom.loc['densidade', 'max']:.3f}; produtos no treino {cobertura_prod}; "
               f"teste final fora da faixa: {len(fora)} variável(is)", v10))

ck = pd.DataFrame(checks, columns=["teste", "resultado", "veredito"])
ck.to_csv(C.SAIDA / "etapa3_checks.csv", index=False)
(C.SAIDA / "etapa3_detalhes.txt").write_text("\n".join(det), encoding="utf-8")
print("\n" + ck.to_string(index=False))
