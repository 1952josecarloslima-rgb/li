# -*- coding: utf-8 -*-
"""
Etapa 5. Avaliação única no teste final (lotes >= 2026-04-01), ajuste do modelo de produção em todos
os 60 lotes, pipeline salvo, equação simplificada e figuras.
Saídas: saidas/etapa5_teste_final.csv, saidas/modelo_tml_final.joblib, saidas/modelo_tml_final.json,
        saidas/fig_previsto_real.png, fig_residuos.png, fig_ranking.png, saidas/equacao.txt
"""
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

import comum as C

b = C.carrega_base()
dev, teste = C.separa(b)
X, y, g = dev[C.FEATS].to_numpy(float), dev["tml"].to_numpy(float), dev["grupo"].to_numpy()
Xt, yt = teste[C.FEATS].to_numpy(float), teste["tml"].to_numpy(float)
rk = pd.read_csv(C.SAIDA / "ranking.csv", index_col=0)
vencedor, candidatos, melhor_num = C.escolhe_vencedor(rk)
if len(sys.argv) > 1:
    vencedor = sys.argv[1]
margens = json.load(open(C.SAIDA / "etapa4_margem.json"))
cat = C.catalogo()


def ajusta_em_dev(nome):
    if nome in ("voting", "stacking"):
        return C.monta_ensemble(nome).fit(X, y), {}
    pipe, espaco, _ = cat[nome]
    if espaco is None:
        return clone(pipe).fit(X, y), {}
    cv = list(GroupKFold(5, shuffle=True, random_state=C.SEMENTE).split(X, y, g))
    bus = RandomizedSearchCV(clone(pipe), espaco, n_iter=30, cv=cv, scoring="neg_root_mean_squared_error",
                             random_state=C.SEMENTE, n_jobs=4).fit(X, y)
    return bus.best_estimator_, bus.best_params_


def bootstrap_rmse(yv, pv, B=2000):
    rng = np.random.default_rng(C.SEMENTE)
    n = len(yv)
    vals = [C.metricas(yv[i], pv[i])["rmse"] for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(vals, [2.5, 97.5])


# --- avaliação única no teste final: vencedor, linhas de base e melhor numérico (para contexto) ---
linhas = []
preds_teste = {}
for nome in dict.fromkeys([vencedor, melhor_num, "media_historica", "mediana_por_produto", "linear_ppc"]):
    mod, hp = ajusta_em_dev(nome)
    p = mod.predict(Xt)
    preds_teste[nome] = p
    m = C.metricas(yt, p)
    lo, hi = bootstrap_rmse(yt, p)
    m.update({"modelo": nome, "rmse_ic95_inf": lo, "rmse_ic95_sup": hi, "n_teste": len(yt), "hiperparametros": json.dumps(hp, default=str)})
    if nome == vencedor:
        mod_vencedor_dev, hp_vencedor = mod, hp
        lb = p - margens["margem_95"]
        m["cobertura_limite_inferior_95"] = float(np.mean(yt >= lb))
        m["pct_super_0p5_com_margem"] = float(np.mean(lb - yt > C.TOLERANCIA_SUPER) * 100)
    linhas.append(m)
TF = pd.DataFrame(linhas).set_index("modelo")
pd.set_option("display.width", 250)
print(f"Vencedor escolhido antes do teste: {vencedor} (melhor numérico: {melhor_num}; candidatos: {candidatos})")
print("\nTESTE FINAL (n = %d lotes, 2026-04 a 2026-10), avaliação única:" % len(yt))
print(TF[["rmse", "rmse_ic95_inf", "rmse_ic95_sup", "mae", "r2", "vies", "erro_p95", "pct_super_0p5"]].round(3).to_string())
print("Cobertura do limite inferior 95% no teste final (vencedor):", TF.loc[vencedor, "cobertura_limite_inferior_95"])
TF.to_csv(C.SAIDA / "etapa5_teste_final.csv")
det = teste[["linha_excel", "data", "material", "produto", "origem", "tml"]].copy()
det["previsto"] = preds_teste[vencedor]
det["limite_inferior_95"] = det["previsto"] - margens["margem_95"]
det["erro"] = det["previsto"] - det["tml"]
print("\nLote a lote:\n" + det.round(2).to_string(index=False))
det.to_csv(C.SAIDA / "etapa5_teste_final_lotes.csv", index=False)

# --- modelo de produção: ajustado nos 60 lotes com os hiperparâmetros escolhidos em dev ---
pipe_final = clone(cat[vencedor][0])
if hp_vencedor:
    pipe_final.set_params(**hp_vencedor)
Xa, ya = b[C.FEATS].to_numpy(float), b["tml"].to_numpy(float)
pipe_final.fit(Xa, ya)
joblib.dump(pipe_final, C.SAIDA / "modelo_tml_final.joblib")
meta = {"modelo": vencedor, "hiperparametros": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in hp_vencedor.items()},
        "variaveis_entrada_ordem": C.FEATS, "n_lotes_treino": int(len(b)), "periodo": [str(b.data.min().date()), str(b.data.max().date())],
        "margem_seguranca_pp": {"90": margens["margem_90"], "95": margens["margem_95"], "98": margens["margem_98"]},
        "dominio": {c: [float(b[c].min()), float(b[c].max())] for c in C.NUMERICAS},
        "rmse_cv_agrupada": float(rk.loc[vencedor, "rmse_mean"]), "rmse_teste_final": float(TF.loc[vencedor, "rmse"]),
        "uso": "TML_estimado = pipeline.predict(X); TML_seguro = TML_estimado - margem_95"}
json.dump(meta, open(C.SAIDA / "modelo_tml_final.json", "w"), indent=2, ensure_ascii=False)

# --- equação simplificada (se o modelo for linear) ---
eq_txt = ""
est = pipe_final.named_steps.get("m")
if hasattr(est, "coef_") and "escala" in pipe_final.named_steps and "poly" not in pipe_final.named_steps and "pca" not in pipe_final.named_steps:
    esc = pipe_final.named_steps["escala"]
    coef_std = np.ravel(est.coef_)
    idx = pipe_final.named_steps["sel"].idx if "sel" in pipe_final.named_steps else list(range(len(C.FEATS)))
    nomes = [C.FEATS[i] for i in idx]
    coef_orig = coef_std / esc.scale_
    b0 = float(est.intercept_ - np.sum(coef_orig * esc.mean_))
    termos = [f"{b0:+.4f}"] + [f"{c:+.4f} x {C.ROTULOS[n]}" for c, n in zip(coef_orig, nomes)]
    eq_txt = "TML (%) = " + " ".join(termos)
    eq_txt = eq_txt.replace(".", ",")
    eq_txt += f"\nTML seguro (limite inferior 95%) = TML - {margens['margem_95']:.2f}".replace(".", ",")
    eq_txt += "\n\nCoeficientes padronizados (efeito de +1 desvio padrão da variável, em pp de TML):\n"
    eq_txt += "\n".join(f"  {C.ROTULOS[n]:<28} {c:+.3f}".replace(".", ",") for n, c in sorted(zip(nomes, coef_std), key=lambda t: -abs(t[1])))
    print("\n" + eq_txt)
    (C.SAIDA / "equacao.txt").write_text(eq_txt, encoding="utf-8")

# --- figuras ---
P = pd.read_csv(C.SAIDA / "cv_predicoes.csv")
oof = P[(P.modelo == vencedor) & (P.esquema == "grupo")].groupby("idx").agg(y=("y", "first"), pred=("pred", "mean"), sd=("pred", "std"))
est_cor = {"CV agrupada (desenvolvimento)": C.COR["azul"], "Teste final": C.COR["laranja"]}


def estilo(ax, virgula_x=True, virgula_y=True):
    C.eixo_virgula(ax, x=virgula_x, y=virgula_y)
    ax.set_facecolor(C.COR["superficie"])
    ax.grid(color=C.COR["grade"], lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=C.COR["texto2"])


fig, ax = plt.subplots(figsize=(6.8, 6.4), facecolor=C.COR["superficie"])
lim = [10.2, 12.3]
ax.plot(lim, lim, color=C.COR["texto2"], lw=1, ls="--", label="previsto = real")
ax.fill_between(lim, lim, [l + C.TOLERANCIA_SUPER for l in lim], color=C.COR["grade"], alpha=0.7, lw=0, label="superestimação tolerada (até 0,5 pp)")
ax.text(11.45, 10.42, "acima da diagonal: modelo superestima o TML\n(lado de maior risco operacional)", fontsize=8, color=C.COR["texto2"])
ax.scatter(oof.y, oof.pred, s=42, color=C.COR["azul"], edgecolor=C.COR["superficie"], lw=1, label="CV agrupada (desenvolvimento)")
ax.scatter(yt, preds_teste[vencedor], s=60, marker="D", color=C.COR["laranja"], edgecolor=C.COR["superficie"], lw=1, label="Teste final (2026-04 a 2026-10)")
ax.set_xlabel("TML real (%)", color=C.COR["texto2"])
ax.set_ylabel("TML previsto (%)", color=C.COR["texto2"])
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_title(f"Previsto x real, modelo {vencedor}", color=C.COR["texto"])
ax.legend(frameon=False, fontsize=9, loc="upper left")
estilo(ax)
fig.tight_layout()
fig.savefig(C.SAIDA / "fig_previsto_real.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=C.COR["superficie"])
res = oof.pred - oof.y
axes[0].axhline(0, color=C.COR["texto2"], lw=1)
axes[0].axhline(C.TOLERANCIA_SUPER, color=C.COR["ruim"], lw=1, ls=":")
axes[0].scatter(oof.pred, res, s=36, color=C.COR["azul"], edgecolor=C.COR["superficie"])
axes[0].scatter(preds_teste[vencedor], preds_teste[vencedor] - yt, s=50, marker="D", color=C.COR["laranja"], edgecolor=C.COR["superficie"])
axes[0].set_xlabel("TML previsto (%)", color=C.COR["texto2"])
axes[0].set_ylabel("Erro = previsto - real (pp)", color=C.COR["texto2"])
axes[0].set_title("Resíduo x previsto", color=C.COR["texto"])
datas = pd.concat([dev.data, teste.data])
erros = np.r_[res.values, preds_teste[vencedor] - yt]
cores = [C.COR["azul"]] * len(res) + [C.COR["laranja"]] * len(yt)
axes[1].axhline(0, color=C.COR["texto2"], lw=1)
axes[1].axhline(C.TOLERANCIA_SUPER, color=C.COR["ruim"], lw=1, ls=":")
axes[1].scatter(datas, erros, s=36, c=cores, edgecolor=C.COR["superficie"])
axes[1].axvline(C.CORTE_TESTE_FINAL, color=C.COR["texto2"], lw=1, ls="--")
axes[1].set_title("Resíduo ao longo do tempo (tracejado: início do teste final)", color=C.COR["texto"], fontsize=10)
axes[1].tick_params(axis="x", rotation=30)
from scipy import stats
stats.probplot(res, dist="norm", plot=axes[2])
axes[2].get_lines()[0].set(color=C.COR["azul"], marker="o", ms=5, mec=C.COR["superficie"])
axes[2].get_lines()[1].set(color=C.COR["texto2"], lw=1)
axes[2].set_title("Q-Q normal dos resíduos (CV)", color=C.COR["texto"])
axes[2].set_xlabel("Quantis teóricos", color=C.COR["texto2"])
axes[2].set_ylabel("Resíduo ordenado (pp)", color=C.COR["texto2"])
for i, ax in enumerate(axes):
    estilo(ax, virgula_x=(i != 1))
fig.tight_layout()
fig.savefig(C.SAIDA / "fig_residuos.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 8), facecolor=C.COR["superficie"])
r = rk.sort_values("rmse_mean", ascending=False)
fam_cor = {"base": C.COR["texto2"], "linear": C.COR["azul"], "nao_linear": C.COR["aqua"], "arvore": C.COR["amarelo"], "ensemble": C.COR["laranja"]}
cores = [fam_cor[f] for f in r.familia]
ax.barh(r.index, r.rmse_mean, xerr=r.rmse_std, color=cores, height=0.62, error_kw={"ecolor": C.COR["texto2"], "lw": 1, "capsize": 2})
ax.axvline(rk.loc["media_historica", "rmse_mean"], color=C.COR["texto2"], ls="--", lw=1)
ax.axvline(C.SD_REPETIBILIDADE, color=C.COR["ruim"], ls=":", lw=1)
ax.text(rk.loc["media_historica", "rmse_mean"] + 0.005, len(r) - 1.3, "média histórica", color=C.COR["texto2"], fontsize=8)
ax.text(C.SD_REPETIBILIDADE + 0.005, len(r) - 0.6, "repetibilidade do ensaio", color=C.COR["ruim"], fontsize=8, rotation=90, va="top")
for i, (n, v) in enumerate(zip(r.index, r.rmse_mean)):
    ax.text(v + r.loc[n, "rmse_std"] + 0.01, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=8, color=C.COR["texto2"])
    if n == vencedor:
        ax.get_yticklabels()[i].set_fontweight("bold")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=l) for l, c in zip(["linha de base", "linear", "não linear", "árvores/boosting", "ensemble combinado"], fam_cor.values())],
          frameon=False, fontsize=9, loc="upper right")
ax.set_xlabel("RMSE fora da dobra (pp de TML), média ± desvio padrão em 25 dobras agrupadas por navio/lote", color=C.COR["texto2"], fontsize=9)
ax.set_title("Ranking dos modelos, validação cruzada agrupada e aninhada", color=C.COR["texto"])
ax.set_xlim(0, max(r.rmse_mean + r.rmse_std) + 0.1)
estilo(ax, virgula_y=False)
fig.tight_layout()
fig.savefig(C.SAIDA / "fig_ranking.png", dpi=150)
plt.close(fig)
print("\nFiguras e modelo salvos em", C.SAIDA)
