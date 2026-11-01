# -*- coding: utf-8 -*-
"""
Etapas 1 e 2. Validação anti-vazamento e competição de modelos.
- Teste final (lotes >= 2026-04-01) separado e não tocado aqui.
- Validação cruzada agrupada por navio/lote (GroupKFold embaralhado, 5 dobras x 5 sementes).
- Validação temporal walk-forward (4 blocos).
- Hiperparâmetros por busca aleatória DENTRO de cada dobra (validação aninhada).
Saídas: saidas/cv_predicoes.csv (previsões fora da dobra), saidas/cv_metricas_dobra.csv,
        saidas/ranking.csv, saidas/ranking_temporal.csv, saidas/hiperparametros.csv
"""
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

import comum as C

N_REP = 5
N_ITER = 15
apenas = sys.argv[1:]  # opcional: nomes de modelos para rodar

b = C.carrega_base()
dev, teste = C.separa(b)
print(f"Base de modelagem: {len(b)} lotes | desenvolvimento: {len(dev)} | teste final (intocado): {len(teste)}")
print(f"Grupos no desenvolvimento: {dev['grupo'].nunique()}")
X = dev[C.FEATS].to_numpy(float)
y = dev["tml"].to_numpy(float)
g = dev["grupo"].to_numpy()

dobras_g = C.dobras_agrupadas(dev, n_splits=5, n_repeticoes=N_REP)
dobras_t = C.dobras_temporais(dev)
print("Walk-forward:", [(n, len(tr), len(te)) for n, _, tr, te in dobras_t])

cat = C.catalogo()


def roda(nome, pipe, espaco, dobras, esquema):
    preds, mets, hp = [], [], []
    for rep, k, tr, te in dobras:
        sem = C.SEMENTE + (int(rep[3:]) if rep.startswith("rep") else 50 + k) * 100 + k
        t0 = time.time()
        mod, params = C.ajusta_com_busca(pipe, espaco, X[tr], y[tr], g[tr], sem, n_iter=N_ITER)
        p = mod.predict(X[te])
        p_tr = mod.predict(X[tr])
        m = C.metricas(y[te], p)
        m.update({"modelo": nome, "esquema": esquema, "rep": rep, "dobra": k, "n_treino": len(tr), "n_teste": len(te),
                  "rmse_treino": C.metricas(y[tr], p_tr)["rmse"], "segundos": time.time() - t0})
        mets.append(m)
        hp.append({"modelo": nome, "esquema": esquema, "rep": rep, "dobra": k, "params": json.dumps(params, default=str)})
        for i, pi in zip(te, p):
            preds.append({"modelo": nome, "esquema": esquema, "rep": rep, "dobra": k, "idx": int(i), "y": y[i], "pred": float(pi)})
    return preds, mets, hp


todos_p, todos_m, todos_hp = [], [], []
nomes = [n for n in cat if not apenas or n in apenas]
for nome in nomes:
    pipe, espaco, fam = cat[nome]
    t0 = time.time()
    p1, m1, h1 = roda(nome, pipe, espaco, dobras_g, "grupo")
    p2, m2, h2 = roda(nome, pipe, espaco, dobras_t, "temporal")
    todos_p += p1 + p2
    todos_m += m1 + m2
    todos_hp += h1 + h2
    mg = pd.DataFrame(m1)
    print(f"{nome:<22} RMSE grupo = {mg.rmse.mean():.3f} ± {mg.rmse.std():.3f} | temporal = {pd.DataFrame(m2).rmse.mean():.3f} | {time.time() - t0:5.0f}s", flush=True)

# ---------------------------------------------------------------------------
# Ensembles combinados: melhores de famílias diferentes (escolhidos pela CV agrupada)
# ---------------------------------------------------------------------------
M = pd.DataFrame(todos_m)
res_g = M[M.esquema == "grupo"].groupby("modelo").rmse.mean()
fam = {n: cat[n][2] for n in cat}
melhores = {}
for n, v in res_g.sort_values().items():
    f = fam[n]
    if f != "base" and f not in melhores:
        melhores[f] = n
print("Melhores por família:", melhores)
hp_df = pd.DataFrame(todos_hp)


def hp_moda(nome):
    """Hiperparâmetros mais frequentes nas dobras agrupadas (para compor os ensembles)."""
    s = hp_df[(hp_df.modelo == nome) & (hp_df.esquema == "grupo")].params
    if s.iloc[0] == "null":
        return {}
    return json.loads(s.mode().iloc[0])


componentes = []
for f, n in melhores.items():
    pipe = clone(cat[n][0])
    pipe.set_params(**hp_moda(n))
    componentes.append((n, pipe))
voting = VotingRegressor(componentes)
stacking = StackingRegressor(componentes, final_estimator=RidgeCV(alphas=np.logspace(-2, 3, 20)),
                             cv=KFold(n_splits=4, shuffle=True, random_state=C.SEMENTE), passthrough=False)
# Obs.: a CV interna do stacking usa KFold simples porque o StackingRegressor não recebe grupos;
# 42 dos 45 grupos do desenvolvimento têm 1 amostra, então o efeito é desprezível e está documentado.
for nome, mod in [("voting", voting), ("stacking", stacking)]:
    t0 = time.time()
    p1, m1, h1 = roda(nome, mod, None, dobras_g, "grupo")
    p2, m2, h2 = roda(nome, mod, None, dobras_t, "temporal")
    todos_p += p1 + p2
    todos_m += m1 + m2
    todos_hp += h1 + h2
    print(f"{nome:<22} RMSE grupo = {pd.DataFrame(m1).rmse.mean():.3f} ± {pd.DataFrame(m1).rmse.std():.3f} | temporal = {pd.DataFrame(m2).rmse.mean():.3f} | {time.time() - t0:5.0f}s", flush=True)
    fam[nome] = "ensemble"

# ---------------------------------------------------------------------------
# Ranking com intervalo e testes pareados
# ---------------------------------------------------------------------------
P = pd.DataFrame(todos_p)
M = pd.DataFrame(todos_m)
M["familia"] = M.modelo.map(fam)
P.to_csv(C.SAIDA / "cv_predicoes.csv", index=False)
M.to_csv(C.SAIDA / "cv_metricas_dobra.csv", index=False)
pd.DataFrame(todos_hp).to_csv(C.SAIDA / "hiperparametros.csv", index=False)
json.dump({"melhores_por_familia": melhores, "componentes_ensemble": [n for _, n in melhores.items()]},
          open(C.SAIDA / "ensembles.json", "w"), indent=2)

cols = ["rmse", "mae", "r2", "mape", "vies", "erro_p95", "pct_super_0p5", "rmse_treino"]


def ranking(esquema):
    sub = M[M.esquema == esquema]
    agg = sub.groupby("modelo")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["familia"] = agg.index.map(fam)
    agg["rmse_p05"] = sub.groupby("modelo").rmse.quantile(0.05)
    agg["rmse_p95"] = sub.groupby("modelo").rmse.quantile(0.95)
    agg["n_dobras"] = sub.groupby("modelo").size()
    agg = agg.sort_values("rmse_mean")
    # teste pareado vs melhor e vs média histórica (Nadeau-Bengio)
    piv = sub.pivot_table(index=["rep", "dobra"], columns="modelo", values="rmse")
    melhor = agg.index[0]
    n_tr, n_te = sub.n_treino.mean(), sub.n_teste.mean()
    for alvo, rot in [(melhor, "vs_melhor"), ("media_historica", "vs_media")]:
        ps, difs = [], []
        for m in agg.index:
            d = piv[m] - piv[alvo]
            media, t, p = C.teste_t_corrigido(d.values, n_tr, n_te)
            ps.append(p)
            difs.append(media)
        agg[f"dif_rmse_{rot}"] = difs
        agg[f"p_{rot}"] = ps
    # regra do erro padrão: empate com o melhor se rmse_mean <= rmse_melhor + EP_melhor
    ep = agg.loc[melhor, "rmse_std"] / np.sqrt(agg.loc[melhor, "n_dobras"])
    agg["empate_1EP_com_melhor"] = agg.rmse_mean <= agg.loc[melhor, "rmse_mean"] + ep
    return agg


rk_g = ranking("grupo")
rk_t = ranking("temporal")
rk_g.to_csv(C.SAIDA / "ranking.csv")
rk_t.to_csv(C.SAIDA / "ranking_temporal.csv")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("\nRANKING, validação agrupada por navio/lote (5 dobras x 5 sementes):")
print(rk_g[["familia", "rmse_mean", "rmse_std", "rmse_p05", "rmse_p95", "mae_mean", "r2_mean", "vies_mean", "erro_p95_mean",
            "pct_super_0p5_mean", "rmse_treino_mean", "p_vs_melhor", "p_vs_media", "empate_1EP_com_melhor"]].round(3).to_string())
print("\nRANKING, validação temporal walk-forward (4 blocos):")
print(rk_t[["familia", "rmse_mean", "rmse_std", "mae_mean", "vies_mean", "pct_super_0p5_mean", "p_vs_media"]].round(3).to_string())
