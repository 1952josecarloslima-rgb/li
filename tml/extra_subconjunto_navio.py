# -*- coding: utf-8 -*-
"""Análise complementar: o que a química sozinha rende DENTRO do subconjunto de amostras de navio
(população-alvo da decisão de embarque), sem variáveis de contexto. Validação agrupada repetida."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import comum as C

b = C.carrega_base()
dev, teste = C.separa(b)
nav = dev[dev.orig_navio == 1].reset_index(drop=True)
print(f"Amostras de navio no desenvolvimento: {len(nav)} em {nav.grupo.nunique()} navios; TML médio {nav.tml.mean():.2f}, sd {nav.tml.std():.2f}")
y, g = nav.tml.to_numpy(), nav.grupo.to_numpy()
modelos = {
    "média (só navios)": (DummyRegressor(), C.FEATS_QUIMICA),
    "linear PPC": (Pipeline([("e", StandardScaler()), ("m", LinearRegression())]), ["ppc"]),
    "ridge química (SiO2, P, PPC, dens.)": (Pipeline([("e", StandardScaler()), ("m", Ridge(alpha=10))]), C.FEATS_QUIMICA),
    "ridge química + blend": (Pipeline([("e", StandardScaler()), ("m", Ridge(alpha=10))]), C.NUMERICAS),
    "ridge completo (modelo escolhido)": (Pipeline([("e", StandardScaler()), ("m", Ridge(alpha=10))]), C.FEATS),
}
linhas = []
for nome, (pipe, cols) in modelos.items():
    X = nav[cols].to_numpy(float)
    rm, sup = [], []
    for r in range(10):
        for tr, te in GroupKFold(5, shuffle=True, random_state=C.SEMENTE + r).split(X, y, g):
            p = clone(pipe).fit(X[tr], y[tr]).predict(X[te])
            m = C.metricas(y[te], p)
            rm.append(m["rmse"]); sup.append(m["pct_super_0p5"])
    linhas.append({"modelo": nome, "rmse": np.mean(rm), "sd": np.std(rm), "pct_super_0p5": np.mean(sup)})
T = pd.DataFrame(linhas).round(3)
print(T.to_string(index=False))
T.to_csv(C.SAIDA / "extra_subconjunto_navio.csv", index=False)
# o mesmo, com o modelo escolhido treinado em TODO o desenvolvimento e avaliado só nos lotes de navio fora da dobra
P = pd.read_csv(C.SAIDA / "cv_predicoes.csv")
oof = P[(P.modelo == "ridge") & (P.esquema == "grupo")].groupby("idx").agg(y=("y", "first"), pred=("pred", "mean"))
idx_nav = dev.index[dev.orig_navio == 1]
m = C.metricas(oof.loc[idx_nav, "y"], oof.loc[idx_nav, "pred"])
print(f"\nRidge completo treinado em tudo, avaliado fora da dobra só nos lotes de navio: RMSE {m['rmse']:.3f}, viés {m['vies']:+.3f}, super>0,5: {m['pct_super_0p5']:.1f}%")
