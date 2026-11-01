# -*- coding: utf-8 -*-
"""
Etapa 4. Intervalos de previsão para o modelo escolhido.
(a) Conformal por validação cruzada (cross-conformal, resíduos fora da dobra no treino de cada dobra externa),
    cobertura empírica medida nas dobras externas agrupadas. Intervalo simétrico e limite inferior unilateral.
(b) Regressão quantílica linear (q05/q95) como comparação.
(c) Margem de segurança operacional baseada no limite inferior.
Saídas: saidas/etapa4_cobertura.csv, saidas/etapa4_margem.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import comum as C

b = C.carrega_base()
dev, teste = C.separa(b)
X, y, g = dev[C.FEATS].to_numpy(float), dev["tml"].to_numpy(float), dev["grupo"].to_numpy()
rk = pd.read_csv(C.SAIDA / "ranking.csv", index_col=0)
vencedor = sys.argv[1] if len(sys.argv) > 1 else C.escolhe_vencedor(rk)[0]
pipe, _ = C.modelo_pronto(vencedor)
dobras = C.dobras_agrupadas(dev, 5, 5)
ALFAS = [0.10, 0.05]


def quantil_conformal(v, alfa):
    """Quantil corrigido para amostra finita: ceil((n+1)(1-alfa))/n."""
    n = len(v)
    k = int(np.ceil((n + 1) * (1 - alfa)))
    k = min(max(k, 1), n)
    return float(np.sort(v)[k - 1])


linhas, resid_cal_total, e_oof = [], [], np.full(len(y), np.nan)
for rep, k, tr, te in dobras:
    # resíduos de calibração: fora da dobra interna, só com dados de treino
    inner = GroupKFold(n_splits=5, shuffle=True, random_state=C.SEMENTE + k)
    cal = np.zeros(len(tr))
    for itr, ite in inner.split(X[tr], y[tr], g[tr]):
        m = clone(pipe).fit(X[tr][itr], y[tr][itr])
        cal[ite] = m.predict(X[tr][ite]) - y[tr][ite]        # e = previsto - real (positivo = superestimou)
    mod = clone(pipe).fit(X[tr], y[tr])
    p = mod.predict(X[te])
    e = p - y[te]
    if rep == "rep0":
        e_oof[te] = e
    resid_cal_total.append(cal)
    for alfa in ALFAS:
        q_sim = quantil_conformal(np.abs(cal), alfa)       # intervalo simétrico
        q_inf = quantil_conformal(cal, alfa)               # limite inferior unilateral: y >= p - q_inf com prob 1-alfa
        linhas.append({"rep": rep, "dobra": k, "alfa": alfa, "metodo": "conformal_simetrico",
                       "cobertura": float(np.mean(np.abs(e) <= q_sim)), "largura": 2 * q_sim, "margem": q_sim})
        linhas.append({"rep": rep, "dobra": k, "alfa": alfa, "metodo": "conformal_limite_inferior",
                       "cobertura": float(np.mean(y[te] >= p - q_inf)), "largura": np.nan, "margem": q_inf})
    # regressão quantílica linear como comparação (q_alfa/2 e q_1-alfa/2; limite inferior q_alfa)
    for alfa in ALFAS:
        qlo = Pipeline([("esc", StandardScaler()), ("m", QuantileRegressor(quantile=alfa / 2, alpha=0.05, solver="highs"))]).fit(X[tr], y[tr])
        qhi = Pipeline([("esc", StandardScaler()), ("m", QuantileRegressor(quantile=1 - alfa / 2, alpha=0.05, solver="highs"))]).fit(X[tr], y[tr])
        qinf = Pipeline([("esc", StandardScaler()), ("m", QuantileRegressor(quantile=alfa, alpha=0.05, solver="highs"))]).fit(X[tr], y[tr])
        lo, hi, inf = qlo.predict(X[te]), qhi.predict(X[te]), qinf.predict(X[te])
        linhas.append({"rep": rep, "dobra": k, "alfa": alfa, "metodo": "quantilica_simetrica",
                       "cobertura": float(np.mean((y[te] >= lo) & (y[te] <= hi))), "largura": float(np.mean(hi - lo)), "margem": np.nan})
        linhas.append({"rep": rep, "dobra": k, "alfa": alfa, "metodo": "quantilica_limite_inferior",
                       "cobertura": float(np.mean(y[te] >= inf)), "largura": np.nan, "margem": float(np.mean(p - inf))})

L = pd.DataFrame(linhas)
cob = L.groupby(["metodo", "alfa"]).agg(cobertura_media=("cobertura", "mean"), cobertura_sd=("cobertura", "std"),
                                         largura_media=("largura", "mean"), margem_media=("margem", "mean")).round(3)
cob["cobertura_nominal"] = 1 - cob.index.get_level_values("alfa")
print(f"Modelo: {vencedor}\n")
print(cob.to_string())
cob.to_csv(C.SAIDA / "etapa4_cobertura.csv")

# margem operacional: calibrada com resíduos fora da dobra de TODO o desenvolvimento (GroupKFold 5, rep0)
e_cal = e_oof[~np.isnan(e_oof)]
margens = {f"margem_{int((1 - a) * 100)}": quantil_conformal(e_cal, a) for a in [0.10, 0.05, 0.02]}
margens.update({"modelo": vencedor, "n_calibracao": int(len(e_cal)), "erro_medio_oof": float(e_cal.mean()),
                "sd_erro_oof": float(e_cal.std(ddof=1)), "sd_repetibilidade_ensaio": C.SD_REPETIBILIDADE,
                "sd_alvo_dev": float(y.std(ddof=1))})
print("\nMargens unilaterais (previsto - margem = limite inferior do TML):")
for k_, v in margens.items():
    print(f"  {k_}: {v}")
json.dump(margens, open(C.SAIDA / "etapa4_margem.json", "w"), indent=2)
np.save(C.SAIDA / "etapa4_erros_oof.npy", e_oof)
