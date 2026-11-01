# -*- coding: utf-8 -*-
"""
Módulo comum das Etapas 1 a 5: preparo da base de modelagem, divisões de
validação (agrupada por navio/lote e temporal), métricas e catálogo de modelos
com espaços de busca. Sementes fixas. Todo pré-processamento vai dentro do
Pipeline, portanto é ajustado em cada dobra.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (ElasticNet, HuberRegressor, Lasso,
                                  LinearRegression, Ridge)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

RAIZ = Path(__file__).resolve().parent
SAIDA = RAIZ / "saidas"
SEMENTE = 20251101
SD_REPETIBILIDADE = 0.029      # pp, estimado na Etapa 0 com 3 pares T01/T02
TOLERANCIA_SUPER = 0.5         # pp, falha operacional: previsto > real + 0,5
CORTE_TESTE_FINAL = pd.Timestamp("2026-04-01")  # teste final: lotes a partir desta data

NUMERICAS = ["sio2", "p", "ppc", "densidade", "pfuf", "pm", "pfsa"]
BINARIAS = ["is_pfsa", "is_ppc_alto", "orig_navio", "orig_patio", "desidratante"]
FEATS = NUMERICAS + BINARIAS
FEATS_QUIMICA = ["sio2", "p", "ppc", "densidade"]           # usado no teste de proxy
FEATS_SEM_CONTEXTO = NUMERICAS                              # sem produto/origem/desidratante
ROTULOS = {
    "sio2": "SiO2 (%)", "p": "P (%)", "ppc": "PPC (%)", "densidade": "Densidade (g/cm³)",
    "pfuf": "PFUF no blend (%)", "pm": "PM no blend (%)", "pfsa": "PFSA no blend (%)",
    "is_pfsa": "Produto PF SA", "is_ppc_alto": "Produto PPC alto", "orig_navio": "Amostra de navio",
    "orig_patio": "Amostra de pátio/filtrado", "desidratante": "Com desidratante",
}

# Paleta (dataviz reference palette, modo claro)
COR = {"azul": "#2a78d6", "laranja": "#eb6834", "aqua": "#1baf7a", "amarelo": "#eda100",
       "texto": "#0b0b0b", "texto2": "#52514e", "grade": "#e3e2de", "superficie": "#fcfcfb",
       "bom": "#1a7f37", "atencao": "#b35c00", "ruim": "#c62828"}


# ---------------------------------------------------------------------------
# Base de modelagem
# ---------------------------------------------------------------------------
def carrega_base():
    """Aplica as decisões da Etapa 0 e devolve a base de modelagem (1 linha = 1 lote)."""
    b = pd.read_csv(SAIDA / "base_auditada.csv", parse_dates=["data"])
    b = b[b["excluir_motivo"].fillna("") == ""].copy()
    # consolidação das réplicas T01/T02 pela média do TML (mesma amostra física)
    for principal, replica in [(43, 44), (54, 55), (56, 57)]:
        m_p = b["linha_excel"].eq(principal)
        m_r = b["linha_excel"].eq(replica)
        b.loc[m_p, "tml"] = float(b.loc[m_p | m_r, "tml"].mean())
        b = b[~m_r]
    b = b.sort_values(["data", "linha_excel"]).reset_index(drop=True)
    # variáveis de entrada
    for c in ["pfuf", "pm", "pfsa"]:
        b[c] = b[c].fillna(0.0)          # confirmado pelo usuário: vazio = 0% de incorporação
    b["is_pfsa"] = (b["produto"] == "PF_SA").astype(int)
    b["is_ppc_alto"] = (b["produto"] == "PFN_PPC_ALTO").astype(int)
    b["orig_navio"] = (b["origem"] == "navio").astype(int)
    b["orig_patio"] = b["origem"].isin(["patio", "filtrado"]).astype(int)
    b["desidratante"] = b["desidratante"].astype(int)
    b["t_dias"] = (b["data"] - b["data"].min()).dt.days
    b["teste_final"] = (b["data"] >= CORTE_TESTE_FINAL).astype(int)
    return b


def separa(b):
    dev = b[b["teste_final"] == 0].reset_index(drop=True)
    teste = b[b["teste_final"] == 1].reset_index(drop=True)
    return dev, teste


# ---------------------------------------------------------------------------
# Divisões
# ---------------------------------------------------------------------------
def dobras_agrupadas(dev, n_splits=5, n_repeticoes=5, semente=SEMENTE):
    """GroupKFold embaralhado, repetido com sementes diferentes. Grupo = navio ou lote."""
    saida = []
    for r in range(n_repeticoes):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=semente + r)
        for k, (tr, te) in enumerate(gkf.split(dev, dev["tml"], dev["grupo"])):
            saida.append((f"rep{r}", k, tr, te))
    return saida


def dobras_temporais(dev):
    """Walk-forward com janela expansiva. Blocos de teste definidos por data."""
    cortes = [("2025-09-01", "2025-11-01"), ("2025-11-01", "2025-12-01"),
              ("2025-12-01", "2026-01-01"), ("2026-01-01", "2026-04-01")]
    saida = []
    for k, (ini, fim) in enumerate(cortes):
        ini, fim = pd.Timestamp(ini), pd.Timestamp(fim)
        tr = np.where(dev["data"] < ini)[0]
        te = np.where((dev["data"] >= ini) & (dev["data"] < fim))[0]
        if len(te) > 0:
            saida.append((f"wf{k}_{ini.date()}", k, tr, te))
    return saida


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def metricas(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = p - y
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 and np.var(y) > 0 else np.nan,
        "mape": float(np.mean(np.abs(e) / y) * 100),
        "vies": float(np.mean(e)),
        "erro_p95": float(np.percentile(np.abs(e), 95)),
        "pct_super_0p5": float(np.mean(e > TOLERANCIA_SUPER) * 100),
    }


def teste_t_corrigido(dif, n_treino, n_teste):
    """Teste t pareado com correção de Nadeau-Bengio para validação cruzada repetida.
    dif: vetor de diferenças de métrica por dobra (modelo A - modelo B)."""
    from scipy import stats
    dif = np.asarray(dif, float)
    k = len(dif)
    media = dif.mean()
    var = dif.var(ddof=1)
    if var == 0:
        return media, np.nan, 1.0
    t = media / np.sqrt(var * (1.0 / k + n_teste / n_treino))
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return media, t, p


# ---------------------------------------------------------------------------
# Estimadores auxiliares
# ---------------------------------------------------------------------------
class MedianaPorProduto(BaseEstimator, RegressorMixin):
    """Linha de base: mediana do TML por produto (PFN, PF SA, PPC alto)."""

    def __init__(self, col_pfsa=FEATS.index("is_pfsa"), col_ppc=FEATS.index("is_ppc_alto")):
        self.col_pfsa = col_pfsa
        self.col_ppc = col_ppc

    def _chave(self, X):
        X = np.asarray(X)
        return X[:, self.col_pfsa].astype(int) * 2 + X[:, self.col_ppc].astype(int)

    def fit(self, X, y):
        y = np.asarray(y, float)
        ch = self._chave(X)
        self.global_ = float(np.median(y))
        self.tabela_ = {int(k): float(np.median(y[ch == k])) for k in np.unique(ch)}
        return self

    def predict(self, X):
        ch = self._chave(X)
        return np.array([self.tabela_.get(int(k), self.global_) for k in ch])


class SelecionaColunas(BaseEstimator):
    def __init__(self, idx):
        self.idx = idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.idx]


class GAMRegressor(BaseEstimator, RegressorMixin):
    """Envoltório sklearn para pygam.LinearGAM: splines nas numéricas, termos lineares nas binárias."""

    def __init__(self, lam=1.0, n_splines=5):
        self.lam = lam
        self.n_splines = n_splines

    def fit(self, X, y):
        from pygam import LinearGAM, l, s
        X = np.asarray(X, float)
        self.binaria_ = [len(np.unique(X[:, j])) <= 2 for j in range(X.shape[1])]
        termos = None
        for j, bina in enumerate(self.binaria_):
            t = l(j) if bina else s(j, n_splines=self.n_splines, lam=self.lam)
            termos = t if termos is None else termos + t
        self.gam_ = LinearGAM(termos).fit(X, np.asarray(y, float))
        return self

    def predict(self, X):
        return self.gam_.predict(np.asarray(X, float))


class PLSWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.m_ = PLSRegression(n_components=min(self.n_components, np.asarray(X).shape[1])).fit(X, y)
        return self

    def predict(self, X):
        return self.m_.predict(X).ravel()


# ---------------------------------------------------------------------------
# Catálogo de modelos: (pipeline, espaço de busca, família)
# ---------------------------------------------------------------------------
def _pre(escala=True):
    passos = [("imput", SimpleImputer(strategy="median"))]
    if escala:
        passos.append(("escala", StandardScaler()))
    return passos


def catalogo(semente=SEMENTE):
    cat = {}
    idx_ppc = [FEATS.index("ppc")]

    # Linhas de base
    cat["media_historica"] = (Pipeline(_pre(False) + [("m", DummyRegressor(strategy="mean"))]), None, "base")
    cat["mediana_por_produto"] = (Pipeline(_pre(False) + [("m", MedianaPorProduto())]), None, "base")
    cat["linear_ppc"] = (Pipeline([("sel", SelecionaColunas(idx_ppc))] + _pre(True) + [("m", LinearRegression())]), None, "base")

    # Lineares
    cat["ols"] = (Pipeline(_pre() + [("m", LinearRegression())]), None, "linear")
    cat["ridge"] = (Pipeline(_pre() + [("m", Ridge())]), {"m__alpha": loguniform(1e-2, 1e3)}, "linear")
    cat["lasso"] = (Pipeline(_pre() + [("m", Lasso(max_iter=50000))]), {"m__alpha": loguniform(1e-3, 1)}, "linear")
    cat["elasticnet"] = (Pipeline(_pre() + [("m", ElasticNet(max_iter=50000))]),
                         {"m__alpha": loguniform(1e-3, 1), "m__l1_ratio": uniform(0.05, 0.9)}, "linear")
    cat["huber"] = (Pipeline(_pre() + [("m", HuberRegressor(max_iter=5000))]),
                    {"m__epsilon": uniform(1.1, 1.4), "m__alpha": loguniform(1e-4, 10)}, "linear")
    cat["pls"] = (Pipeline(_pre() + [("m", PLSWrapper())]), {"m__n_components": randint(1, 7)}, "linear")
    cat["pcr"] = (Pipeline(_pre() + [("pca", PCA()), ("m", LinearRegression())]), {"pca__n_components": randint(1, 9)}, "linear")

    # Não lineares
    cat["poly2_ridge"] = (Pipeline(_pre() + [("poly", PolynomialFeatures(2, include_bias=False)), ("m", Ridge())]),
                          {"m__alpha": loguniform(1e-1, 1e4)}, "nao_linear")
    cat["gam"] = (Pipeline(_pre(False) + [("m", GAMRegressor())]),
                  {"m__lam": loguniform(1e-2, 1e3), "m__n_splines": randint(4, 8)}, "nao_linear")
    cat["svr_linear"] = (Pipeline(_pre() + [("m", SVR(kernel="linear"))]),
                         {"m__C": loguniform(1e-3, 10), "m__epsilon": uniform(0.01, 0.3)}, "nao_linear")
    cat["svr_rbf"] = (Pipeline(_pre() + [("m", SVR(kernel="rbf"))]),
                      {"m__C": loguniform(1e-2, 100), "m__gamma": loguniform(1e-3, 1), "m__epsilon": uniform(0.01, 0.3)}, "nao_linear")
    cat["knn"] = (Pipeline(_pre() + [("m", KNeighborsRegressor())]),
                  {"m__n_neighbors": randint(2, 13), "m__weights": ["uniform", "distance"]}, "nao_linear")
    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(np.ones(len(FEATS)), (1e-1, 1e3)) + WhiteKernel(0.1, (1e-3, 1.0))
    cat["processo_gaussiano"] = (Pipeline(_pre() + [("m", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                                                  n_restarts_optimizer=2, random_state=semente))]), None, "nao_linear")
    cat["mlp"] = (Pipeline(_pre() + [("m", MLPRegressor(solver="lbfgs", max_iter=5000, random_state=semente))]),
                  {"m__hidden_layer_sizes": [(4,), (8,), (16,), (8, 4)], "m__alpha": loguniform(1e-3, 30)}, "nao_linear")

    # Árvores e ensembles
    cat["arvore"] = (Pipeline(_pre(False) + [("m", DecisionTreeRegressor(random_state=semente))]),
                     {"m__max_depth": randint(1, 7), "m__min_samples_leaf": randint(2, 11)}, "arvore")
    cat["random_forest"] = (Pipeline(_pre(False) + [("m", RandomForestRegressor(n_estimators=300, random_state=semente, n_jobs=1))]),
                            {"m__max_depth": [None, 2, 3, 5], "m__min_samples_leaf": randint(1, 9),
                             "m__max_features": ["sqrt", 0.5, 1.0]}, "arvore")
    cat["extra_trees"] = (Pipeline(_pre(False) + [("m", ExtraTreesRegressor(n_estimators=300, random_state=semente, n_jobs=1))]),
                          {"m__max_depth": [None, 2, 3, 5], "m__min_samples_leaf": randint(1, 9),
                           "m__max_features": ["sqrt", 0.5, 1.0]}, "arvore")
    cat["gradient_boosting"] = (Pipeline(_pre(False) + [("m", GradientBoostingRegressor(random_state=semente))]),
                                {"m__n_estimators": randint(50, 400), "m__learning_rate": loguniform(0.01, 0.3),
                                 "m__max_depth": randint(1, 4), "m__subsample": uniform(0.6, 0.4),
                                 "m__min_samples_leaf": randint(2, 9)}, "arvore")
    from xgboost import XGBRegressor
    cat["xgboost"] = (Pipeline(_pre(False) + [("m", XGBRegressor(random_state=semente, n_jobs=1, verbosity=0))]),
                      {"m__n_estimators": randint(50, 400), "m__learning_rate": loguniform(0.01, 0.3),
                       "m__max_depth": randint(1, 4), "m__subsample": uniform(0.6, 0.4),
                       "m__colsample_bytree": uniform(0.5, 0.5), "m__reg_lambda": loguniform(0.1, 30),
                       "m__min_child_weight": randint(1, 7)}, "arvore")
    from lightgbm import LGBMRegressor
    cat["lightgbm"] = (Pipeline(_pre(False) + [("m", LGBMRegressor(random_state=semente, n_jobs=1, verbose=-1, min_child_samples=3))]),
                       {"m__n_estimators": randint(50, 400), "m__learning_rate": loguniform(0.01, 0.3),
                        "m__num_leaves": randint(3, 12), "m__min_child_samples": randint(2, 9),
                        "m__subsample": uniform(0.6, 0.4), "m__subsample_freq": [1], "m__colsample_bytree": uniform(0.5, 0.5),
                        "m__reg_lambda": loguniform(0.1, 30)}, "arvore")
    from catboost import CatBoostRegressor
    cat["catboost"] = (Pipeline(_pre(False) + [("m", CatBoostRegressor(random_seed=semente, verbose=0, thread_count=1,
                                                                        allow_writing_files=False))]),
                       {"m__iterations": randint(100, 500), "m__depth": randint(2, 6), "m__learning_rate": loguniform(0.01, 0.3),
                        "m__l2_leaf_reg": loguniform(0.5, 30)}, "arvore")
    return cat


def ajusta_com_busca(pipe, espaco, X, y, grupos, semente, n_iter=15, n_inner=4, n_jobs=4):
    """Ajuste com busca aleatória de hiperparâmetros em validação interna agrupada (validação aninhada)."""
    if espaco is None:
        return clone(pipe).fit(X, y), None
    n_grp = len(np.unique(grupos))
    inner = GroupKFold(n_splits=min(n_inner, n_grp), shuffle=True, random_state=semente)
    busca = RandomizedSearchCV(clone(pipe), espaco, n_iter=n_iter, cv=list(inner.split(X, y, grupos)),
                               scoring="neg_root_mean_squared_error", random_state=semente, n_jobs=n_jobs,
                               error_score=np.nan, refit=True)
    busca.fit(X, y)
    return busca.best_estimator_, busca.best_params_


# ---------------------------------------------------------------------------
# Escolha do vencedor: empate dentro do ruído -> o mais simples e interpretável
# ---------------------------------------------------------------------------
ORDEM_SIMPLICIDADE = ["media_historica", "mediana_por_produto", "linear_ppc", "ols", "ridge", "lasso", "elasticnet",
                      "huber", "pls", "pcr", "svr_linear", "poly2_ridge", "gam", "knn", "svr_rbf", "processo_gaussiano",
                      "mlp", "arvore", "random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm",
                      "catboost", "voting", "stacking"]


def escolhe_vencedor(rk):
    """rk: ranking.csv (validação agrupada). Candidatos = empate com o melhor pela regra de 1 erro padrão
    OU teste pareado não significativo (p > 0,05). Entre eles, o mais simples."""
    cand = rk[((rk["empate_1EP_com_melhor"]) | (rk["p_vs_melhor"] > 0.05)) & (rk["p_vs_media"] < 0.05)
              & (rk["familia"] != "base")].index.tolist()
    cand.sort(key=lambda n: ORDEM_SIMPLICIDADE.index(n) if n in ORDEM_SIMPLICIDADE else 99)
    return cand[0], cand, rk.index[0]


def hiperparametros_moda(nome, esquema="grupo"):
    import json
    hp = pd.read_csv(SAIDA / "hiperparametros.csv")
    s = hp[(hp.modelo == nome) & (hp.esquema == esquema)].params
    if s.iloc[0] == "null" or pd.isna(s.iloc[0]):
        return {}
    return json.loads(s.mode().iloc[0])


def modelo_pronto(nome):
    """Pipeline do catálogo com os hiperparâmetros mais frequentes nas dobras agrupadas."""
    pipe, espaco, fam = catalogo()[nome]
    pipe = clone(pipe)
    hp = hiperparametros_moda(nome)
    if hp:
        pipe.set_params(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()})
    return pipe, espaco


def monta_ensemble(nome):
    """Reconstrói voting/stacking com os mesmos componentes e hiperparâmetros usados na competição."""
    import json
    from sklearn.ensemble import StackingRegressor, VotingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    comp = json.load(open(SAIDA / "ensembles.json"))["componentes_ensemble"]
    componentes = [(n, modelo_pronto(n)[0]) for n in comp]
    if nome == "voting":
        return VotingRegressor(componentes)
    return StackingRegressor(componentes, final_estimator=RidgeCV(alphas=np.logspace(-2, 3, 20)),
                             cv=KFold(n_splits=4, shuffle=True, random_state=SEMENTE))


def eixo_virgula(ax, x=True, y=True):
    """Decimais com vírgula nos eixos numéricos."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    if x:
        ax.xaxis.set_major_formatter(f)
    if y:
        ax.yaxis.set_major_formatter(f)
