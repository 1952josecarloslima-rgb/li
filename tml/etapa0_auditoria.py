# -*- coding: utf-8 -*-
"""
Etapa 0. Auditoria da base de TML (Laboratório de Controle de Embarque).
Lê a planilha original, padroniza colunas, extrai metadados do campo
"Material" (produto, origem da amostra, navio, cliente, desidratante,
réplica), audita faltantes/duplicidades/outliers/unidades e grava:
  - saidas/base_bruta.csv          : leitura fiel da planilha (ambas as abas)
  - saidas/base_auditada.csv       : base com metadados e flags de auditoria
  - saidas/etapa0_resumo.txt       : tabelas do diagnóstico
Nenhum modelo é treinado aqui.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
ARQ = RAIZ / "dados" / "Historico_TML_2025_2026.xlsx"
SAIDA = RAIZ / "saidas"
SAIDA.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

log_linhas = []


def log(*args):
    txt = " ".join(str(a) for a in args)
    print(txt)
    log_linhas.append(txt)


# ---------------------------------------------------------------------------
# 1. Leitura fiel
# ---------------------------------------------------------------------------
RENOMEIA = {
    "Ano": "ano",
    "Material": "material",
    "PFUF": "pfuf",
    "PM": "pm",
    "PFSA": "pfsa",
    "Data": "data",
    "SiO2": "sio2",
    "P": "p",
    "PPC": "ppc",
    "Densidade (g/cm³)": "densidade",
    "-325#": "menos_325",
    "Hematita Especular": "hem_especular",
    "Hematita Porosa": "hem_porosa",
    "Magnetita": "magnetita",
    "Goethita": "goethita",
    "TML %": "tml",
    "Umidade Embarque": "umid_embarque",
    "Umidade x TML": "umid_x_tml",
    "Validade": "validade",
}

abas = pd.read_excel(ARQ, sheet_name=None, header=3)
partes = []
for nome, df in abas.items():
    df = df.dropna(axis=1, how="all").rename(columns=RENOMEIA)
    df["aba"] = nome
    df["linha_excel"] = df.index + 5  # cabeçalho na linha 4
    partes.append(df)
bruta = pd.concat(partes, ignore_index=True, sort=False)
bruta.to_csv(SAIDA / "base_bruta.csv", index=False)

log("=" * 100)
log("1. ESTRUTURA")
log("=" * 100)
for nome, df in abas.items():
    log(f"Aba '{nome}': {df.shape[0]} linhas x {df.dropna(axis=1, how='all').shape[1]} colunas")
log(f"Total concatenado: {bruta.shape[0]} linhas x {bruta.shape[1]} colunas")
log("\nTipos e preenchimento por coluna (base concatenada):")
tab = pd.DataFrame({
    "tipo": bruta.dtypes.astype(str),
    "n_preenchidos": bruta.notna().sum(),
    "pct_faltante": (bruta.isna().mean() * 100).round(1),
    "n_distintos": bruta.nunique(dropna=True),
})
log(tab.to_string())

# ---------------------------------------------------------------------------
# 2. Metadados extraídos do campo texto "Material"
# ---------------------------------------------------------------------------
def sem_acento(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def classifica_produto(m, aba):
    t = sem_acento(m)
    if "psc" in aba.lower() or t.startswith("psc"):
        return "PSC"
    if re.search(r"pf\s*/?\s*s\s*/?\s*a\b|pfsa", t):
        return "PF_SA"
    if "ppc alto" in t:
        return "PFN_PPC_ALTO"
    if "pfn" in t:
        return "PFN"
    return "INDEFINIDO"


def classifica_origem(m):
    t = sem_acento(m)
    if "navio" in t or "embarcado" in t or re.search(r"\bemb\.", t):
        return "navio"
    if "alimentacao" in t:
        return "alimentacao_usina"
    if "filtrado" in t:
        return "filtrado"
    if "patio" in t or "bancada" in t or "empilhado" in t:
        return "patio"
    if "producao" in t or "batch" in t or "pot grate" in t or "us 03/04" in t:
        return "producao"
    return "outro"


def extrai_navio(m):
    t = str(m)
    t = re.sub(r"\s+", " ", t).strip()
    mm = re.search(r"[Nn]avio\s*:?\s*(.+?)(?:\s*[-–]\s*|\s+para\s+|\s+Cliente|\s*$)", t)
    if not mm:
        return None
    nome = mm.group(1)
    nome = re.sub(r"\b(Cliente|PSU Fino|T\s*0?\d|ofc|Sem Desidratante|Com Desidratante)\b.*$", "", nome, flags=re.I)
    nome = nome.strip(" :-.")
    # normalizações de grafia observadas na planilha
    nome = sem_acento(nome)
    nome = nome.replace("sydneiy", "sydney")
    nome = re.sub(r"o[lr]dendorf+", "oldendorff", nome)
    nome = re.sub(r"\s+", " ", nome)
    return nome.title() if nome else None


def extrai_cliente(m):
    t = re.sub(r"\s+", " ", str(m))
    mm = re.search(r"Cliente\s*:?\s*([A-Za-z][A-Za-z ]*?)\s*(?:ofc|$)", t)
    if not mm:
        mm = re.search(r"\bpara\s+([A-Za-z][A-Za-z ]*?)\s*(?:\.|T\s*0?\d|$)", t)
    if not mm:
        mm = re.search(r"[Nn]avio\s*:?\s*[A-Za-z0-9 ]+?\s*[-–]\s*([A-Za-z][A-Za-z ]*?)\s*(?:[-–]|$)", t)
    if not mm:
        return None
    c = sem_acento(mm.group(1)).strip()
    c = c.replace("tosyale", "tosyali").replace("psu fino", "psu")
    return c.title() if c else None


def flag(m, padrao):
    return int(bool(re.search(padrao, sem_acento(m))))


b = bruta.copy()
b["produto"] = [classifica_produto(m, a) for m, a in zip(b["material"], b["aba"])]
b["origem"] = b["material"].map(classifica_origem)
b["navio"] = b["material"].map(extrai_navio)
b["cliente"] = b["material"].map(extrai_cliente)
b["desidratante"] = b["material"].map(lambda m: flag(m, r"desidratante"))
b["sem_desidratante"] = b["material"].map(lambda m: flag(m, r"sem desidratante"))
b["replica"] = b["material"].map(lambda m: flag(m, r"\bt\s*-?\s*0?\d\b|interlaboratorial"))

# grupo de amostragem: navio quando houver, senão a própria linha (lote independente)
b["grupo"] = b["navio"].fillna("lote_" + b["linha_excel"].astype(str))

log("\n" + "=" * 100)
log("2. METADADOS EXTRAÍDOS DO CAMPO 'Material'")
log("=" * 100)
log("\nProduto (classificado por regra textual):")
log(b.groupby(["aba", "produto"]).size().to_string())
log("\nOrigem da amostra:")
log(b["origem"].value_counts().to_string())
log("\nAmostras com desidratante mencionado:", int(b["desidratante"].sum()))
log("Amostras marcadas como réplica/teste repetido (T01/T02/Interlaboratorial):", int(b["replica"].sum()))
log("\nNavios identificados e nº de amostras por navio:")
nav = b.dropna(subset=["navio"]).groupby("navio").agg(n=("tml", "size"), produtos=("produto", lambda s: ",".join(sorted(set(s)))), tml=("tml", lambda s: list(s.round(2))))
log(nav.sort_values("n", ascending=False).to_string())
log(f"\nNº de navios distintos: {b['navio'].nunique()}; amostras ligadas a navio: {b['navio'].notna().sum()}; "
    f"amostras sem navio (pátio/produção/alimentação): {b['navio'].isna().sum()}")
log("\nClientes identificados:")
log(b["cliente"].value_counts(dropna=False).to_string())
log("\nConferência manual da extração (linha, material -> produto | origem | navio | cliente):")
for _, r in b.iterrows():
    log(f"  L{r.linha_excel:>2} | {str(r.material)[:62]:<62} -> {r.produto:<12} | {r.origem:<17} | {str(r.navio):<22} | {r.cliente}")

# ---------------------------------------------------------------------------
# 3. Datas
# ---------------------------------------------------------------------------
log("\n" + "=" * 100)
log("3. DATAS")
log("=" * 100)
b["data"] = pd.to_datetime(b["data"], errors="coerce")
log(f"Faixa: {b['data'].min().date()} a {b['data'].max().date()}; sem data: {b['data'].isna().sum()} linha(s) -> "
    + str(b.loc[b['data'].isna(), 'linha_excel'].tolist()))
log("Amostras por ano-mês:")
log(b.groupby(b["data"].dt.to_period("M")).size().to_string())
ordem = b.dropna(subset=["data"]).sort_values("linha_excel")
fora_ordem = (ordem["data"].diff().dt.days < 0).sum()
log(f"Linhas fora de ordem cronológica em relação à anterior: {fora_ordem}")

# ---------------------------------------------------------------------------
# 4. Faltantes, constantes, quase constantes
# ---------------------------------------------------------------------------
log("\n" + "=" * 100)
log("4. FALTANTES / CONSTANTES (apenas aba PFN, n=65, pois PSC é outro produto)")
log("=" * 100)
num_cols = ["pfuf", "pm", "pfsa", "sio2", "p", "ppc", "densidade", "menos_325",
            "hem_especular", "hem_porosa", "magnetita", "goethita", "tml", "umid_embarque"]
pfn = b[b["aba"].str.contains("PFN")].copy()
falt = pd.DataFrame({
    "n_preenchidos": pfn[num_cols].notna().sum(),
    "pct_faltante": (pfn[num_cols].isna().mean() * 100).round(1),
    "pct_falt_2025": (pfn.loc[pfn.ano == 2025, num_cols].isna().mean() * 100).round(1),
    "pct_falt_2026": (pfn.loc[pfn.ano == 2026, num_cols].isna().mean() * 100).round(1),
    "n_distintos": pfn[num_cols].nunique(),
    "cv_pct": (pfn[num_cols].std() / pfn[num_cols].mean() * 100).round(1),
})
falt["flag"] = np.where(falt.pct_faltante > 30, ">30% faltante",
                np.where(falt.n_distintos <= 1, "constante",
                np.where(falt.cv_pct < 5, "quase constante (CV<5%)", "")))
log(falt.to_string())
log("\nDensidade: CV baixo é esperado (propriedade física pouco variável), mas a faixa 4,51 a 4,78 g/cm³ é informativa.")
log("Densidade em função do produto (média):")
log(b.groupby("produto")["densidade"].agg(["count", "mean", "min", "max"]).round(3).to_string())

# ---------------------------------------------------------------------------
# 5. Estatística descritiva, faixas físicas e outliers
# ---------------------------------------------------------------------------
log("\n" + "=" * 100)
log("5. DESCRITIVA, UNIDADES E OUTLIERS")
log("=" * 100)
log(b[num_cols].describe().T.round(3).to_string())

faixas = {  # faixas fisicamente plausíveis para pellet feed de minério de ferro (%, g/cm³)
    "sio2": (0.5, 8), "p": (0.02, 0.12), "ppc": (0.1, 6), "densidade": (4.3, 5.2),
    "menos_325": (70, 100), "hem_especular": (0, 100), "hem_porosa": (0, 100),
    "magnetita": (0, 20), "goethita": (0, 100), "tml": (7, 15), "umid_embarque": (6, 13),
    "pfuf": (0, 100), "pm": (0, 100), "pfsa": (0, 100),
}
log("\nValores fora de faixa física plausível:")
achou = False
for c, (lo, hi) in faixas.items():
    m = b[c].notna() & ((b[c] < lo) | (b[c] > hi))
    if m.any():
        achou = True
        log(f"  {c}: linhas {b.loc[m, 'linha_excel'].tolist()} valores {b.loc[m, c].tolist()}")
if not achou:
    log("  nenhum. Unidades coerentes (óxidos e TML em %, densidade em g/cm³).")

log("\nOutliers estatísticos (regra 1,5*IQR, dentro de cada produto, variáveis principais):")
for prod, g in b.groupby("produto"):
    if len(g) < 8:
        continue
    for c in ["sio2", "p", "ppc", "densidade", "menos_325", "hem_especular", "hem_porosa", "magnetita", "goethita", "tml"]:
        s = g[c].dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        m = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        if m.any():
            log(f"  {prod:<12} {c:<14} linhas {g.loc[m[m].index, 'linha_excel'].tolist()} valores {s[m].round(3).tolist()}  (faixa típica {q1 - 1.5 * iqr:.2f} a {q3 + 1.5 * iqr:.2f})")

log("\nSoma das fases mineralógicas (esperado ~95 a 100% quando todas as fases são reportadas):")
b["soma_mineral"] = b[["hem_especular", "hem_porosa", "magnetita", "goethita"]].sum(axis=1, min_count=1)
b["n_fases"] = b[["hem_especular", "hem_porosa", "magnetita", "goethita"]].notna().sum(axis=1)
log(b.dropna(subset=["soma_mineral"]).groupby(["ano", "n_fases"])["soma_mineral"].agg(["count", "min", "median", "max"]).round(1).to_string())
log("Leitura: em 2025 só 2 fases são reportadas (Hematita Porosa e Goethita) e a soma fica entre ~28 e ~85%,")
log("portanto a mineralogia de 2025 NÃO é comparável à de 2026 (4 fases, soma ~96 a 98%). Em 2025 a 'Hematita Porosa'")
log("provavelmente representa hematita total ou foi medida por outro método. Tratar como regimes distintos.")

# ---------------------------------------------------------------------------
# 6. Duplicidades
# ---------------------------------------------------------------------------
log("\n" + "=" * 100)
log("6. DUPLICIDADES")
log("=" * 100)
quim = ["sio2", "p", "ppc"]
dup_quim = b[b.duplicated(subset=quim, keep=False) & b[quim].notna().all(axis=1)].sort_values(quim + ["linha_excel"])
log("Linhas com química (SiO2, P, PPC) idêntica a outra linha (mesma amostra física ensaiada mais de uma vez ou cópia):")
log(dup_quim[["linha_excel", "material", "data", "sio2", "p", "ppc", "densidade", "tml", "umid_embarque"]].to_string(index=False))
dup_total = b.duplicated(subset=num_cols + ["data"], keep=False).sum()
log(f"\nLinhas totalmente duplicadas (todas as variáveis + data): {dup_total}")
log("\nPares de réplica da MESMA amostra (dão estimativa da repetibilidade do ensaio TML):")
for ids, rotulo in [((43, 44), "PF SA Int. Vale T01/T02"), ((54, 55), "PFN Interlab Vale T01 (2 datas)"),
                    ((56, 57), "Alexandra T01/T02"), ((20, 21), "Xin He vs empilhado c/ desidratante"),
                    ((24, 25), "Ksl Sydney sem/com desidratante"), ((38, 39), "Azimuth sem/com desidratante"),
                    ((33, 35), "Azimuth embarcado vs 'ofc'")]:
    v = b.set_index("linha_excel").loc[list(ids), "tml"].tolist()
    log(f"  L{ids[0]}/L{ids[1]} {rotulo:<40} TML = {v[0]:.2f} / {v[1]:.2f}  dif = {abs(v[0] - v[1]):.2f} pp")
rep = b.set_index("linha_excel").loc[[43, 44, 54, 55, 56, 57], "tml"]
pares = np.array([[rep[43], rep[44]], [rep[54], rep[55]], [rep[56], rep[57]]])
sd_rep = np.sqrt(np.mean((pares[:, 0] - pares[:, 1]) ** 2) / 2)
log(f"\nDesvio padrão de repetibilidade estimado com as 3 réplicas verdadeiras: {sd_rep:.3f} pp "
    f"(limite de repetibilidade 2,8*sd = {2.8 * sd_rep:.2f} pp). Qualquer modelo com RMSE abaixo disso está sobreajustado.")
log("\nALERTA L35: 'PFN STD Emb. Navio Azimuth ofc' tem química idêntica à L33 (Azimuth), mas TML=11,35 e Umidade Embarque=10,98,")
log("valores idênticos aos da L31 (navio Ngm Bond). Validade +89 dias em vez de +179. Forte suspeita de erro de digitação/cópia.")
log("Decisão: excluir L35 da modelagem (não é réplica confiável nem lote novo).")

# ---------------------------------------------------------------------------
# 7. Vazamento e disponibilidade no momento da previsão
# ---------------------------------------------------------------------------
log("\n" + "=" * 100)
log("7. VAZAMENTO DE INFORMAÇÃO E DISPONIBILIDADE NO MOMENTO DA PREVISÃO")
log("=" * 100)
log("umid_x_tml : fórmula =(TML - Umidade Embarque)/TML. Calculada a partir do alvo. VAZAMENTO DIRETO. Excluída.")
log("validade   : fórmula =Data+179 (uma linha com +89). Sem informação própria. Excluída.")
log("umid_embarque: umidade real do lote embarcado, medida DEPOIS do TML e controlada pela operação para ficar abaixo do TML")
log("             (tipicamente ~90% do TML). Não existe no momento da previsão e é consequência do TML. VAZAMENTO INDIRETO. Excluída.")
m = b[["tml", "umid_embarque"]].dropna()
log(f"             Correlação TML x Umidade embarque (n={len(m)}): r = {m.corr().iloc[0, 1]:.2f}; razão média umid/TML = {(m.umid_embarque / m.tml).mean():.3f}")
log(f"             Lotes embarcados com Umidade > TML: {(m.umid_embarque > m.tml).sum()} de {len(m)} -> "
    + str(b.loc[m.index[m.umid_embarque > m.tml], ['linha_excel', 'navio', 'tml', 'umid_embarque']].to_dict('records')))
log("ano        : redundante com data. Excluída como variável; usada apenas para estratificação temporal.")
log("material   : texto livre com identidade do navio/cliente. Usado SÓ para derivar produto, origem, desidratante e grupo.")
log("navio/cliente: identidade. Entram apenas como GRUPO da validação cruzada, nunca como preditor.")
log("pfuf, pm, pfsa: preenchidos em 17 linhas. Hipótese: % de outros produtos (PF ultrafino, PM, PF SA) misturados ao lote")
log("             embarcado; vazio = 0% (lote puro). Disponíveis antes do embarque (plano de blend). Mantidas com imputação 0, sob confirmação.")
log("sio2, p, ppc, densidade, -325#, mineralogia: ensaios de caracterização da mesma amostra, independentes do ensaio de TML. Candidatas legítimas.")

# ---------------------------------------------------------------------------
# 8. Correlações com o alvo (exploratório, sem modelo)
# ---------------------------------------------------------------------------
log("\n" + "=" * 100)
log("8. CORRELAÇÃO DE SPEARMAN COM O TML (aba PFN, n variável por coluna; exploratório)")
log("=" * 100)
cand = ["sio2", "p", "ppc", "densidade", "menos_325", "hem_especular", "hem_porosa", "magnetita", "goethita", "pfuf", "pm", "pfsa", "umid_embarque"]
rows = []
for c in cand:
    d = pfn[[c, "tml"]].dropna()
    if len(d) >= 5:
        rows.append({"variavel": c, "n": len(d), "spearman": round(d[c].corr(d["tml"], method="spearman"), 2),
                     "pearson": round(d[c].corr(d["tml"]), 2)})
log(pd.DataFrame(rows).to_string(index=False))
log("\nTML por produto e ano:")
log(b.groupby(["produto", "ano"])["tml"].agg(["count", "mean", "std", "min", "max"]).round(2).to_string())
log("\nTML por origem da amostra (PFN):")
log(pfn.groupby("origem")["tml"].agg(["count", "mean", "std"]).round(2).to_string())
log("\nCorrelação entre preditores (Pearson, PFN):")
log(pfn[["sio2", "p", "ppc", "densidade", "menos_325", "hem_especular", "hem_porosa", "magnetita", "goethita"]].corr().round(2).to_string())

# ---------------------------------------------------------------------------
# 9. Decisões de limpeza e impacto no tamanho
# ---------------------------------------------------------------------------
log("\n" + "=" * 100)
log("9. DECISÕES DE LIMPEZA E IMPACTO NO TAMANHO DA BASE")
log("=" * 100)
b["excluir_motivo"] = ""
b.loc[b["tml"].isna(), "excluir_motivo"] = "sem TML (alvo ausente)"
b.loc[b["linha_excel"].eq(35) & b["aba"].str.contains("PFN"), "excluir_motivo"] = "suspeita de cópia/digitação (L35)"
b.loc[b["aba"].str.contains("PSC"), "excluir_motivo"] = "produto PSC (n=3, modelado à parte / só baseline)"
b["replica_consolidar"] = 0
b.loc[b["linha_excel"].isin([44, 55, 57]) & b["aba"].str.contains("PFN"), "replica_consolidar"] = 1

n0 = len(b)
n1 = (b["excluir_motivo"] == "").sum()
n2 = ((b["excluir_motivo"] == "") & (b["replica_consolidar"] == 0)).sum()
log(f"Linhas lidas (2 abas):                                   {n0}")
log(f"Após excluir sem alvo, L35 suspeita e aba PSC:           {n1}")
log(f"Após consolidar réplicas T01/T02 pela média (3 pares):   {n2}  (base de modelagem PFN + PF SA)")
pfn_mod = b[(b["excluir_motivo"] == "")]
log(f"  dos quais PFN/PFN_PPC_ALTO: {pfn_mod.produto.isin(['PFN', 'PFN_PPC_ALTO']).sum()}, PF_SA: {(pfn_mod.produto == 'PF_SA').sum()}")
log(f"  com mineralogia completa (4 fases) e -325#: {(pfn_mod.n_fases == 4).sum()} (todos de 2026)")
log(f"  ligados a navio (grupo real): {pfn_mod.navio.notna().sum()} amostras em {pfn_mod.navio.nunique()} navios")
log("\nTratamento de faltantes (a ser aplicado DENTRO de cada dobra, via pipeline):")
log("  pfuf/pm/pfsa -> 0 (ausência = não há mistura), sob confirmação.")
log("  menos_325, mineralogia 2026 -> cenário A: usar só base 2026 completa; cenário B: excluir essas variáveis e usar toda a base.")
log("  mineralogia 2025 (2 fases) -> não usar como se fosse a mesma grandeza de 2026.")
log("  Imputação por mediana apenas para faltas pontuais (< 10%) em sio2/p/ppc/densidade; aqui não há nenhuma.")

b.to_csv(SAIDA / "base_auditada.csv", index=False)
(SAIDA / "etapa0_resumo.txt").write_text("\n".join(log_linhas), encoding="utf-8")
log(f"\nArquivos gravados em {SAIDA}")
