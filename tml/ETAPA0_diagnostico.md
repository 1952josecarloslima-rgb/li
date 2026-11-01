# Etapa 0. Diagnóstico da base de TML (Histórico 2025-2026, Laboratório de Controle de Embarque)

Arquivos: `tml/etapa0_auditoria.py` (script), `tml/saidas/etapa0_resumo.txt` (tabelas completas),
`tml/saidas/base_auditada.csv` (base com metadados e flags). Nenhum modelo foi treinado.

## Hipóteses adotadas (os campos do briefing vieram em branco)

| Item | Hipótese adotada |
|---|---|
| Variável alvo | coluna `TML %` |
| Produto principal | PFN (pellet feed normal/standard), que concentra 55 dos 68 registros. PF SA (8) e PFN PPC alto (2) entram com indicador de produto. PSC (3) fica fora da modelagem e só recebe linha de base |
| Identificador de navio/lote | não existe coluna própria. Extraído por regra textual do campo `Material` (ex. "Navio: Azimuth"). Amostras de pátio, produção e alimentação são lotes independentes |
| Data | coluna `Data` (data do ensaio) |
| Variáveis candidatas | todas, exceto identificadores e colunas derivadas do alvo (ver seção 4) |

## 1. Estrutura

| Aba | Linhas de dados | Colunas úteis | Observação |
|---|---|---|---|
| Histórico PFN 2025-2026 | 65 | 19 | linhas 5 a 69 da planilha, cabeçalho na linha 4 |
| Histórico PSC 2025-2026 | 3 | 11 | sem granulometria e mineralogia |

- Faixa de datas: 09/01/2025 a 15/10/2026. Uma linha sem data e sem TML (L69).
- Distribuição temporal muito irregular: 23 das 68 amostras estão em novembro e dezembro de 2025; vários meses têm 1 ou 2 amostras.
- Produtos (por regra textual): PFN 55, PF SA 8, PFN PPC alto 2, PSC 3.
- Origem da amostra: navio 36, produção 13, pátio 7, filtrado 7, alimentação de usina 3, outro 2.
- Navios: 30 distintos, 36 amostras ligadas a navio. Só 4 navios têm mais de uma amostra (Azimuth 4, Alexandra 2, ASL Classic 2 com produtos diferentes, Ksl Sydney 2). Os outros 32 registros são lotes de pátio, filtragem ou produção, sem navio.
- Tipos: todas as variáveis analíticas são numéricas e com unidade coerente (óxidos, PPC, fases e TML em %, densidade em g/cm³, -325# em % passante). Nenhum valor fora de faixa física plausível.

## 2. Alvo

| Estatística do TML (%) | Valor |
|---|---|
| n | 67 |
| média | 11,26 |
| desvio padrão | 0,48 |
| mínimo | 10,43 (navio Frontier Explorer, 07/2026) |
| máximo | 12,64 (pátio A 17/01/2026, PPC 4,32 e goethita 43,6; coerente fisicamente, mantido) |
| repetibilidade do ensaio (3 pares T01/T02) | sd = 0,03 pp; limite de repetibilidade 0,08 pp |

A faixa útil do alvo é de cerca de 2 pontos percentuais e o coeficiente de variação é 3,8%. O erro de qualquer modelo deve ser lido contra o desvio padrão do alvo (0,48 pp) e contra a tolerância operacional de 0,5 pp. Um modelo que não reduzir o RMSE claramente abaixo de 0,48 pp não supera a média histórica.

Efeito da origem da amostra no TML (PFN): navio 10,96 (n=34), produção 11,43 (n=12), filtrado 11,53 (n=7), pátio 11,72 (n=6). As amostras embarcadas têm TML sistematicamente menor que as de pátio e filtragem. Isso será tratado como variável de contexto conhecida no momento da previsão e também como suspeita de confundimento (Etapa 3, teste 5).

## 3. Faltantes, duplicidades, outliers e unidades

### 3.1 Faltantes (aba PFN, n=65)

| Variável | % faltante total | % faltante 2025 | % faltante 2026 | Decisão |
|---|---|---|---|---|
| PFUF | 81,5 | 86,8 | 74,1 | hipótese: % de blend, vazio = 0 (ver pergunta 1) |
| PM | 89,2 | 89,5 | 88,9 | idem |
| PFSA | 93,8 | 100 | 85,2 | idem |
| -325# | 60,0 | 100 | 3,7 | só existe em 2026; cenário B |
| Hematita especular | 63,1 | 100 | 11,1 | só existe em 2026; cenário B |
| Magnetita | 64,6 | 100 | 14,8 | só existe em 2026; cenário B |
| Hematita porosa | 16,9 | 18,4 | 14,8 | 2025 não comparável a 2026 (ver 3.4) |
| Goethita | 16,9 | 18,4 | 14,8 | idem |
| Umidade embarque | 47,7 | 42,1 | 55,6 | excluída por vazamento |
| SiO2, P, PPC, densidade | 0 a 1,5 | 0 | 0 a 3,7 | completas |

Não há falta pontual em SiO2, P, PPC e densidade, portanto não haverá imputação por mediana nessas variáveis. A ausência de granulometria e mineralogia em 2025 divide a base em dois regimes de informação.

### 3.2 Variáveis constantes ou quase constantes

- Nenhuma variável é constante.
- Densidade: CV 1,3%, mas a faixa 4,51 a 4,78 g/cm³ separa produtos (PF SA 4,57; PFN 4,71; PSC 4,97) e correlaciona com SiO2 (r = -0,80). Mantida.
- P: apenas 18 valores distintos em 65 linhas, resolução de 0,001%. Pouca informação; será testada e provavelmente removida.
- -325#: CV 1,5% dentro de 2026 (84,5 a 89,9%). Pouca variabilidade, poder preditivo limitado.

### 3.3 Duplicidades

| Linhas | Caso | TML | Decisão |
|---|---|---|---|
| L43 / L44 | PF SA, mesma amostra, T01 e T02 | 11,60 / 11,63 | consolidar pela média (1 lote) |
| L54 / L55 | PFN interlaboratorial Vale, mesma amostra, datas 28 e 29/03 | 11,94 / 11,89 | consolidar pela média |
| L56 / L57 | navio Alexandra, T01 e T02 | 10,81 / 10,85 | consolidar pela média |
| L20 / L21 | PF SA Xin He e empilhado com desidratante, mesma química, densidade diferente | 11,30 / 11,33 | manter as duas (tratamentos diferentes), mesmo grupo |
| L24 / L25 | Ksl Sydney sem e com desidratante | 10,56 / 10,53 | manter, mesmo grupo |
| L38 / L39 | Azimuth sem e com desidratante | 10,75 / 11,01 | manter, mesmo grupo |
| L33 / L35 | Azimuth embarcado e "Azimuth ofc" | 10,86 / 11,35 | **excluir L35** |

Alerta sobre L35: a química é idêntica à de L33 (Azimuth), mas o TML (11,35) e a umidade de embarque (10,98) são exatamente os valores de L31 (navio Ngm Bond), e a validade foi calculada com +89 dias em vez de +179. A hipótese mais provável é cópia ou digitação incorreta. A linha foi excluída e deve ser verificada no laboratório.

Não há linha totalmente duplicada.

### 3.4 Outliers e coerência física

- Outliers estatísticos (1,5 IQR dentro do produto): SiO2 3,85 em L67 (SSI Brave, provavelmente blend com PF SA, PFSA = 1,86 preenchido), PPC 4,32 e TML 12,64 em L47, densidade 4,598 e 4,603 em L47 e L67, -325# 89,0 e 89,9 em L59 e L66. Todos fisicamente plausíveis e coerentes entre si (PPC alto acompanha goethita alta e densidade baixa). Nenhum foi removido. Serão monitorados na análise de resíduos e de influência.
- Mineralogia: em 2026 as quatro fases somam 96 a 98%, como esperado. Em 2025 só há hematita porosa e goethita e a soma varia de 41 a 85%, portanto a grandeza "hematita porosa" de 2025 não é a mesma de 2026 (provavelmente hematita total ou outro método). Decisão: a mineralogia de 2025 não será misturada com a de 2026.
- Desidratante: 4 amostras mencionam o aditivo no texto. O efeito observado é inconsistente (0,03 pp em Ksl Sydney, 0,26 pp em Azimuth). Será um indicador binário, com baixa expectativa de significância.

## 4. Relação física entre variáveis e vazamento de informação

| Coluna | Situação | Decisão |
|---|---|---|
| `Umidade x TML` | fórmula `=(TML - Umidade)/TML`, calculada a partir do alvo | **vazamento direto, excluída** |
| `Validade` | fórmula `=Data + 179` | sem informação própria, excluída |
| `Umidade Embarque` | medida após o TML, consequência da decisão de embarque, inexistente no momento da previsão | **vazamento indireto, excluída** |
| `Ano` | redundante com a data | só para estratificação temporal |
| `Material` | texto com identidade de navio e cliente | só para derivar produto, origem, desidratante e grupo |
| navio, cliente | identidade | só como grupo da validação cruzada, nunca como preditor |
| PFUF, PM, PFSA | provável % de outros produtos no lote embarcado, conhecida no plano de blend | mantidas com vazio = 0, sob confirmação |
| SiO2, P, PPC, densidade, -325#, mineralogia | caracterização da mesma amostra, independente do ensaio de TML | candidatas legítimas |

Achado inesperado: em 18 dos 36 lotes embarcados a `Umidade Embarque` da linha é maior que o `TML %` da mesma linha (ex. Ksl Sydney 11,08 contra 10,56; Frontier Explorer 10,83 contra 10,43). Ou o TML de referência da decisão foi outro certificado (mais antigo ou "oficial", como o L35), ou a coluna de umidade não representa a umidade final da carga. Isso não afeta a modelagem do TML, porque a coluna foi excluída, mas afeta a leitura operacional e precisa de esclarecimento.

Correlação exploratória com o TML (Spearman, PFN): PPC +0,60 (n=64), magnetita +0,56 (n=23), goethita +0,51 (n=54), densidade -0,50 (n=64), hematita porosa -0,42 (n=54), hematita especular -0,37 (n=23), -325# -0,28 (n=26), P -0,14, SiO2 +0,04. O sentido é coerente com a física do TML: mais goethita e PPC (mineral hidratado, poroso, menor densidade) retêm mais água antes de fluir, o que eleva o TML. Multicolinearidade relevante: SiO2 x densidade (-0,80), PPC x densidade (-0,67), PPC x goethita (+0,60).

## 5. Impacto das decisões no tamanho da base

| Etapa | Linhas |
|---|---|
| Lidas (2 abas) | 68 |
| Excluir sem alvo (L69), L35 suspeita e aba PSC (3) | 63 |
| Consolidar 3 pares de réplica pela média | **60** lotes para modelagem (54 PFN incl. PPC alto, 6 PF SA) |
| dos quais com mineralogia completa e -325# (só 2026) | 20 |
| dos quais ligados a navio | 31 amostras em 28 navios |

Cenário A (principal): 60 lotes, variáveis SiO2, P, PPC, densidade, produto, origem, desidratante, blend (PFUF, PM, PFSA). Cenário B (exploratório): 20 lotes de 2026 com granulometria e mineralogia; serve apenas para verificar se essas variáveis acrescentam algo, não para modelo operacional, porque 20 pontos não sustentam validação aninhada com confiança.

## 6. Perguntas objetivas (prossigo com a hipótese indicada se não houver resposta)

1. PFUF, PM e PFSA são percentuais de outros produtos misturados ao lote (blend)? Vazio significa 0%? Hipótese adotada: sim, vazio = 0.
2. Em 18 lotes embarcados a umidade de embarque supera o TML da linha. Qual TML foi a referência da decisão de embarque, e `Umidade Embarque` é a umidade final da carga? Hipótese adotada: a coluna não é usada na previsão, independentemente da resposta.
3. A "Hematita Porosa" de 2025 foi medida pelo mesmo método de 2026 (quatro fases)? Hipótese adotada: não, e a mineralogia de 2025 fica fora do conjunto de entrada.

## 7. Implicações para as próximas etapas

- Com 60 lotes e cerca de 8 variáveis, modelos de árvore e ensembles (XGBoost, LightGBM, CatBoost, MLP, Processo Gaussiano) serão treinados conforme o briefing, mas a expectativa a priori é que não superem com significância um modelo linear regularizado. A razão amostras por variável já é baixa.
- A tolerância operacional (0,5 pp) é da mesma ordem do desvio padrão do alvo (0,48 pp). O ganho possível sobre a média histórica é limitado e precisa ser demonstrado com intervalo, não com média de dobras.
- Validação: GroupKFold por grupo (navio ou lote) repetida com sementes diferentes, mais validação temporal com treino até 12/2025 e teste em 2026. O conjunto de teste final intocado será o último bloco cronológico (amostras de 06/2026 em diante, 11 lotes). Com essa base, o teste final terá baixo poder e será reportado com intervalo, não como veredito isolado.
- A base está em dois regimes de informação (2025 sem granulometria e mineralogia; 2026 completo), e o TML médio subiu de 11,10 em 2025 para 11,28 em 2026 no PFN. A deriva temporal será testada explicitamente (Etapa 3, testes 6 e 7).
