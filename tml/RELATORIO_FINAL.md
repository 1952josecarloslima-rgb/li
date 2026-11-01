# Previsão do TML de pellet feed (PFN) a partir do histórico 2025-2026. Relatório final das Etapas 1 a 5

Base: `tml/dados/Historico_TML_2025_2026.xlsx` (Laboratório de Controle de Embarque, UBU). Diagnóstico da base na Etapa 0: `tml/ETAPA0_diagnostico.md`. Código reprodutível em `tml/*.py`, saídas em `tml/saidas/`. Sementes fixas (20251101). Decisões confirmadas pelo usuário: PFUF, PM e PFSA são percentuais de outros produtos incorporados ao lote (vazio = 0%); a umidade de embarque fica fora da previsão; a mineralogia de 2025 não é comparável à de 2026 e ficou fora do conjunto de entrada.

## 1. Resumo executivo

1. Modelo recomendado: regressão Ridge com 12 variáveis (SiO2, P, PPC, densidade, % de PFUF, PM e PFSA no blend, indicadores de produto, de origem da amostra e de desidratante). Empatou estatisticamente com todos os ensembles e modelos de árvore e foi escolhido por ser o mais simples e interpretável.
2. Desempenho esperado: erro típico (RMSE) de 0,29 pp de TML na validação cruzada agrupada por navio, com intervalo entre dobras de 0,19 a 0,41 pp, e 0,33 pp na validação temporal. A média histórica erra 0,41 pp. O ganho é real (teste de permutação e teste pareado significativos), mas modesto: cerca de 0,12 pp.
3. No teste final intocado (11 lotes de abril a outubro de 2026) o erro subiu para 0,39 pp (IC 95% de 0,24 a 0,53), com viés de +0,22 pp (superestimação sistemática) e 2 dos 11 lotes superestimados em mais de 0,5 pp. A média histórica errou 0,47 pp e superestimou 4 de 11.
4. O check anti-espúrio reprovou o teste de variáveis proxy: 82% do ganho sobre a média vem de saber se a amostra é de navio ou de pátio, não da química. A química (PPC e densidade) tem sinal físico coerente e real, mas explica pouco da variação.
5. Intervalo de previsão calibrado (conformal): limite inferior unilateral a 95% = TML previsto menos 0,54 pp, com cobertura empírica de 96% na validação e 91% (10 de 11) no teste final. O intervalo simétrico a 90% tem 1,06 pp de largura, metade da faixa histórica do TML.
6. Principais riscos: superestimação nos lotes de TML baixo (viés de +0,22 pp na faixa abaixo de 10,9%), deriva entre períodos (bloco de novembro de 2025 com 54% de superestimações no walk-forward), base de 60 lotes e ausência de granulometria e mineralogia comparáveis.
7. Veredito: o modelo não deve ser usado como base da decisão de embarque, nem para substituir ou adiar o ensaio de TML. Pode ser usado como triagem de plausibilidade (sinalizar lotes cujo TML previsto menos 0,54 pp fique próximo da umidade planejada) e para priorizar reensaios.
8. Nível de confiança para uso operacional: baixo. Justificativa na seção 9.

## 2. Protocolo de validação (Etapa 1)

- Base de modelagem: 60 lotes (54 PFN, inclusive 2 de PPC alto, e 6 PF SA), após as exclusões da Etapa 0. PSC (3 lotes) ficou fora.
- Teste final intocado: os 11 lotes com data a partir de 01/04/2026. Usado uma única vez, após a escolha do modelo, na seção 6.
- Desenvolvimento: 49 lotes em 45 grupos (navio quando há navio, senão o próprio lote).
- Validação cruzada agrupada: GroupKFold com 5 dobras, embaralhado, repetido com 5 sementes (25 dobras). Nenhuma amostra do mesmo navio aparece em treino e teste ao mesmo tempo.
- Validação temporal: walk-forward com janela expansiva, 4 blocos de teste (set-out/2025, nov/2025, dez/2025, jan-mar/2026), treinando só com o passado.
- Validação aninhada: busca aleatória de hiperparâmetros (RandomizedSearchCV, 15 combinações, GroupKFold interno com 4 dobras) refeita dentro de cada dobra externa. Optuna não foi usado por não trazer ganho com espaços deste tamanho; RandomizedSearch foi a alternativa informada.
- Todo pré-processamento (imputação, escala, PCA, polinômios) está dentro do Pipeline e é ajustado só com o treino de cada dobra.
- Testes pareados entre modelos: teste t corrigido de Nadeau-Bengio sobre as 25 dobras. Regra de empate: diferença não significativa (p > 0,05) ou dentro de 1 erro padrão do melhor. Em caso de empate, escolhe-se o mais simples.
- Biblioteca GAM: pygam (instalada). XGBoost, LightGBM e CatBoost instalados.

## 3. Ranking dos modelos (Etapa 2)

Validação agrupada por navio (média e desvio padrão em 25 dobras; p-valores do teste pareado corrigido; última coluna: RMSE médio no walk-forward temporal).

| # | Modelo | Família | RMSE médio ± dp | RMSE p5 a p95 | MAE | R² | Viés | Erro p95 | Super. > 0,5 pp (%) | RMSE treino | p vs melhor | p vs média | RMSE temporal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | voting (huber + knn + lightgbm) | ensemble | 0,262 ± 0,072 | 0,18 a 0,40 | 0,197 | 0,47 | −0,015 | 0,47 | 2,7 | 0,123 | 1,000 | 0,001 | 0,354 |
| 2 | stacking (mesmos componentes, meta Ridge) | ensemble | 0,271 ± 0,087 | 0,18 a 0,43 | 0,204 | 0,43 | 0,000 | 0,49 | 2,6 | 0,103 | 0,494 | 0,001 | 0,319 |
| 3 | knn | não linear | 0,274 ± 0,083 | 0,17 a 0,42 | 0,217 | 0,41 | −0,016 | 0,48 | 2,5 | 0,078 | 0,523 | 0,001 | 0,321 |
| 4 | lightgbm | árvores/boosting | 0,280 ± 0,099 | 0,18 a 0,45 | 0,210 | 0,40 | −0,002 | 0,51 | 2,6 | 0,172 | 0,454 | 0,001 | 0,295 |
| 5 | random_forest | árvores/boosting | 0,283 ± 0,091 | 0,19 a 0,47 | 0,212 | 0,38 | −0,009 | 0,51 | 3,5 | 0,183 | 0,154 | 0,002 | 0,332 |
| 6 | processo_gaussiano | não linear | 0,285 ± 0,112 | 0,18 a 0,50 | 0,215 | 0,38 | −0,010 | 0,51 | 1,7 | 0,112 | 0,516 | 0,002 | 0,344 |
| 7 | huber | linear | 0,286 ± 0,053 | 0,23 a 0,38 | 0,221 | 0,34 | −0,014 | 0,52 | 3,5 | 0,211 | 0,323 | 0,015 | 0,445 |
| 8 | gradient_boosting | árvores/boosting | 0,286 ± 0,094 | 0,17 a 0,46 | 0,214 | 0,36 | 0,003 | 0,52 | 3,9 | 0,175 | 0,111 | 0,003 | 0,321 |
| 9 | **ridge (escolhido)** | linear | 0,286 ± 0,076 | 0,19 a 0,41 | 0,214 | 0,36 | −0,020 | 0,53 | 4,4 | 0,226 | 0,309 | 0,008 | 0,330 |
| 10 | extra_trees | árvores/boosting | 0,288 ± 0,077 | 0,18 a 0,43 | 0,220 | 0,34 | −0,004 | 0,51 | 2,4 | 0,161 | 0,182 | 0,005 | 0,322 |
| 11 | lasso | linear | 0,288 ± 0,064 | 0,20 a 0,40 | 0,226 | 0,34 | −0,008 | 0,52 | 5,0 | 0,232 | 0,190 | 0,007 | 0,354 |
| 12 | elasticnet | linear | 0,291 ± 0,065 | 0,20 a 0,42 | 0,225 | 0,35 | −0,001 | 0,52 | 5,6 | 0,238 | 0,171 | 0,004 | 0,353 |
| 13 | pls | linear | 0,291 ± 0,062 | 0,22 a 0,39 | 0,216 | 0,28 | −0,018 | 0,54 | 4,4 | 0,226 | 0,358 | 0,054 | 0,329 |
| 14 | pcr | linear | 0,291 ± 0,057 | 0,21 a 0,38 | 0,224 | 0,31 | −0,034 | 0,53 | 4,4 | 0,230 | 0,323 | 0,024 | 0,348 |
| 15 | svr_linear | não linear | 0,295 ± 0,059 | 0,23 a 0,39 | 0,232 | 0,32 | −0,016 | 0,53 | 4,7 | 0,229 | 0,062 | 0,009 | 0,353 |
| 16 | catboost | árvores/boosting | 0,295 ± 0,091 | 0,19 a 0,45 | 0,222 | 0,30 | −0,006 | 0,54 | 2,2 | 0,094 | 0,110 | 0,015 | 0,325 |
| 17 | xgboost | árvores/boosting | 0,301 ± 0,104 | 0,18 a 0,47 | 0,225 | 0,27 | −0,002 | 0,56 | 3,9 | 0,169 | 0,210 | 0,027 | 0,336 |
| 18 | mlp | não linear | 0,303 ± 0,093 | 0,20 a 0,49 | 0,222 | 0,18 | −0,008 | 0,57 | 4,4 | 0,210 | 0,396 | 0,121 | 0,324 |
| 19 | ols | linear | 0,303 ± 0,051 | 0,23 a 0,38 | 0,238 | 0,23 | −0,008 | 0,55 | 4,4 | 0,206 | 0,161 | 0,063 | 0,479 |
| 20 | arvore | árvores/boosting | 0,304 ± 0,100 | 0,19 a 0,49 | 0,233 | 0,27 | −0,005 | 0,55 | 3,5 | 0,235 | 0,095 | 0,013 | 0,330 |
| 21 | svr_rbf | não linear | 0,308 ± 0,072 | 0,24 a 0,44 | 0,234 | 0,24 | −0,004 | 0,55 | 4,7 | 0,201 | 0,069 | 0,036 | 0,360 |
| 22 | linear_ppc (base: só PPC) | linha de base | 0,335 ± 0,093 | 0,19 a 0,47 | 0,266 | 0,10 | −0,003 | 0,61 | 7,9 | 0,314 | 0,070 | 0,148 | 0,348 |
| 23 | gam | não linear | 0,339 ± 0,144 | 0,23 a 0,49 | 0,248 | −0,13 | 0,025 | 0,63 | 4,4 | 0,179 | 0,332 | 0,463 | 0,376 |
| 24 | poly2_ridge | não linear | 0,380 ± 0,100 | 0,20 a 0,53 | 0,289 | −0,11 | −0,009 | 0,68 | 9,0 | 0,265 | 0,004 | 0,456 | 1,279 |
| 25 | mediana_por_produto (base) | linha de base | 0,405 ± 0,105 | 0,28 a 0,61 | 0,322 | −0,19 | 0,012 | 0,71 | 13,1 | 0,403 | 0,000 | 0,415 | 0,425 |
| 26 | media_historica (base) | linha de base | 0,410 ± 0,112 | 0,27 a 0,62 | 0,323 | −0,22 | −0,004 | 0,72 | 13,1 | 0,407 | 0,001 | 1,000 | 0,424 |

Leitura do ranking:

- Os 21 modelos entre voting (0,262) e svr_rbf (0,308) estão dentro do ruído: nenhum difere do melhor com p < 0,05 e a dispersão entre dobras (0,05 a 0,11) é maior que a diferença entre eles (0,046). Declara-se empate técnico e escolhe-se o Ridge.
- O Ridge perde 0,024 pp para o voting em média (p = 0,31). Em troca, tem RMSE de treino 0,23 contra 0,12 do voting, ou seja, sobreajusta muito menos, e produz equação explícita.
- Na validação temporal a ordem muda (LightGBM 0,295, Ridge 0,330, voting 0,354, Huber 0,445), o que confirma que as diferenças da validação agrupada não são estáveis. Huber e OLS degradam no tempo; poly2_ridge e GAM são instáveis com 49 lotes.
- A regressão simples em PPC (0,335) não difere do Ridge com significância (p = 0,21) nem da média (p = 0,15). PPC sozinho carrega boa parte do sinal químico.
- Taxa de superestimações acima de 0,5 pp: 4,4% para o Ridge contra 13,1% da média histórica.

## 4. Check de modelo falso ou espúrio (Etapa 3)

Modelo avaliado: Ridge (alpha ≈ 10,3, moda das dobras). Resultado: 3 aprovados, 6 em atenção, 1 reprovado.

| Teste | Resultado | Veredito |
|---|---|---|
| 1. Permutação do alvo | 30 embaralhamentos do TML com busca aninhada: RMSE 0,42 ± 0,02 (nível de acaso) contra 0,29 real; p = 0,032 | **aprovado** |
| 2. Ganho sobre linhas de base | ganho de 0,12 pp sobre a média (p = 0,008) e sobre a mediana por produto (p = 0,006); sobre a regressão simples em PPC, 0,05 pp (p = 0,21, não significativo) | **atenção** |
| 3. Sobreajuste e curva de aprendizado | RMSE treino 0,23 contra validação 0,29 (razão 1,27); curva de aprendizado: validação 0,37 com 12 lotes, 0,30 com 18, 0,28 com 39, ainda em queda | **aprovado** |
| 4. Multicolinearidade | VIF máximo 14,9 (SiO2) e 13,5 (indicador PF SA), que medem a mesma coisa; densidade 6,4; demais < 4. Com a regularização, todos os coeficientes relevantes mantêm o sinal em 100% das dobras | **atenção** |
| 5. Proxy e confundidores | RMSE com todas 0,28; só química 0,32; só contexto (produto, origem, desidratante) 0,31; só tempo 0,41. O contexto sozinho entrega 82% do ganho sobre a média. O modelo acerta em boa parte porque sabe se a amostra veio de navio ou de pátio | **reprovado** |
| 6. Tendência espúria | TML em ordem cronológica tem ACF1 0,46 e Durbin-Watson 1,07 (forte autocorrelação); resíduos fora da dobra ACF1 0,30, DW 1,38 (autocorrelação residual moderada). Modelo com alvo detrendizado: RMSE 0,284 contra 0,286, logo o desempenho não depende da tendência | **atenção** |
| 7. Estabilidade temporal e deriva | walk-forward: RMSE 0,18 / 0,55 / 0,18 / 0,41 por bloco, viés +0,16 / +0,45 / −0,10 / −0,30, superestimações > 0,5 pp de 54% no bloco de nov/2025. Vence a média em 4 de 4 blocos, mas o viés muda de sinal entre períodos. Deriva dev → teste final: PF SA ausente no teste (SMD 0,52), blends maiores (SMD 0,3 a 0,5) | **atenção** |
| 8. Plausibilidade física | PDP e SHAP: PPC +0,46 pp (p5→p95) e densidade −0,20 pp, ambos no sentido esperado; P −0,22, SiO2 −0,08, PFUF −0,16, PM −0,07, desidratante −0,15 sem expectativa física única; indicadores PPC alto (1 lote) e PFSA (2 lotes) indeterminados. Nenhum efeito fisicamente incoerente, mas três efeitos sem mecanismo estabelecido | **aprovado** |
| 9. Resíduos | Shapiro-Wilk p = 0,09; Breusch-Pagan p = 0,52 (sem heterocedasticidade); viés por faixa: TML < 10,9 superestimado em +0,22 pp, 10,9 a 11,3 +0,01, > 11,3 subestimado em −0,17 (regressão à média); 2026 subestimado em −0,21 no desenvolvimento. Um resíduo de −1,0 pp (lote de pátio L47, TML 12,64) | **atenção** |
| 10. Domínio de aplicação | Treino: SiO2 1,47 a 4,88% (PFN até 2,5%), P 0,048 a 0,076%, PPC 2,89 a 4,42%, densidade 4,51 a 4,78 g/cm³, PFUF até 6,3%, PM até 8,2%, PFSA até 6,9%. Produtos: PFN 42, PF SA 6, PPC alto 1. Dois lotes do teste final extrapolam (PFUF 11,3% e PM 10,2%) | **atenção** |

Conclusão do check: o modelo não é falso (sobrevive à permutação, não sobreajusta, tem efeitos físicos coerentes), mas é parcialmente espúrio no sentido operacional: a variável que mais contribui depois do PPC é a origem da amostra, que não é uma propriedade do minério. Isso exige a restrição de uso descrita na seção 9.

Análise complementar (arquivo `saidas/extra_subconjunto_navio.csv`): restringindo-se aos 25 lotes de navio do desenvolvimento (a população da decisão de embarque), um Ridge só com SiO2, P, PPC e densidade obtém RMSE 0,215 ± 0,059 contra 0,310 da média desses lotes, sem nenhuma superestimação acima de 0,5 pp em 10 repetições. O sinal químico existe e é mais limpo dentro da população-alvo, mas com 25 lotes esse resultado é indicativo, não conclusivo.

## 5. Importância e interpretação das variáveis

| Variável | SHAP médio (pp) | Coef. padronizado (média ± dp, 25 dobras) | Sinal consistente | Efeito PDP p5→p95 (pp) | Esperado | Leitura física |
|---|---|---|---|---|---|---|
| PPC | 0,114 | +0,132 ± 0,016 | 100% | +0,46 | + | perda ao fogo mede hidroxilas da goethita; mineral poroso e hidratado retém mais água antes de atingir o ponto de fluxo. Efeito mais forte e mais estável do modelo |
| Amostra de navio | 0,109 | −0,106 ± 0,013 | 100% | −0,22 | sem mecanismo | lotes embarcados têm TML 0,2 a 0,7 pp menor que lotes de pátio com química semelhante. Hipóteses: amostra de navio integra blend, desidratante e material compactado; ou diferença sistemática de amostragem. Variável de contexto, não de minério |
| Amostra de pátio/filtrado | 0,061 | +0,077 ± 0,020 | 100% | +0,20 | sem mecanismo | complemento da anterior |
| P | 0,048 | −0,058 ± 0,017 | 100% | −0,22 | sem expectativa | sem mecanismo direto; P varia em só 18 valores distintos e pode estar capturando lote ou período |
| Densidade | 0,045 | −0,060 ± 0,018 | 100% | −0,20 | − | menor densidade = mais goethita e porosidade, logo TML maior; coerente e redundante com PPC (r = −0,67) |
| PFUF no blend | 0,030 | −0,042 ± 0,017 | 100% | −0,16 | sem expectativa única | ultrafino eleva superfície específica (tende a elevar TML), mas finos também reduzem porosidade do empacotamento. Só 12 lotes com PFUF > 0 |
| Desidratante | 0,023 | −0,042 ± 0,013 | 100% | −0,15 | sem expectativa | 4 lotes; efeito não separável do navio |
| PM no blend | 0,019 | −0,043 ± 0,015 | 100% | −0,07 | desconhecido | composição do PM não informada; 7 lotes |
| PFSA no blend | 0,017 | +0,036 ± 0,023 | 80% | 0,00 | + | 2 lotes no desenvolvimento: indeterminado |
| SiO2 | 0,014 | −0,019 ± 0,013 | 88% | −0,08 | sem expectativa única | quartzo grosso e não poroso reduz TML, mas a variável marca o produto PF SA (VIF 14,9) |
| Produto PF SA | 0,004 | −0,005 ± 0,013 | 72% | −0,02 | + | anulado pela colinearidade com SiO2 e densidade |
| Produto PPC alto | 0,003 | −0,007 ± 0,006 | 72% | + | −0,09 | 1 lote: indeterminado |

Leitura geral: o modelo é essencialmente "PPC e densidade" (a goethita) mais um deslocamento por origem da amostra. Variáveis clássicas do TML (superfície específica ou Blaine, distribuição granulométrica completa, teor de goethita medido) não estão na base ou não são comparáveis entre anos, e isso limita o que qualquer modelo pode aprender aqui.

## 6. Teste final intocado (11 lotes, abril a outubro de 2026)

Avaliação feita uma única vez, após a escolha do Ridge. Linhas de base e o melhor numérico (voting) são mostrados só para contexto. Hiperparâmetros ajustados em GroupKFold sobre todo o desenvolvimento (30 combinações).

| Modelo | RMSE (IC 95% bootstrap) | MAE | R² | Viés | Erro p95 | Super. > 0,5 pp |
|---|---|---|---|---|---|---|
| ridge (escolhido) | 0,394 (0,24 a 0,53) | 0,319 | 0,03 | +0,221 | 0,67 | 18% (2 de 11) |
| voting | 0,372 (0,22 a 0,49) | 0,296 | 0,14 | +0,208 | 0,65 | 18% (2 de 11) |
| media_historica | 0,468 (0,33 a 0,58) | 0,413 | −0,36 | +0,241 | 0,75 | 36% (4 de 11) |
| mediana_por_produto | 0,452 (0,31 a 0,58) | 0,385 | −0,27 | +0,271 | 0,75 | 36% (4 de 11) |
| linear_ppc | 0,441 (0,30 a 0,55) | 0,371 | −0,21 | +0,301 | 0,66 | 36% (4 de 11) |

Lote a lote (Ridge): os erros acima de 0,5 pp foram o lote filtrado de 14/06/2026 (TML 10,66, previsto 11,48, +0,82) e o navio SSI Brave (10,79, previsto 11,30, +0,51). Os navios Frontier Explorer (+0,48) e Alpha Gallant (+0,49) ficaram no limite. O limite inferior a 95% (previsto menos 0,54) cobriu 10 de 11 lotes; a exceção foi o lote filtrado de 14/06.

Leitura: o teste final é desfavorável em relação à validação. O RMSE subiu de 0,29 para 0,39, o R² caiu para 0,03 e o viés passou de −0,02 para +0,22, isto é, o modelo superestimou sistematicamente o TML de 2026, exatamente a falha de maior risco. O TML médio dos lotes de teste (11,01) ficou abaixo do dos lotes de 2026 no desenvolvimento (11,57). Com 11 lotes, o intervalo de confiança do RMSE (0,24 a 0,53) inclui o valor da validação, mas a direção do viés é consistente com o walk-forward e com o teste 9, e não pode ser atribuída a acaso.

## 7. Incerteza (Etapa 4)

Intervalos calibrados por cross-conformal (resíduos fora da dobra interna, cobertura medida nas dobras externas agrupadas) e, como comparação, regressão quantílica linear.

| Método | Nominal | Cobertura empírica (média ± dp entre dobras) | Largura média (pp) | Margem média (pp) |
|---|---|---|---|---|
| conformal, limite inferior | 95% | 96,1% ± 7,7 | | 0,55 |
| conformal, limite inferior | 90% | 93,0% ± 11,8 | | 0,41 |
| conformal, simétrico | 95% | 95,0% ± 8,9 | 1,43 | 0,72 |
| conformal, simétrico | 90% | 91,3% ± 10,8 | 1,06 | 0,53 |
| quantílica, limite inferior | 95% | 91,7% ± 11,7 | | 0,57 |
| quantílica, limite inferior | 90% | 85,1% ± 15,0 | | 0,34 |
| quantílica, simétrica | 95% | 92,8% ± 11,3 | 1,34 | |
| quantílica, simétrica | 90% | 79,6% ± 14,4 | 0,94 | |

O método conformal manteve a cobertura nominal; a regressão quantílica subcobriu com 49 lotes. Margens calibradas nos 49 lotes de desenvolvimento (erro fora da dobra, sd 0,30 pp): 0,35 pp a 90%, 0,54 pp a 95%, 0,70 pp a 98%.

Margem de segurança operacional proposta: TML_seguro = TML_previsto − 0,54 pp (limite inferior a 95%). No teste final essa margem cobriu 10 de 11 lotes. Observação: a largura do intervalo a 90% (1,06 pp) corresponde a metade da faixa histórica do TML (10,4 a 12,6), o que mostra que a previsão é pouco informativa para discriminar lotes individuais.

## 8. Equação de aplicação (Ridge ajustado nos 60 lotes)

```
TML (%) = 15,3010
          − 0,0612 × SiO2 (%)
          − 11,1386 × P (%)
          + 0,3496 × PPC (%)
          − 0,9421 × Densidade (g/cm³)
          − 0,0203 × PFUF no blend (%)
          − 0,0179 × PM no blend (%)
          + 0,0377 × PFSA no blend (%)
          + 0,0867 × [produto PF SA]
          + 0,0489 × [produto PPC alto]
          − 0,2352 × [amostra de navio]
          + 0,1433 × [amostra de pátio ou filtrado]
          − 0,0939 × [com desidratante]

TML seguro (limite inferior 95%) = TML − 0,54
```

Indicadores valem 1 ou 0. Para amostra de produção ou alimentação de usina, ambos os indicadores de origem valem 0. O pipeline salvo (`saidas/modelo_tml_final.joblib`, metadados em `saidas/modelo_tml_final.json`) reproduz exatamente esta equação com a ordem de colunas listada em `variaveis_entrada_ordem`. Aplicação: `joblib.load(...).predict(X)` com X na ordem SiO2, P, PPC, densidade, PFUF, PM, PFSA, is_pfsa, is_ppc_alto, orig_navio, orig_patio, desidratante.

Procedimento simplificado para triagem: calcular o TML previsto, subtrair 0,54 e comparar com a umidade planejada do lote. Se a umidade planejada ficar acima de TML_seguro, tratar o lote como prioritário para ensaio e não embarcar sem resultado de laboratório. Em nenhuma hipótese usar o valor previsto como TML certificado.

## 9. Veredito, limitações e recomendações

Nível de confiança para uso operacional: **baixo**. Razões: (i) o teste final mostrou superestimação sistemática (+0,22 pp) e 18% de lotes acima da tolerância; (ii) o teste de proxy reprovou, com 82% do ganho vindo da origem da amostra; (iii) o intervalo de 90% tem 1,06 pp de largura em um alvo que varia 2,2 pp; (iv) 60 lotes, dos quais 6 PF SA e 1 PPC alto, com granulometria e mineralogia inexistentes ou incomparáveis em 2025. O modelo pode ser usado como ferramenta de triagem e de priorização de ensaios, sempre com a margem de 0,54 pp e nunca como substituto do ensaio de TML na decisão de embarque.

Limitações:

1. Tamanho: 49 lotes de desenvolvimento e 11 de teste; a curva de aprendizado ainda cai, ou seja, falta dado.
2. Regime de informação: granulometria (−325#) e mineralogia em quatro fases só existem em 2026 (20 lotes); a hematita porosa de 2025 foi medida de outra forma e ficou fora.
3. Variáveis físicas clássicas do TML ausentes: superfície específica (Blaine), distribuição granulométrica completa (d50, % < 10 µm), teor de argila, método do ensaio (Proctor-Fagerberg ou mesa de fluxo) e FMP.
4. Origem da amostra como preditor: conhecida no momento da previsão, mas não é propriedade do minério e pode mudar se a amostragem mudar.
5. Autocorrelação temporal do TML (ACF1 0,46) com viés que troca de sinal por período; o modelo não captura deriva de processo.
6. Umidade de embarque acima do TML em 18 dos 36 lotes embarcados (Etapa 0), não esclarecida; não afeta o modelo, mas afeta a leitura operacional do histórico.
7. Linha L35 (Azimuth "ofc") com forte suspeita de cópia, excluída; precisa de verificação no laboratório.

Recomendações de coleta de dados:

1. Criar colunas estruturadas para navio/lote, cliente, origem da amostra (navio, pátio, filtrado, produção, alimentação), produto e dose de desidratante, em vez de texto livre.
2. Registrar sempre PFUF, PM e PFSA, inclusive quando 0, e a composição do PM.
3. Medir granulometria (−325#, d50, fração < 10 µm) e superfície específica (Blaine) em todas as amostras de TML, com o mesmo método.
4. Padronizar a mineralogia em quatro fases para todas as amostras e reportar o método; registrar goethita diretamente.
5. Registrar o método do ensaio de TML, o FMP e a umidade da amostra na chegada ao laboratório, e manter réplicas (T01/T02) identificadas por chave de amostra.
6. Registrar qual TML foi a referência de cada decisão de embarque e a umidade final da carga, para esclarecer os 18 casos com umidade acima do TML.
7. Meta de base: ao menos 150 lotes com as variáveis acima para que modelos não lineares tenham chance de superar o linear com significância. Reavaliar o modelo a cada 20 novos lotes com o mesmo protocolo (scripts desta pasta).

## 10. Arquivos entregues

| Arquivo | Conteúdo |
|---|---|
| `tml/etapa0_auditoria.py`, `tml/ETAPA0_diagnostico.md` | auditoria da base e decisões de limpeza |
| `tml/comum.py` | preparo da base, divisões, métricas, catálogo de 24 modelos com espaços de busca, regra de escolha |
| `tml/etapa12_competicao.py` | validação agrupada repetida, walk-forward, busca aninhada, ranking e testes pareados |
| `tml/etapa3_checks.py` | os 10 testes anti-espúrios, PDP, SHAP, resíduos, domínio |
| `tml/etapa4_incerteza.py` | cross-conformal, regressão quantílica, margens |
| `tml/etapa5_final.py` | teste final, modelo de produção, equação, figuras |
| `tml/extra_subconjunto_navio.py` | análise complementar nos lotes de navio |
| `tml/saidas/ranking.csv`, `ranking_temporal.csv`, `cv_metricas_dobra.csv`, `cv_predicoes.csv` | resultados da competição |
| `tml/saidas/etapa3_*.csv`, `etapa3_detalhes.txt` | resultados dos checks |
| `tml/saidas/etapa4_cobertura.csv`, `etapa4_margem.json` | incerteza |
| `tml/saidas/etapa5_teste_final.csv`, `etapa5_teste_final_lotes.csv` | teste final |
| `tml/saidas/modelo_tml_final.joblib`, `modelo_tml_final.json`, `equacao.txt` | pipeline salvo e equação |
| `tml/saidas/fig_previsto_real.png`, `fig_residuos.png`, `fig_ranking.png`, `fig_shap.png`, `fig_pdp.png` | figuras |

Reprodução: `cd tml && python3 etapa0_auditoria.py && python3 etapa12_competicao.py && python3 etapa3_checks.py && python3 etapa4_incerteza.py && python3 etapa5_final.py` (Python 3.13, scikit-learn 1.9, xgboost 3.4, lightgbm 4.7, catboost 1.2, pygam 0.12, shap 0.54, statsmodels 0.15).
