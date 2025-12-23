# House Prices — Trabalho Final ML 1 (Turma 1486 - AdaTech)

## Visão geral
Este projeto aborda um problema de regressão: prever o preço de venda de imóveis (`SalePrice`) a partir de características numéricas e categóricas do imóvel. O foco foi construir uma pipeline reprodutível, evitar vazamento de dados, comparar modelos e justificar a escolha final com validação cruzada.

## Aluno 
Thiago da Silva Mendonça 
## Professor 
Maurício Luiz Sobrinho 

---

## Dataset
Base: House Prices (Ames Housing).  
Alvo: `SalePrice`.

### Observações do EDA
- O `SalePrice` apresentou **forte assimetria à direita** e **outliers** (cauda longa até ~755k).
  - Isso é consistente com **média (~180,9k) maior que a mediana (~163k)** e boxplot com muitos extremos.
  - Para estabilizar o treinamento e reduzir influência de valores muito altos, foi testada **transformação log no alvo** via `TransformedTargetRegressor` (`log1p`/`expm1`).

- Foram identificadas **19 colunas com valores ausentes (NaNs)**.
  - No House Prices, muitos NaNs representam **ausência real** do atributo (ex.: `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, colunas de garagem/porão), e não necessariamente erro de coleta.
  - Para manter o pipeline simples e reprodutível, foi aplicada imputação automática:
    - Numéricas: **mediana**
    - Categóricas: **valor mais frequente**
  - Em seguida, categóricas foram codificadas com **One-Hot Encoding**.

- Maiores correlações numéricas com `SalePrice`:
  - `OverallQual` (~0.79)
  - `GrLivArea` (~0.71)
  - `GarageCars` / `GarageArea` (~0.64 / ~0.62)
  - `TotalBsmtSF`, `1stFlrSF` (áreas de porão/primeiro piso)
- O heatmap evidenciou **colinearidade** entre variáveis relacionadas (ex.: `GarageCars` vs `GarageArea`, `YearBuilt` vs `GarageYrBlt`), reforçando o uso de:
  - regularização em modelos lineares
  - modelos baseados em árvores/boosting para capturar interações

---

## Pipeline de Pré-processamento
Foi construída uma pipeline com `ColumnTransformer`, mantendo todo o pré-processamento **dentro da Pipeline** para evitar vazamento:

- **Numéricas**: `SimpleImputer(strategy="median")` + `StandardScaler()`
- **Categóricas**: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

Além disso, foi testada a transformação do alvo com:
- `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`

---

## Modelos testados
Famílias comparadas:
- Ridge (baseline linear regularizado)
- KNN
- SVR
- RandomForest
- GradientBoosting

---

## Estratégia de avaliação
- **Holdout (80/20)** para avaliação rápida (treina em `X_train`, valida em `X_valid`)
- **Cross-Validation (5-fold)** no conjunto de treino para decisão mais robusta

---

## Resultados e decisão
- No **holdout**, `Ridge_log` apresentou melhor RMSE (~23.9k) e maior R² (~0.93), mas por ser uma única divisão, isso pode depender do split.
- No **cross-validation (5-fold)**, o Ridge apresentou **erro médio maior e alta variância**, indicando instabilidade.
- O **GradientBoosting** apresentou **melhor RMSE médio e desempenho mais consistente** em CV, sendo escolhido como **modelo final**.

### Fine Tuning (extra)
- Foi realizado fine tuning no **RandomForest** com `RandomizedSearchCV`, por ser um modelo robusto e sensível a hiperparâmetros de complexidade.
- O tuning reduziu o RMSE médio do RandomForest em CV, porém o **GradientBoosting permaneceu como melhor desempenho global**.

---

## Explicabilidade (extra)
Foi utilizada **Permutation Importance** no conjunto de validação (subset de 300 amostras).
- A técnica mede quanto a performance do modelo piora ao embaralhar cada feature, estimando a dependência do modelo em cada variável.
- Entre as principais features apareceram:
  - Banheiros: `BsmtFullBath`, `FullBath`
  - Lote/terreno: `LotShape_IR1`, `LandSlope`
  - Zoneamento: `MSZoning_FV`
  - Qualidade/estrutura: `OverallQual`, `GarageArea`
- Observação: importâncias são dependentes do modelo e podem ser afetadas por correlação/redundância entre variáveis.

---

## Como executar
1) Instale as dependências:
```bash
pip install -r requirements.txt
