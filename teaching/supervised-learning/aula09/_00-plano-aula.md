## Resumo — Aula 9: Teoria de Ensembles e Redução de Variância (Bagging e Random Forests)

Esta aula inaugura a Parte 3 do curso (Modelos Avançados e Otimização), operacionalizando a decomposição viés-variância estabelecida na Aula 8 para resolver a instabilidade de preditores de alta capacidade, em especial as árvores de decisão (CART) da Aula 3. Estudamos a teoria estatística do agregamento por bootstrap (Bagging) e a derivação exata da variância da média de variáveis aleatórias identicamente distribuídas e correlacionadas ($\rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$). Demonstramos por que o Bagging tem como gargalo a correlação positiva $\rho$ induzida pela dependência dos dados de treino, e como o algoritmo de Florestas Aleatórias (Breiman, 2001) quebra deliberadamente essa correlação ao restringir o particionamento a um subconjunto aleatório de atributos ($m \le p$). Por fim, formalizamos a avaliação Out-of-Bag (OOB) como consequência do limite assintótico $(1 - 1/N)^N \to 1/e \approx 0{,}368$ e contrastamos a importância de variáveis por impureza (MDI) e por permutação (MDA). O dataset-fio é o California Housing (`gvlassis/california_housing`), conectando com os resultados de árvores da Aula 3 e a regressão da Aula 5.

**Estratégia Pedagógica:** A (Outside-In) — Partimos da intuição prática da média de preditores instáveis para mitigar o overfitting, formalizamos a mecânica matemática da variância correlacionada e da amostragem OOB, e sintetizamos com as propriedades e limites práticos de Random Forests.

---

## Plano de aula — Aula 9 (carga horária: 120 min)

1. **Revisão e Introdução** (~15 min)
   - *Revisão:* A decomposição formal do EMSE da Aula 8 ($\text{EMSE} = \text{Bias}^2 + \text{Var} + \sigma^2$), a instabilidade estrutural de árvores CART não podadas da Aula 3 (baixo viés, variância catastrófica), e o conceito de reamostragem bootstrap da Aula 4.
   - *Ideia Central:* Se um estimador não possui viés significativo mas flutua violentamente entre amostras, a média de muitas realizações estocásticas reduz a variância sem degradar o viés.
   - *Roteiro:* 5 perguntas fundamentais da aula (por que a média reduz variância mas não viés; qual o limite matemático do Bagging puro; como a seleção aleatória de variáveis em Random Forests quebra a correlação $\rho$; por que cerca de $36{,}8\%$ dos dados não participam de cada árvore e como usá-los para validação OOB sem custo extra; e como interpretar a importância de variáveis por impureza vs. permutação).
   - *Problema Motivador:* Uma única árvore profunda de regressão no California Housing sofre de sobreajuste severo (gap substancial entre treino e teste), enquanto a média simples de 100 árvores bootstrapped reduz o erro de generalização imediatamente.
   - *Pausa Ativa 1:* A mecânica da média e a preservação do viés.

2. **Intuição — A Sabedoria das Multidões Correlacionadas** (~15 min)
   - Analogia do júri / comitê: votos independentes convergem para a verdade pelo Teorema Central do Limite e Lei dos Grandes Números; votos altamente correlacionados repetem o mesmo erro sistemático.
   - Por que estimadores lineares estáveis (como média amostral ou OLS) quase nada ganham com Bagging ($\rho \approx 0{,}5$), enquanto árvores de decisão ganham dramaticamente devido à sua natureza descontínua e sensível a pequenas permutações nos nós superiores.
   - Diagrama TikZ contrastando o processo de particionamento estocástico e a suavização das fronteiras em degrau.
   - *Pausa Ativa 2:* Estabilidade de estimadores sob reamostragem bootstrap.

3. **Bloco 1: Bagging e a Teoria Estatística da Média de Estimadores** (~25 min)
   - Definição formal do estimador Bagging: $\hat{f}_{\text{bag}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(\mathbf{x})$.
   - Demonstração rigorosa de que o viés não se reduz: $\mathbb{E}[\hat{f}_{\text{bag}}(\mathbf{x})] = \mathbb{E}[\hat{f}^*(\mathbf{x})]$, sendo idêntico (ou ligeiramente superior) ao da árvore individual.
   - Demonstração passo a passo da fórmula da variância de variáveis aleatórias com variância $\sigma^2$ e correlação par-a-par positiva $\rho$:
     $$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B T_b\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
   - Análise do comportamento assintótico quando $B \to \infty$: o termo $\frac{1-\rho}{B}\sigma^2 \to 0$, mas o termo $\rho \sigma^2$ permanece inalterado.
   - Diagnóstico: $\rho$ é a barreira intransponível do Bagging convencional (árvores dividem os mesmos preditores dominantes nos nós superiores).
   - *Pausa Ativa 3:* O teto assintótico da variância em função de $\rho$ e $B$.

4. **Bloco 2: Florestas Aleatórias (Random Forests) e a Quebra da Correlação** (~25 min)
   - O algoritmo de Breiman (2001): introdução de aleatoriedade dupla (amostras bootstrap de observações + seleção aleatória de $m \le p$ variáveis em cada divisão).
   - Regras práticas para a escolha de $m$: $m = \lfloor \sqrt{p} \rfloor$ para classificação e $m = \lfloor p/3 \rfloor$ para regressão; tamanho mínimo de nó ($n_{\min}=1$ para classificação, $n_{\min}=5$ para regressão).
   - O mecanismo analítico de descorrelação: forçar atributos secundários a liderarem divisões nos nós superiores derruba drasticamente $\rho$, sem elevar excessivamente a variância individual $\sigma^2$.
   - O tradeoff viés-variância na escolha de $m$: valores muito pequenos de $m$ reduzem $\rho$ mas elevam o viés individual de cada árvore.
   - Visualização do decaimento de $\rho$ e do erro quadrático médio em função de $m$ no California Housing.
   - *Pausa Ativa 4:* A dinâmica do hiperparâmetro $m$ e a supressão de atributos dominantes.

5. **Bloco 3: Avaliação Out-of-Bag (OOB) e Importância de Atributos** (~25 min)
   - Dedução formal da probabilidade de uma observação não ser sorteada em uma amostra bootstrap de tamanho $N$:
     $$P(i \notin Z^*) = \left(1 - \frac{1}{N}\right)^N \xrightarrow{N \to \infty} e^{-1} \approx 0{,}367879... \approx 36{,}8\%$$
   - Construção do preditor OOB para cada ponto $\mathbf{z}_i$: agregação exclusiva das árvores em cujo conjunto bootstrap $\mathbf{z}_i$ não participou.
   - Teorema/propriedade: a estimativa de erro OOB converge para o erro obtido por validação cruzada leave-one-out sem custo computacional adicional.
   - Mensuração da importância de preditores:
     - *Mean Decrease Impurity (MDI)*: acúmulo da redução de Gini/MSE nas divisões atribuídas à variável $j$ ao longo de todas as árvores (viés em favor de variáveis com muitas categorias ou alta cardinalidade).
     - *Mean Decrease Accuracy (MDA / Permutation Importance)*: embaralhamento aleatório dos valores da variável $j$ exclusivamente nas amostras OOB e medição do salto no erro de predição.
   - *Pausa Ativa 5:* Validação OOB e armadilhas do MDI vs. MDA.

6. **Fechamento** (~15 min)
   - *Retomando as 5 Perguntas da Aula:* Síntese objetiva das 5 respostas técnicas.
   - *O Que Fica em Aberto:* Se Random Forests reduzem a variância de modelos complexos por paralelismo e independência estocástica, o que fazer quando o modelo base é simples demais (alto viés) ou quando os dados possuem forte estrutura aditiva sequencial?
   - *Ponte para a Aula 10:* Gradient Boosting e Otimização Numérica no Espaço de Funções. Ao contrário do Bagging (que projeta paralelamente modelos de baixo viés para reduzir variância), o Boosting parte de modelos muito simples (*decision stumps* de alto viés) e adiciona árvores sequencialmente ao longo da direção do gradiente negativo da função de perda.
   - Links para `exercicios.qmd` e `soluções.qmd`.

---

## Fontes usadas — Aula 9

### Fonte 1: ESL (Hastie, Tibshirani & Friedman, 2009), §8.7, pp. 282–283
**Uso pretendido:** Definição formal do Bagging como média de predições sobre réplicas bootstrap e conexão com a redução de variância.
**Trecho:**
> "Consider first the regression problem. Suppose we fit a model to our training data $Z = \{(x_1, y_1), (x_2, y_2), \dots, (x_N, y_N)\}$, obtaining the prediction $\hat{f}(x)$ at input $x$. Bootstrap aggregation or bagging averages this prediction over a collection of bootstrap samples, thereby reducing its variance. For each bootstrap sample $Z^{*b}$, $b = 1, 2, \dots, B$, we fit our model, giving prediction $\hat{f}^{*b}(x)$. The bagging estimate is defined by $\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B \hat{f}^{*b}(x)$."

### Fonte 2: ESL (Hastie, Tibshirani & Friedman, 2009), §15.2, pp. 587–589
**Uso pretendido:** Derivação da fórmula da variância da média de variáveis aleatórias identicamente distribuídas com correlação positiva $\rho$, e introdução de Random Forests como mecanismo de descorrelação via subamostragem aleatória de atributos.
**Trecho:**
> "An average of $B$ i.i.d. random variables, each with variance $\sigma^2$, has variance $\frac{1}{B}\sigma^2$. If the variables are simply i.d. (identically distributed, but not necessarily independent) with positive pairwise correlation $\rho$, the variance of the average is
> $$\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2.$$
> As $B$ increases, the second term disappears, but the first remains, and hence the size of the correlation of pairs of bagged trees limits the benefits of averaging. The idea in random forests (Algorithm 15.1) is to improve the variance reduction of bagging by reducing the correlation between the trees, without increasing the variance too much. This is achieved in the tree-growing process through random selection of the input variables."

### Fonte 3: Breiman, Leo (2001). "Random Forests", *Machine Learning*, 45(1), pp. 5–7
**Uso pretendido:** Definição original do algoritmo de Random Forests, convergência quase-certa pelo Teorema dos Grandes Números e ausência de overfitting assintótico com o crescimento de $B$.
**Trecho:**
> "A random forest is a classifier consisting of a collection of tree-structured classifiers $\{h(\mathbf{x}, \Theta_k), k=1, \dots\}$ where the $\{\Theta_k\}$ are independent identically distributed random vectors and each tree casts a unit vote for the most popular class at input $\mathbf{x}$... As the number of trees increases, for almost all sequences $\Theta_1, \dots$ the generalization error $PE^*$ converges almost surely to
> $$P_{\mathbf{x}, y}\left(P_\Theta(h(\mathbf{x}, \Theta) = y) - \max_{j \neq y} P_\Theta(h(\mathbf{x}, \Theta) = j) < 0\right).$$
> This result explains why random forests do not overfit as more trees are added, but produce a limiting value of the generalization error."

### Fonte 4: ESL (Hastie, Tibshirani & Friedman, 2009), §15.3.1–15.3.2, pp. 592–594
**Uso pretendido:** Mecânica exata das amostras Out-of-Bag (OOB) e distinção matemática entre importância de variáveis por impureza (MDI) e por permutação aleatória (MDA).
**Trecho:**
> "For each observation $z_i = (x_i, y_i)$, construct its random forest predictor by averaging only those trees corresponding to bootstrap samples in which $z_i$ did not appear. An oob error estimate is almost identical to that obtained by $N$-fold cross-validation... Random forests also use the oob samples to construct a different variable-importance measure, apparently to measure the prediction strength of each variable. When the $b$th tree is grown, the oob samples are passed down the tree, and the prediction accuracy is recorded. Then the values for the $j$th variable are randomly permuted in the oob samples, and the accuracy is again computed. The decrease in accuracy as a result of this permuting is averaged over all trees, and is used as a measure of the importance of variable $j$."

