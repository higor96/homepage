# Respostas das Pausas Ativas — Aula 9: Teoria de Ensembles e Redução de Variância

Este documento contém o gabarito detalhado, as discussões das perguntas abertas e as diretrizes pedagógicas para a condução das 5 Pausas Ativas da Aula 9.

---

## Pausa Ativa 1: A Mecânica da Média e o Comportamento do Viés

### Pergunta Aberta
**Cenário:** Um cientista de dados treina uma regressão linear simples $y = \beta_0 + \beta_1 x$ para aproximar uma função fortemente senoidal e observa um erro de treino elevado e um erro de validação igualmente elevado ($\text{MSE} \approx 0{,}85$). O pesquisador então sugere: *"Vamos aplicar Bagging com $B = 500$ réplicas bootstrap para resolver o problema."* Qual é o erro fundamental dessa estratégia?

**Discussão e Condução Pedagógica:**
- O erro fundamental reside em confundir o papel da variância com o papel do viés.
- A regressão linear simples $y = \beta_0 + \beta_1 x$ sofre de **alto viés** (erro de especificação estrutural): retas não possuem flexibilidade matemática para descrever oscilações periódicas senoidais.
- A esperança matemática da média de estimadores é igual à esperança do estimador base:
  $$\mathbb{E}[\hat{f}_{\text{bag}}(\mathbf{x})] = \mathbb{E}\left[\frac{1}{B}\sum_{b=1}^B \hat{f}^{*b}(\mathbf{x})\right] = \mathbb{E}[\hat{f}^*(\mathbf{x})]$$
- Como a média de 500 retas continua sendo uma reta, o erro sistemático estrutural de $0{,}85$ permanecerá intacto.
- O Bagging só é eficaz quando aplicado a estimadores que já possuem **baixo viés**, mas sofrem de **alta variância** (como árvores CART não podadas ou redes neurais profundas). Para combater o viés de modelos simples, a família adequada é o **Boosting** (Aula 10).

### Teste V/F
- ✔ **V:** Pela linearidade da esperança, $\mathbb{E}\left[\frac{1}{B}\sum T_b\right] = \frac{1}{B}\sum \mathbb{E}[T_b] = \mathbb{E}[T]$, preservando o viés inalterado.
- ✗ **F:** O Bagging atua exclusivamente na variância; modelos com alto viés permanecem enviesados após a média.
- ✔ **V:** Árvores profundas têm alta flexibilidade não paramétrica para criar partições locais refinadas, mantendo baixo viés.
- ✗ **F:** Conforme provado por Breiman (2001, Teorema 1.2), o erro de generalização de ensembles por agregação converge assintoticamente para um limite finito positivo quando $B \to \infty$, não sofrendo de overfitting pelo simples acréscimo de árvores.

---

## Pausa Ativa 2: Perturbação de Dados e Estabilidade de Estimadores

### Pergunta Aberta
**Cenário:** Considere dois algoritmos supervisionados: um classificador vizinhos mais próximos com $k = 1$ ($1$-NN) e um classificador linear por Análise Discriminante Linear (LDA) com matriz de covariância compartilhada. Em qual dos dois algoritmos você espera que o Bagging produza a redução de erro de generalização mais expressiva?

**Discussão e Condução Pedagógica:**
- O ganho do Bagging é uma função direta da **instabilidade** do estimador base perante perturbações na amostra de treinamento.
- O classificador $1$-NN é o exemplo canônico de estimador instável e descontínuo: suas fronteiras de decisão são polígonos de Voronoi determinados rigidamente pela posição de pontos individuais. Retirar ou duplicar pontos via bootstrap altera substancialmente a vizinhança de múltiplos pontos de teste. A média de muitos classificadores $1$-NN produz estimativas suaves de probabilidade a posteriori, resultando em um ganho de acurácia expressivo.
- O LDA, por outro lado, estima médias de classe e uma matriz de covariância global agrupada. Trata-se de um estimador paramétrico global muito estável. O bootstrap altera muito pouco essas médias globais, fazendo com que a correlação entre as réplicas seja altíssima ($\rho \approx 0{,}5$ a $0{,}7$). Portanto, a média de múltiplos LDAs reproduz praticamente o mesmo hiperplano de separação do LDA original.

### Teste V/F
- ✗ **F:** OLS é um estimador linear estável; sua correlação bootstrap é alta ($\approx 0{,}5$) e o Bagging quase não altera os coeficientes finais.
- ✔ **V:** O $1$-NN cria fronteiras de Voronoi extremamente fragmentadas e sensíveis à posição de pontos isolados; a média suaviza essas fronteiras de decisão.
- ✔ **V:** A média de funções em degrau ortogonais geradas por árvores produz superfícies de probabilidade suaves e contínuas no espaço de atributos.
- ✔ **V:** Substituindo $\rho = 1$ na fórmula da variância, $\text{Var}(\bar{T}) = 1 \cdot \sigma^2 + 0 = \sigma^2$, não havendo redução de variância.

---

## Pausa Ativa 3: O Teto Assintótico da Variância e o Número de Árvores

### Pergunta Aberta
**Cenário:** Um engenheiro de infraestrutura nota que treinar $B = 50$ árvores reduziu o MSE de $0{,}60$ para $0{,}32$. Ele propõe alocar mais servidores para rodar $B = 5.000$ árvores na esperança de reduzir o MSE para perto de zero. Avalie se esse investimento se justifica com base na fórmula da variância correlacionada.

**Discussão e Condução Pedagógica:**
- A fórmula da variância correlacionada é $\text{Var}(\bar{T}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$.
- O erro é composto por duas parcelas: a parcela decrescente $\frac{1-\rho}{B}\sigma^2$ e o piso constante $\rho \sigma^2$.
- Com $B = 50$, o termo decrescente já foi reduzido por um fator $50$. A maior parte do ganho possível de variância já foi capturada.
- Ao expandir de $50$ para $5.000$ árvores, o termo decrescente cai de $\frac{1-\rho}{50}\sigma^2$ para $\frac{1-\rho}{5.000}\sigma^2$, mas o piso $\rho \sigma^2$ (que frequentemente representa mais de $70\%$ da variância restante) permanece inalterado.
- Além disso, existe o ruído irredutível da natureza $\sigma^2 > 0$ e o viés do modelo, que jamais convergirão a zero.
- Portanto, o custo computacional cresce linearmente em 100 vezes para um retorno marginal praticamente imperceptível.

### Teste V/F
- ✔ **V:** Se $\rho = 0$, a fórmula se reduz a $\frac{\sigma^2}{B}$, cuja variância assintótica converge estritamente a zero quando $B \to \infty$.
- ✗ **F:** Apenas a parcela $\frac{1-\rho}{B}\sigma^2$ é dividida por 2; a parcela $\rho \sigma^2$ permanece intocada.
- ✔ **V:** Tomando o limite $B \to \infty$, o termo com $B$ no denominador desaparece, restando $\rho \sigma^2$.
- ✗ **F:** O viés independe de $B$ e permanece constante ($\mathbb{E}[\hat{f}_{\text{RF}}] = \mathbb{E}[T]$).

---

## Pausa Ativa 4: A Dinâmica de $m$ e a Supressão de Atributos Dominantes

### Pergunta Aberta
**Cenário:** Imagine uma base de dados médica com $p = 100$ atributos contínuos onde um único biomarcador $X_1$ possui correlação linear extrema com a doença ($r = 0{,}98$), enquanto os outros $99$ atributos possuem correlações sutis ($r \approx 0{,}15$). O que ocorrerá com a estrutura das árvores no Bagging ($m = 100$) comparado a uma Random Forest com $m = 10$?

**Discussão e Condução Pedagógica:**
- **No Bagging ($m = 100$):** Em todas as árvores e em todas as divisões superiores, o algoritmo guloso examina todas as variáveis. Como $X_1$ possui o maior poder de redução de impureza disparado, ele será selecionado no nó raiz de praticamente $100\%$ das árvores. Isso torna as árvores "gêmeas estruturais": todas compartilham a mesma partição inicial. A correlação par-a-par $\rho$ será próxima de $1$, neutralizando o poder do ensemble.
- **Na Random Forest ($m = 10$):** Em cada divisão de cada nó, sorteia-se um subconjunto aleatório de $10$ variáveis dentre as $100$. A probabilidade de $X_1$ **não** estar entre as $10$ variáveis sorteadas no nó raiz é:
  $$P(X_1 \notin \text{amostra}) = \frac{\binom{99}{10}}{\binom{100}{10}} = \frac{90}{100} = 0{,}90 \ (90\%)$$
- Em $90\%$ das árvores, a raiz será forçada a dividir em um dos outros $99$ atributos. Isso quebra drasticamente a semelhança entre as árvores, forçando o comitê a aprender padrões multivariados sutis e derrubando $\rho$ de maneira espetacular.

### Teste V/F
- ✗ **F:** Reduzir $m$ restringe o espaço de busca pelo corte ótimo, o que tende a **aumentar** ligeiramente o viés de cada árvore individual.
- ✔ **V:** Com $m = p$, o atributo dominante vence a disputa de ganho de impureza em quase todas as raízes das árvores bootstrap.
- ✔ **V:** Em $m < p$, há probabilidade $(1 - m/p)$ de o atributo dominante não participar do sorteio no nó, permitindo que outros preditores liderem a divisão.
- ✔ **V:** Com $m = 1$, as divisões são cegas à capacidade preditiva dos atributos, aumentando o viés e a variância individual além do ponto de equilíbrio ótimo.

---

## Pausa Ativa 5: Validação OOB e Armadilhas de MDI vs. MDA

### Pergunta Aberta
**Cenário:** Você adiciona a uma base de dados um atributo espúrio `ID_Cliente` com $2.000$ inteiros aleatórios únicos sorteados uniformemente ($1$ a $2.000$), que não guarda qualquer relação causal com a variável resposta. Ao ajustar uma Random Forest, o gráfico de MDI posiciona o `ID_Cliente` entre os três atributos mais importantes da floresta, enquanto o MDA o posiciona rigorosamente com importância zero. Por que ocorre essa divergência e em qual das duas métricas o engenheiro deve confiar?

**Discussão e Condução Pedagógica:**
- O engenheiro deve confiar com absoluta certeza no **MDA (Permutation Importance)**.
- O **MDI (Mean Decrease Impurity)** é calculado sobre os dados de treinamento. Um atributo numérico com $2.000$ valores únicos para $2.000$ instâncias oferece $1.999$ possíveis limiares de corte. Em nós profundos da árvore, o algoritmo guloso invariavelmente encontra um corte nesse ID que separa perfeitamente um punhado de instâncias de treino, acumulando reduções ilusórias de impureza. Isso é um viés conhecido de alta cardinalidade do MDI.
- O **MDA**, por outro lado, mede a degradação preditiva nas observações **Out-of-Bag (OOB)**, que não foram vistas no treino daquela árvore. Quando os valores de `ID_Cliente` são embaralhados nas amostras OOB, o erro do modelo não se altera, porque o modelo não aprendeu nenhuma relação real que generalize. Portanto, o acréscimo de erro é nulo ($\text{MDA} \approx 0$).

### Teste V/F
- ✔ **V:** Decorre diretamente da dedução matemática do limite $\lim_{N \to \infty} (1 - 1/N)^N = 1/e \approx 36{,}8\%$.
- ✗ **F:** O erro OOB avalia exclusivamente observações que a árvore não utilizou durante o treino, sendo uma estimativa imparcial e não otimista comparável à validação cruzada.
- ✔ **V:** O MDI favorece atributos contínuos e categóricos com muitos níveis distintos devido ao excesso de graus de liberdade de corte nos dados de treino.
- ✔ **V:** O embaralhamento rompe a correlação entre a variável preditora e o alvo preservando a distribuição empírica marginal de $X_j$.

