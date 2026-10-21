## Resumo — Aula 09

A Aula 8 estabeleceu o algoritmo de Descida do Gradiente e provou sua dinâmica de convergência em modelos lineares nos parâmetros e perdas convexas (como a Regressão Logística). No entanto, para capturar interações complexas no mundo real — como no problema-fio do California Housing —, os modelos preditivos precisam abandonar a linearidade através de representações hierárquicas profundas: os **Perceptrons Multicamadas** (*Multilayer Perceptrons* - MLPs). Esta aula unifica a fundamentação matemática e o mecanismo computacional que viabilizam o treinamento de modelos profundos: a **Regra da Cadeia Multivariada**, os **Grafos de Computação (DAGs)** e a **Diferenciação Automática** (*Automatic Differentiation* - Autodiff). 

Iniciamos pela formulação matricial de redes neurais densas e deduzimos analiticamente, via Jacobianos em *denominator layout*, as clássicas equações de sensibilidade retrógrada $\boldsymbol{\delta}^{(l)}$ e gradiente matricial dos pesos $\nabla_{W^{(l)}}\mathcal{L} = \boldsymbol{\delta}^{(l)}(\mathbf{a}^{(l-1)})^T$. Em seguida, diagnosticamos o colapso da diferenciação manual e analítica diante de arquiteturas modernas com ramificações e controle de fluxo, comparando rigorosamente os quatro paradigmas de diferenciação (manual, numérica por diferenças finitas, simbólica com seu inchaço exponencial de termos, e automática). Formalizamos programas como Grafos Acíclicos Direcionados (DAGs) e deduzimos os modos **Forward** (tangente / *pushforward*) e **Reverse** (adjunto / *pullback*), demonstrando o célebre **Teorema de Baur-Strassen**, que assegura que o gradiente completo de qualquer perda escalar é computado em tempo linear compatível com a passagem direta. Construímos do zero em Python um mini-motor de autodiff reverso com ordenação topológica por busca em profundidade, treinamos a rede neural no California Housing e analisamos o compromisso tempo-memória da fita de execução (*tape*) e o princípio de *activation checkpointing*.

**Objetivos:**
1. Formular algebricamente um MLP e deduzir analiticamente as equações da retropropagação via Regra da Cadeia multivariada e Jacobianos.
2. Comparar os 4 paradigmas de diferenciação computacional, demonstrando as limitações de diferenças finitas e diferenciação simbólica.
3. Formalizar programas como DAGs e demonstrar a superioridade assintótica do modo reverso via Teorema de Baur-Strassen para perdas escalares ($d_{\text{out}} = 1 \ll d_{\text{in}}$).
4. Implementar do zero em Python um mini-motor de diferenciação automática reversa orientado a objetos com ordenação topológica (DFS).
5. Analisar o custo de memória das ativações armazenadas na fita de execução e o conceito de *activation checkpointing*.

**Pré-requisitos:** Derivadas parciais, vetor gradiente e Jacobiano em *denominator layout* (Aula 6); Descida do Gradiente e Lema da Descida (Aula 8); operações matriciais e produto diádico (Aulas 1 e 2).
**Dataset-fio:** California Housing ($N = 20.640, d = 4$), empregando uma arquitetura de rede neural de 2 camadas ($4 \to 3 \to 1$) para prever `MedHouseVal`, comparando gradientes analíticos, autodiff próprio e diferenças finitas.
**Estratégia Pedagógica:** B (Inside-Out com Problema-Fio) — da necessidade de calcular gradientes em redes neurais não-lineares até a construção do motor de autodiff reverso que sustenta os frameworks modernos de IA.

---

## Plano de aula — Aula 09 (carga horária: ~110 min)

1. **Revisão e Introdução** (~12 min)
   - *Revisão:* Da Aula 8 (Gradient Descent em modelos lineares/convexos) para o desafio de funções compostas hierárquicas.
   - *Ideia Central:* O cálculo de derivadas de modelos profundos não requer soluções simbólicas fechadas; expressar o modelo como uma rede de operações atômicas permite propagar derivadas exatas em tempo linear.
   - *Problema Motivador:* Adicionar uma camada oculta não-linear ao California Housing ($4 \to 3 \to 1$). Como calcular a sensibilidade de parâmetros ocultos que não tocam a perda diretamente?
   - *Roteiro (4 Perguntas Centrais):*
     1. Como a Regra da Cadeia multivariada calcula analiticamente os gradientes de uma rede multicamadas?
     2. Por que a derivação manual e simbólica colapsa em modelos complexos e como a Diferenciação Automática resolve isso?
     3. Qual a diferença entre os modos Forward e Reverse e por que o modo reverso domina o Machine Learning?
     1. Como a Regra da Cadeia explora o reuso de computações no Backpropagation analítico através das 3 equações canônicas?
     2. Por que os paradigmas manual, numérico e simbólico falham em modelos profundos, e como a Diferenciação Automática resolve isso via DAGs e variáveis intermediárias?
     3. Como a regra da cadeia multivariada se propaga em grafos com bifurcações através da soma sobre múltiplos caminhos direcionados?
     4. Qual a dualidade matemática entre os modos Forward e Reverse pela associatividade da regra da cadeia, e por que o Teorema de Baur-Strassen torna o modo reverso o alicerce do Machine Learning?
   - *Pausa Ativa 1.*

2. **Intuição — A Correnteza e o Eco** (~8 min)
   - A passagem direta (*forward pass*) como a água descendo a correnteza das operações elementares.
   - A passagem reversa (*backward pass*) como um eco de sensibilidade gerado na foz (a perda escalar) e propagado rio acima até as nascentes (os pesos e vieses).
   - A modularidade atômica: cada nó só precisa saber derivar sua operação local.

3. **Bloco 1: A Regra da Cadeia Multivariada e Redes Neurais (O Cálculo Analítico)** (~25 min)
   - Composição de funções multivariadas e produto de Jacobianos sob *denominator layout*: $\nabla_{\mathbf{x}}(g \circ \mathbf{f}) = J_{\mathbf{f}}(\mathbf{x})\,\nabla_{\mathbf{u}}g$.
   - Reuso de computações (*prefix caching*) e caixas de parâmetros ($A_i, \mathbf{b}_i$).
   - As 3 Equações Canônicas do Backpropagation:
     - 1. Sensibilidade da Saída: $\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}^{(L)}}\mathcal{L} \odot \sigma'(\mathbf{z}^{(L)})$.
     - 2. Propagação Retrógrada: $\boldsymbol{\delta}^{(l)} = ((W^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}) \odot \sigma'(\mathbf{z}^{(l)})$.
     - 3. Gradientes dos Parâmetros: $\nabla_{W^{(l)}}\mathcal{L} = \boldsymbol{\delta}^{(l)}(\mathbf{a}^{(l-1)})^T$ e $\nabla_{\mathbf{b}^{(l)}}\mathcal{L} = \boldsymbol{\delta}^{(l)}$.
   - Validação empírica no California Housing com arquitetura $4 \to 3 \to 1$.
   - *Transição:* O cálculo matricial funciona para redes sequenciais em camadas, mas como estender a diferenciação exata para programas computacionais com topologia arbitrária e código não-linear?
   - *Pausa Ativa 2.*

4. **Bloco 2: Paradigmas de Diferenciação e Grafos de Computação** (~25 min)
   - Análise comparativa dos 4 paradigmas:
     - Manual (sujeito a erros humanos e inflexível);
     - Numérico por Diferenças Finitas (dilema entre erro de truncamento $\mathcal{O}(h)$ e cancelamento subtrativo catastrófico $\mathcal{O}(\epsilon_{\text{mach}}/h)$, ilustrado pelo gráfico de curva em "V");
     - Simbólico (explosão exponencial de termos — *expression swell* $\mathcal{O}(2^d)$ por duplicação recursiva de subexpressões);
     - Automático (exato em precisão de máquina, linear em tempo, sem aproximação numérica e sem expansão simbólica).
   - O Catálogo de Operações Atômicas e suas Derivadas Simbólicas Pré-Programadas:
     - Tabela de regras primitivas (adição, subtração, multiplicação, potência, exp, log, trigonométricas, ativações).
     - **O papel singular da Regra da Adição:** como $\frac{\partial (x+y)}{\partial x} = 1$ e $\frac{\partial (x+y)}{\partial y} = 1$, a adição atua no backward pass como um *roteador neutro de fluxo*, replicando o gradiente recebido $\bar{z}$ integralmente para ambas as entradas ($\bar{x} = \bar{z}, \bar{y} = \bar{z}$).
   - Decomposição de programas em Variáveis Intermediárias e Grafos Acíclicos Direcionados (DAGs):
     - Aplicação na função motivadora $f(x) = \sqrt{x^2 + \exp(x^2)} + \cos(x^2 + \exp(x^2))$ com variáveis intermediárias $a, b, c, d, e, f$.
     - Diagrama do Grafo de Computação.
   - O Princípio da Acumulação em Bifurcações:
     - Regra da cadeia reversa: a sensibilidade total de um nó é a soma de todos os caminhos direcionados incidentes:
       $$\bar{x}_i := \frac{\partial f}{\partial x_i} = \sum_{x_j \in \text{Filhos}(x_i)} \bar{x}_j \, \frac{\partial x_j}{\partial x_i}$$
     - Demonstração detalhada nos nós bifurcados $c$, $a$ e na entrada $x$.
   - *Transição:* Sabemos como avaliar a regra da cadeia em um DAG usando regras locais; mas em qual direção devemos percorrê-lo: da entrada para a saída ou da saída para a entrada?
   - *Pausa Ativa 3.*

5. **Bloco 3: Modos de Diferenciação Automática: Forward vs. Reverse** (~25 min)
   - A dualidade Forward vs. Reverse sob a ótica da **Associatividade da Multiplicação Matricial**:
     $$J = J_K \cdot J_{K-1} \cdots J_2 \cdot J_1$$
   - Modo Forward (Tangente / Pushforward / JVP):
     - Agrupamento à direita: $J \mathbf{v} = J_K (\dots (J_1 \mathbf{v}))$.
     - Operações atômicas Matriz $\times$ Vetor (*Jacobian-Vector Product* - JVP).
     - Propaga perturbações direcionais $\dot{x}_i$ para a frente.
     - Passo a passo no DAG canônico ($a, b, c, d, e, f$): acumulação de somas nas convergências de entradas (pais).
     - Custo total: $\mathcal{O}(d_{\text{in}} \cdot C_{\text{eval}})$.
   - Modo Reverse (Adjunto / Pullback / VJP / Co-vetores):
     - Agrupamento à esquerda: $\mathbf{u}^T J = (((\mathbf{u}^T J_K) \dots ) J_1)$.
     - Operações atômicas Vetor $\times$ Matriz (*Vector-Jacobian Product* - VJP).
     - Propaga sensibilidades acumuladas (co-vetores) $\bar{x}_i$ para trás.
     - Custo total para perda escalar ($d_{\text{out}} = 1$): $\mathcal{O}(1 \cdot C_{\text{eval}})$.
   - O célebre **Teorema de Baur-Strassen / Speelpenning (1983)**:
     - Para qualquer função escalar $f: \mathbb{R}^n \to \mathbb{R}$, $C_{\text{grad}} \le 5 \cdot C_{\text{eval}}$, independentemente de $n$.
     - Gráfico log-log de escalabilidade assintótica (Forward $\mathcal{O}(d \cdot C_{\text{eval}})$ vs Reverse $\le 5 C_{\text{eval}}$).
   - Análise Comparativa e Decisões Arquiteturais em Machine Learning:
     - $d_{\text{in}} \gg d_{\text{out}}$ (Otimização supervisionada, Redes Neurais, LLMs) $\to$ Reverse Mode é indispensável.
     - $d_{\text{in}} \ll d_{\text{out}}$ (Modelos generativos a partir de espaço latente reduzido, robótica) $\to$ Forward Mode é superior.
     - O Backpropagation clássico como caso particular do Modo Reverse em camadas.
   - O Trade-off Oculto de Memória:
     - Modo Forward descarta variáveis intermediárias imediatamente ($\mathcal{O}(1)$ na profundidade).
     - Modo Reverse congela as ativações intermediárias na fita de execução para alimentar as derivadas locais no retorno ($\mathcal{O}(L)$ na profundidade).
   - *Transição:* Temos a máquina matemática universal para diferenciar modelos em grafos computacionais; mas como treinar quando o dataset tem milhões de amostras e avaliar o grafo completo a cada iteração é inviável?
   - *Pausa Ativa 4.*

6. **Fechamento** (~7 min)
   - *Retomando as 4 Perguntas:* Síntese concisa das respostas construídas na aula.
   - *Fechamento do Bloco 2:* O cálculo diferencial para machine learning está formalizado e unificado.
   - *Ponte para o Bloco 3:* A necessidade de estocasticidade e mini-batches diante de Big Data $\to$ **Aula 10: Gradiente Descendente Estocástico (SGD) e Mini-Batching**. (E mais adiante: modelos de curvatura e aproximações quadráticas de Taylor para aceleradores de Momento e Métodos de Newton).
   - Links para Exercícios e Soluções.

---

## Fontes usadas — Aula 09

Deslocamentos página impressa → PDF: `mathml.pdf` +6; `copt.pdf` +14; `optml.pdf`.

### Fonte 1: Deisenroth, Faisal & Ong, *Mathematics for Machine Learning*, §5.2.2 e §5.3, pp. 149–153 (PDF pp. 155–159)
**Uso pretendido:** Regra da cadeia multivariada, Jacobiano de funções compostas e representação matricial da diferenciação em cadeia.
**Trecho:**
> "The chain rule can be written as a matrix multiplication: $\frac{df}{d(s, t)} = \frac{\partial f}{\partial \mathbf{x}}\frac{\partial \mathbf{x}}{\partial(s, t)} = \begin{bmatrix} \frac{\partial f}{\partial x_1} & \frac{\partial f}{\partial x_2} \end{bmatrix} \begin{bmatrix} \frac{\partial x_1}{\partial s} & \frac{\partial x_1}{\partial t} \\ \frac{\partial x_2}{\partial s} & \frac{\partial x_2}{\partial t} \end{bmatrix}$... This compact way of writing the chain rule as a matrix multiplication only makes sense if the gradient is defined as a row vector. Otherwise, we will need to start transposing gradients for the matrix dimensions to match." (§5.2.2, p. 149)

### Fonte 2: Deisenroth, Faisal & Ong, §5.6.1, pp. 160–161 (PDF pp. 166–167)
**Uso pretendido:** Formulação matricial do fluxo feedforward em redes neurais multicamadas, perda quadrática e estrutura de gradientes com propagação para trás.
**Trecho:**
> "In neural networks with multiple layers, we have functions $f_i(\mathbf{x}_{i-1}) = \sigma(A_{i-1}\mathbf{x}_{i-1} + \mathbf{b}_{i-1})$ in the $i$th layer... In order to train these models, we require the gradient of a loss function $L$ with respect to all model parameters $A_j, \mathbf{b}_j$... The chain rule allows us to determine the partial derivatives as $\frac{\partial L}{\partial \theta_i} = \frac{\partial L}{\partial f_K}\frac{\partial f_K}{\partial f_{K-1}}\cdots\frac{\partial f_{i+1}}{\partial \theta_i}$... gradients are passed backward through the network." (§5.6.1, pp. 160–161)

### Fonte 3: Deisenroth, Faisal & Ong, §5.6.2, pp. 161–164 (PDF pp. 167–170)
**Uso pretendido:** Definição formal de Diferenciação Automática, grafos de computação, distinção entre modos forward e reverse, e equações de acumulação de gradientes em nós dependentes.
**Trechos:**
> "Automatic differentiation is a formalization of Example 5.14. Let $x_1, \dots, x_d$ be the input variables to the function, $x_{d+1}, \dots, x_{D-1}$ be the intermediate variables, and $x_D$ the output variable. Then the computation graph can be expressed as follows: For $i = d + 1, \dots, D: x_i = g_i(x_{\text{Pa}(x_i)})$, where the $g_i(\cdot)$ are elementary functions and $x_{\text{Pa}(x_i)}$ are the parent nodes of the variable $x_i$ in the graph... For other variables $x_i$, we apply the chain rule:
> $$\frac{\partial f}{\partial x_i} = \sum_{x_j: x_i \in \text{Pa}(x_j)} \frac{\partial f}{\partial x_j}\frac{\partial x_j}{\partial x_i} = \sum_{x_j: x_i \in \text{Pa}(x_j)} \frac{\partial f}{\partial x_j}\frac{\partial g_j}{\partial x_i}$$" (§5.6.2, pp. 163–164)
> "Due to the associativity of matrix multiplication, we can choose between $\frac{dy}{dx} = \left(\frac{dy}{db}\frac{db}{da}\right)\frac{da}{dx}$ (reverse mode) and $\frac{dy}{dx} = \frac{dy}{db}\left(\frac{db}{da}\frac{da}{dx}\right)$ (forward mode)... In the context of neural networks, where the input dimensionality is often much higher than the dimensionality of the labels, the reverse mode is computationally significantly cheaper than the forward mode." (§5.6.2, p. 161)

### Fonte 4: Griewank & Walther, *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, SIAM (2008); e Baur & Strassen (1983)
**Uso pretendido:** Teorema de Baur-Strassen / Speelpenning sobre complexidade temporal assintótica do modo reverso.
**Trecho:**
> "Theorem (Baur-Strassen, 1983). Let $f: \mathbb{R}^n \to \mathbb{R}$ be computed by a straight-line arithmetic circuit of length $L$ using operations $\{+, -, \times, /\}$ and unit-cost elementary functions. Then the gradient $\nabla f(x) \in \mathbb{R}^n$ can be computed by an arithmetic circuit of length at most $5L$."

### Fonte 5: Baydin et al., *Automatic Differentiation in Machine Learning: A Survey*, JMLR (2018), §2–§3
**Uso pretendido:** Comparação sistemática dos quatro métodos de diferenciação e demonstração do inchaço de expressões (*expression swell*) na diferenciação simbólica.
**Trecho:**
> "Automatic differentiation (AD) executes a program that computes the analytical derivative of another program... Symbolic differentiation suffers from the problem of 'expression swell', where expressions grow exponentially with nesting depth unless complex algebraic simplification algorithms are applied at each step."
