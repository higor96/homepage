# Gabarito e Guia das Pausas Ativas — Aula 09: Funções Compostas, Redes Neurais e Diferenciação Automática

Este documento contém o gabarito das questões de Verdadeiro/Falso e as diretrizes pedagógicas para condução das discussões das perguntas abertas em cada uma das quatro Pausas Ativas da Aula 09.

---

## Pausa Ativa 1: O Desafio da Profundidade

### Pergunta Aberta
*Por que não podemos simplesmente aplicar o algoritmo de Descida do Gradiente da Aula 8 em um MLP calculando derivadas analíticas individuais para cada um dos parâmetros de forma independente, tratando cada peso como uma função isolada?*

**Diretrizes de Discussão com a Turma:**
- **Inchaço e Redundância Computacional:** Peça aos alunos para imaginarem calcular a derivada da perda em relação a um peso $W^{(1)}_{jk}$ na primeira camada de uma rede profunda. Se tratarmos cada parâmetro como uma função isolada $f_{W_{jk}}(\mathbf{x})$, teríamos que reavaliar todas as ativações das camadas $2, 3, \dots, L$ para cada um dos $d$ pesos. Se a rede possui 1 milhão de parâmetros, seriam necessárias 1 milhão de passagens diretas completas a cada passo de gradiente!
- **Estrutura de Compartilhamento:** O fluxo direto é compartilhado por todos os pesos, e o fluxo reverso compartilha as sensibilidades das camadas superiores. A regra da cadeia multivariada explora a estrutura fatorial de funções compostas, permitindo atualizar todos os parâmetros com o custo de apenas uma passagem direta e uma reversa.

### Julgamento de Verdadeiro ou Falso
- ✗ **F:** Em um MLP com ativações não-lineares, a superfície de perda é fortemente não-convexa, repleta de mínimos locais e pontos de sela; ela não herda as garantias de convexidade da Aula 7.
- ✔ **V:** O produto de transformações afins é fechado: $W^{(2)}(W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) + \mathbf{b}^{(2)} = (W^{(2)}W^{(1)})\mathbf{x} + (W^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)})$. Sem não-linearidades, qualquer rede profunda colapsa a uma regressão linear comum.
- ✗ **F:** Pela regra da cadeia, o sinal de erro de um peso oculto atravessa obrigatoriamente as matrizes de pesos das camadas posteriores até a perda.
- ✔ **V:** Sob $L$-suavidade, o Lema da Descida garante decrescimento da perda até alcançar um ponto estacionário $\nabla \mathcal{L} = \mathbf{0}$, mas não garante mínimo global em superfícies não-convexas.

---

## Pausa Ativa 2: O Mecanismo da Retropropagação

### Pergunta Aberta
*Por que a propagação da sensibilidade de uma camada para a anterior multiplica a sensibilidade pela matriz transposta $(W^{(l+1)})^T$, e por que o gradiente do peso $W^{(l)}$ tem a forma exata de um produto diádico $\boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$?*

**Diretrizes de Discussão com a Turma:**
- **O Operador Adjunto:** Na passagem direta, $W^{(l+1)} \in \mathbb{R}^{d_{l+1} \times d_l}$ mapeia perturbações de ativação do espaço $\mathbb{R}^{d_l}$ para o espaço $\mathbb{R}^{d_{l+1}}$. Na passagem reversa, o sinal de erro $\boldsymbol{\delta}^{(l+1)} \in \mathbb{R}^{d_{l+1}}$ viaja no sentido oposto; o operador linear canônico que transporta formas lineares / gradientes de volta ao espaço de origem é o operador adjunto (a matriz transposta $(W^{(l+1)})^T$).
- **Produto Diádico (Rank-1):** A derivada parcial em relação ao elemento $W^{(l)}_{jk}$ é $\frac{\partial \mathcal{L}}{\partial z^{(l)}_j} \frac{\partial z^{(l)}_j}{\partial W^{(l)}_{jk}} = \delta^{(l)}_j a^{(l-1)}_k$. Isso define formalmente a entrada $(j, k)$ da matriz de produto externo $\boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T$. Fisicamente: "o quanto o neurônio $j$ errou" ponderado por "o quanto a entrada $k$ estava ativa".

### Julgamento de Verdadeiro ou Falso
- ✔ **V:** $\boldsymbol{\delta} \in \mathbb{R}^{3 \times 1}$ e $\mathbf{a}^T \in \mathbb{R}^{1 \times 4}$, logo o produto externo possui dimensão $3 \times 4$, idêntica à dimensão de $W^{(l)}$.
- ✔ **V:** Como $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, quando $|z| \gg 0$, $\sigma'(z) \to 0$. Ao multiplicar elemento a elemento ($\odot$), o vetor $\boldsymbol{\delta}$ é atenuado a zero, caracterizando o desvanecimento do gradiente (*vanishing gradient*).
- ✗ **F:** Para calcular o gradiente $\nabla_{W^{(2)}}\mathcal{L} = \boldsymbol{\delta}^{(2)}(\mathbf{a}^{(1)})^T$, é estritamente necessário ter o vetor de ativações $\mathbf{a}^{(1)}$ armazenado na passagem direta.
- ✔ **V:** Cada peso participa de uma multiplicação na passagem direta e de multiplicações proporcionais na reversa; a complexidade temporal por iteração é $O(|W|)$.

---

## Pausa Ativa 3: Grafos de Computação e Acúmulo de Caminhos

### Pergunta Aberta
*Se um nó intermediário em um grafo de computação possui 3 arestas de saída (alimenta 3 variáveis subsequentes), por que a sua sensibilidade total acumulada é estritamente a **soma** das sensibilidades parciais que retornam de cada aresta, e o que aconteceria com o cálculo se uma biblioteca esquecesse de somar e simplesmente sobrescrevesse o valor do gradiente?*

**Diretrizes de Discussão com a Turma:**
- **Regra da Cadeia Multivariada:** Se $x$ influencia a saída $f$ através de três intermediários $u, v, w$, a derivada total é $\frac{\partial f}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial x} + \frac{\partial f}{\partial w}\frac{\partial w}{\partial x}$. Cada aresta de saída representa um canal independente de propagação de perturbação.
- **O Desastre da Atribuição Simples:** Se o framework fizer `grad = ...` em vez de `grad += ...`, apenas a contribuição do último nó processado será considerada. O gradiente calculado apontará para uma direção completamente distorcida, quebrando a garantia do Lema da Descida e provocando divergência do otimizador.
- **A Regra da Adição:** Lembrar aos alunos que a operação de adição $z = x + y$ atua como um roteador perfeito: $\bar{x} = \bar{z}$ e $\bar{y} = \bar{z}$, transmitindo o gradiente intacto para ambos os ramos.

### Julgamento de Verdadeiro ou Falso
- ✔ **V:** O motor de diferenciação automática não deduz cálculo simbolicamente para cada programa: ele possui uma tabela pré-programada de derivadas analíticas para cada operação atômica (adição, produto, potências, etc.).
- ✔ **V:** Como $\frac{\partial (x+y)}{\partial x} = 1$ e $\frac{\partial (x+y)}{\partial y} = 1$, a regra da adição age como um roteador de fluxo neutro, replicando o gradiente recebido $\bar{z}$ integralmente para ambas as entradas.
- ✗ **F:** Substituir o operador de acúmulo `+=` por atribuição simples `=` sobrescreveria o sinal de erro de caminhos alternativos, violando a regra da cadeia multivariada.
- ✔ **V:** A autodiff avalia números em tempo de execução ao longo do grafo; como nunca constrói a fórmula algébrica simbólica expandida no papel, ela é imune ao inchaço de expressões (*expression swell*).

---

## Pausa Ativa 4: Modos Forward vs. Reverse e Escolhas Arquiteturais

### Pergunta Aberta
*Imagine que você está projetando o sistema de controle de um braço robótico que recebe 3 ângulos de junta motora $\boldsymbol{\theta} \in \mathbb{R}^3$ e calcula a trajetória espacial de 50 pontos da estrutura $\mathbf{p} \in \mathbb{R}^{150}$. Você precisa da matriz Jacobiana completa $\frac{\partial \mathbf{p}}{\partial \boldsymbol{\theta}} \in \mathbb{R}^{150 \times 3}$ a cada ciclo de controle de 1 milissegundo. Qual modo de autodiff você deve adotar e por quantas passagens pelo grafo?*

**Diretrizes de Discussão com a Turma:**
- **A Razão Dimensional $d_{\text{in}} / d_{\text{out}}$:**
  - Modo Forward: cada passagem com um vetor da base $\mathbf{e}_i \in \mathbb{R}^{d_{\text{in}}}$ calcula uma **coluna** da Jacobiana (JVP). Requer $d_{\text{in}} = 3$ passagens completas pelo grafo.
  - Modo Reverse: cada passagem com um vetor da base $\mathbf{u}_j \in \mathbb{R}^{d_{\text{out}}}$ calcula uma **linha** da Jacobiana (VJP). Requer $d_{\text{out}} = 150$ passagens completas pelo grafo.
- **Decisão Ótima:** Como $d_{\text{in}} = 3 \ll 150 = d_{\text{out}}$, o **Modo Forward** é $50\times$ mais rápido, consumindo apenas 3 passagens e memória $\mathcal{O}(1)$, sendo perfeitamente viável para o ciclo de tempo real de 1 ms.
- **Contraste com Deep Learning:** Em redes neurais temos o oposto: $d_{\text{in}} \sim 10^7$ pesos e $d_{\text{out}} = 1$ (perda escalar). Por isso o Modo Reverse domina o treinamento de redes neurais, viabilizado pelo Teorema de Baur-Strassen ($C_{\text{grad}} \le 5 C_{\text{eval}}$).

### Julgamento de Verdadeiro ou Falso
- ✔ **V:** A associatividade da multiplicação matricial $(AB)C = A(BC)$ garante que ambos os modos calculam rigorosamente a mesma Jacobiana total, diferindo apenas na ordem de parênteses e nos custos computacionais intermediários.
- ✔ **V:** O modo forward descarta variáveis intermediárias imediatamente após alimentar seus filhos no forward pass, mantendo consumo de memória $\mathcal{O}(1)$ em relação à profundidade.
- ✗ **F:** Em uma rede com 1 milhão de pesos e perda escalar, o modo forward exigiria 1 milhão de passagens pela rede, enquanto o modo reverso obtém todos os gradientes em um único passe ($\sim 5\times$ o forward).
- ✔ **V:** Pelo Teorema de Baur-Strassen, para qualquer função escalar, o vetor gradiente completo de qualquer dimensão $n$ é computado no modo reverso em no máximo $5\times$ o custo da avaliação direta.
