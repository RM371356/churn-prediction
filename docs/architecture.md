# Arquitetura de Deploy do Modelo de Churn

## 1. Visão Geral

O modelo de Machine Learning desenvolvido para previsão de churn foi implantado utilizando uma arquitetura de inferência em tempo real (real-time), disponibilizada por meio de uma API.

Essa abordagem permite que sistemas externos realizem requisições e obtenham previsões instantaneamente, possibilitando integração com aplicações e tomada de decisão em tempo real.

---

## 2. Tipo de Deploy Escolhido

**Real-time (Online Inference via API)**

O modelo é exposto através de uma API, permitindo que clientes ou sistemas enviem dados e recebam previsões sob demanda.

---

## 3. Justificativa da Escolha

A escolha pelo deploy em tempo real foi baseada nos seguintes fatores:

- Necessidade de respostas imediatas para suporte à decisão  
- Possibilidade de integração com sistemas externos (ex: CRM, atendimento ao cliente)  
- Uso em cenários interativos com clientes  
- Flexibilidade de consumo do modelo via API  

No contexto de churn, essa abordagem permite identificar clientes com risco de cancelamento no momento da interação, possibilitando ações imediatas de retenção.

---

## 4. Comparação com Abordagem Batch

| Critério        | Batch Processing                  | Real-time (Escolhido)          |
|----------------|----------------------------------|--------------------------------|
| Latência        | Alta (execução periódica)        | Baixa (resposta imediata)      |
| Uso             | Processamento em massa           | Predição individual sob demanda |
| Complexidade    | Menor                           | Maior                          |
| Atualização     | Periódica                       | Contínua                       |

**Decisão:**  
A abordagem real-time foi escolhida por atender melhor aos requisitos de resposta imediata e integração com sistemas.

---

## 5. Arquitetura do Sistema

Fluxo geral da aplicação:

Cliente → API → Pipeline → Modelo → Resposta

---

## 6. Componentes da Arquitetura

### 6.1 Cliente

Responsável por enviar os dados do cliente para predição.  
Pode ser um sistema externo, frontend ou serviço interno.

---

### 6.2 API de Inferência

Responsável por:

- Receber requisições HTTP (POST)
- Validar dados de entrada
- Converter dados para o formato esperado
- Encaminhar para o pipeline de Machine Learning
- Retornar a predição em formato JSON

A API funciona como camada de entrada do modelo.

---

### 6.3 Pipeline de Machine Learning

Responsável por garantir consistência entre treino e produção.

Inclui:

- Pré-processamento dos dados  
- Encoding de variáveis categóricas  
- Engenharia de features  
- Seleção de atributos  
- Execução do modelo  

O pipeline encapsula todas as transformações necessárias antes da predição.

---

### 6.4 Modelo Treinado

- Modelo previamente treinado e validado
- Persistido em disco utilizando serialização
- Carregado na inicialização da API

Esse modelo é responsável por gerar a predição final de churn.

---

### 6.5 Resposta

A API retorna um JSON contendo:

- Predição (churn ou não churn)
---

## 7. Fluxo de Execução

1. O cliente envia uma requisição HTTP contendo os dados  
2. A API recebe e valida os dados de entrada  
3. Os dados são convertidos para um formato tabular (DataFrame)  
4. O pipeline aplica todas as transformações necessárias  
5. O modelo gera a predição  
6. O resultado é retornado ao cliente  

---

## 8. Persistência do Modelo

O modelo é salvo localmente utilizando serialização, permitindo:

- Reutilização em produção  
- Consistência entre treino e inferência  
- Facilidade de carregamento    

---

## 9. Escalabilidade e Produção

Para um ambiente produtivo, a arquitetura pode ser evoluída com:

- Containerização da aplicação (Docker)  
- Orquestração de containers (Kubernetes)  
- Balanceamento de carga  
- Pipeline de CI/CD para automação de deploy  
- Monitoramento de performance e logs  

---


## 10. Boas Práticas de Engenharia Aplicadas

A arquitetura foi construída seguindo princípios de MLOps:

- Separação entre treinamento e inferência  
- Uso de pipeline reproduzível  
- Encapsulamento do modelo em serviço  
- Preparação para monitoramento contínuo  
- Estrutura modular do código  

---

## 11. Futuras Evoluções

- Deploy em ambiente cloud (AWS, Azure ou GCP)  
- Monitoramento com ferramentas como Prometheus e Grafana  
- Automação de retreinamento do modelo  
