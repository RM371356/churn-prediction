# Plano de Monitoramento do Modelo de Churn

## 1. Visão Geral

Após o deploy do modelo de Machine Learning, é essencial monitorar continuamente seu desempenho para garantir que continue gerando valor ao longo do tempo.

Modelos de ML não são estáticos e podem sofrer degradação devido a mudanças nos dados ou no comportamento do negócio. Este plano define as métricas, alertas e ações necessárias para manter a qualidade do modelo em produção.

---

## 2. Objetivos do Monitoramento

- Detectar degradação de performance do modelo  
- Identificar mudanças nos dados de entrada (data drift)  
- Monitorar desempenho da API (latência e disponibilidade)  
- Garantir confiabilidade das previsões  
- Suportar decisões de retreinamento  

---

## 3. Coleta de Dados

Durante a execução em produção, o plano seria fazer a coleta de dados para cada requisição feita da API, com as seguintes informações:

- Dados de entrada (features do cliente)  
- Predição gerada pelo modelo  
- Tempo de resposta (latência)  
- Timestamp da requisição  

Esses dados podem ser armazenados via logs estruturados (JSON), permitindo análises posteriores.

---

## 4. Métricas de Monitoramento

### 4.1 Métricas de Performance do Modelo

Avaliam a qualidade das previsões:

- Accuracy  
- Precision  
- Recall  
- F1-score  

**Métrica principal:** Recall  

Justificativa: identificar o maior número possível de clientes com risco de churn.

---

### 4.2 Monitoramento de Model Drift

Avalia a degradação da performance ao longo do tempo.

- Comparação entre previsões e resultados reais caso esteja disponíveis para análise.  
- Queda nas métricas de avaliação indica possível necessidade de retreinamento  .

---

### 4.3 Monitoramento de Data Drift

Avalia mudanças na distribuição dos dados de entrada.

Exemplos de variáveis monitoradas:

- Tenure (tempo de contrato)  
- MonthlyCharges  
- TotalCharges  

Técnicas utilizadas:

- Comparação entre distribuição dos dados de treino e produção  
- Análise estatística (ex: histogramas, média, variância)

Impacto:  
Mudanças significativas indicam que o modelo pode não estar mais adequado ao cenário atual.

---

### 4.4 Monitoramento de Latência

Avalia o tempo de resposta da API.

Métricas monitoradas:

- Tempo médio de resposta  
- P50 (mediana)  
- P95 (95% das requisições)  
- P99 (pior caso comum)

Importância:  
Altos tempos de resposta impactam diretamente a experiência do usuário.

---

### 4.5 Monitoramento de Uso

- Número de requisições  
- Frequência de uso do modelo  
- Volume de dados processados  

Essas métricas ajudam a entender a adoção e necessidade de escalabilidade.

---

## 5. Definição de Alertas

Alertas são definidos para identificar problemas rapidamente.

| Tipo de Problema        | Condição                         | Ação |
|------------------------|----------------------------------|------|
| Queda de performance   | F1-score < 0.70                 | Investigar modelo |
| Recall baixo           | Recall < 0.65                   | Avaliar impacto no churn |
| Data Drift             | Mudança significativa na distribuição | Analisar dados |
| Latência alta          | P95 > 500ms                     | Verificar infraestrutura |
| Falhas na API          | Erro > 5% das requisições       | Investigar sistema |

---

## 6. Playbook de Resposta

Define ações a serem tomadas quando problemas forem detectados.

| Problema               | Ação Recomendada |
|----------------------|-----------------|
| Queda de performance | Retreinar o modelo com dados atualizados |
| Data Drift           | Atualizar dataset e revisar features |
| Model Drift          | Ajustar modelo ou treinar nova versão |
| Alta latência        | Otimizar código ou escalar infraestrutura |
| Erros na API         | Corrigir bugs e validar entradas |

---

## 7. Estratégia de Retreinamento

O modelo deve ser atualizado quando:

- Houver queda consistente nas métricas  
- Data drift for identificado  
- Mudanças no negócio ocorrerem  

Estratégias:

- Retreinamento periódico   
- Retreinamento baseado em eventos (trigger por alerta)  

---

## 8. Logging e Auditoria

A aplicação registrará logs estruturados contendo:

- Request payload  
- Response (predição)  
- Latência  
- Timestamp  

Benefícios:

- Debugging  
- Auditoria  
- Base para análise de drift  
- Base para retreinamento  

---

## 9. Ferramentas de Monitoramento

Possíveis ferramentas para evolução do projeto:

- Prometheus (coleta de métricas)  
- Grafana (visualização)  
- ELK Stack (logs)  

Ferramenta utilizada de monitoramento:
- MLflow  

---

## 10. Boas Práticas Aplicadas

- Monitoramento contínuo do modelo  
- Separação entre métricas de sistema e modelo  
- Uso de logs estruturados  
- Definição de alertas automatizados  
- Planejamento de retreinamento  

---

## 11. Conclusão

O monitoramento é essencial para garantir que o modelo continue performando adequadamente em produção.

A combinação de métricas, alertas e ações permite identificar problemas rapidamente e manter a confiabilidade do sistema ao longo do tempo.
