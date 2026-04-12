# ML Canvas — Previsão de Churn
**Disciplina:** Machine Learning Engineering — Fase 1  
**Dataset:** Telco Customer Churn  — 7.043 clientes

---

## 1. Definição do Problema de Negócio

Uma operadora de telecomunicações enfrenta alta taxa de cancelamento de clientes. O objetivo do projeto é **identificar, com antecedência, quais clientes estão propensos a cancelar o serviço**, permitindo que a equipe de retenção tome ações preventivas antes que o cancelamento ocorra.

> Objetivo: reduzir a taxa de churn mensal identificando clientes em risco com pelo menos uma semana de antecedência.

---

## 2. Métrica de Sucesso (KPI)

| KPI de Negócio | Descrição |
|---|---|
| Taxa de churn mensal | Percentual de clientes que cancelam por mês — baseline atual: **26,5%** |
| Taxa de retenção pós-ação | Percentual de clientes identificados pelo modelo que permaneceram após intervenção |
| ROI | Receita de churn evitado vs custo de desenvolvimento do modelo |


A métrica técnica principal do modelo será o **Recall**, pois o custo de não identificar um cliente que vai cancelar é maior do que acionar retenção desnecessariamente.

---

## 3. Levantamento de Requisitos e Restrições

**Perguntas que seriam feitas aos stakeholders:**

- Com quanto tempo de antecedência precisamos identificar os clientes em risco?
- Qual o orçamento disponível para ações de retenção por cliente?
- Existe alguma restrição legal (LGPD) sobre o uso dos dados dos clientes?
- Os dados históricos de cancelamento estão disponíveis e em que período?
- O modelo precisa ser interpretável para a equipe de negócio entender os motivos do risco?
- Qual sistema receberá as previsões — CRM, planilha, dashboard?

**Restrições identificadas:**

- Conformidade com a **LGPD** no uso de dados pessoais dos clientes
- O modelo precisa ser acionável semanalmente pelo time de CRM
- 11 valores nulos em `Total Charges` — necessita tratamento antes da modelagem

---

## 4. Dados e Variáveis Relevantes

Com base na análise exploratória do dataset, as variáveis mais relevantes para prever churn são:

| Variável | Tipo | Por que é relevante |
|---|---|---|
| `Contract` | Categórico | Clientes mensais têm taxa de churn de **42,7%** vs. 2,8% nos bienais |
| `Tenure Months` | Numérico | Churners têm média de **18 meses** vs. 37 meses dos que ficam |
| `Monthly Charges` | Numérico | Churners pagam em média **R$74/mês** vs. R$61 dos demais |
| `Online Security` | Binário | Sem segurança: **41,8%** de churn vs. 14,6% com o serviço |
| `Tech Support` | Binário | Sem suporte: **41,6%** de churn vs. 15,2% com suporte |
| `Internet Service` | Categórico | Fibra ótica concentra **41,9%** de churn — 2x mais que DSL |
| `Senior Citizen` | Binário | Idosos têm **41,7%** de churn vs. 23,6% dos demais |

---

## 5. Envolvimento dos Stakeholders

| Stakeholder | Papel no Projeto |
|---|---|
| **Diretoria** | Define o objetivo estratégico e aprova o projeto |
| **Time de CRM / Retenção** | Usuário final — executa as ações sobre os clientes identificados |
| **Equipe de Dados / TI** | Fornece acesso aos dados e mantém a infraestrutura |
| **Jurídico / Compliance** | Garante conformidade com LGPD no uso dos dados |

O time de CRM deve ser envolvido desde o início para definir como receberá as previsões (formato, frequência) e para validar se os resultados fazem sentido na prática, conforme orientado pelo framework CRISP-DM na fase de *Business Understanding*.
