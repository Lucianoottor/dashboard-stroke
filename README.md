# Stroke Prediction Dashboard

Este é um projeto desenvolvido com o **Django** e **Python** para prever o risco de acidente vascular cerebral (AVC) com base em dados médicos. O sistema apresenta um painel interativo para visualização de dados e treinamento de modelos de **Machine Learning**, especificamente utilizando o algoritmo **Random Forest**. O projeto também permite o upload de novos dados para treinamento e avaliação do modelo.

## Funcionalidades

1. **Dashboard de Visualização de Dados**:
   - Visualização de gráficos interativos, como:
     - Distribuição de pacientes com e sem AVC
     - Proporções de variáveis como gênero, tipo de trabalho, e status de fumo
     - Histograma de idade, nível médio de glicose e IMC
     - Análise do risco de AVC por idade e nível de glicose

2. **Modelo de Machine Learning**:
   - O modelo **Random Forest** é treinado utilizando dados de pacientes para prever o risco de AVC com base em características como idade, hipertensão, doenças cardíacas, status de fumo, entre outros.
   - O modelo é avaliado com uma precisão e utilizado para previsões em tempo real.

3. **Upload de Dados**:
   - Permite o upload de arquivos CSV contendo dados de pacientes, que são processados e salvos no banco de dados para treinar o modelo ou realizar novas previsões.

4. **Previsão de Risco de AVC**:
   - Com base em dados fornecidos pelo usuário (por exemplo, idade, status de fumo, presença de doenças), o sistema prevê se há um risco alto ou baixo de AVC.

## Tecnologias Utilizadas

- **Django**: Framework web para desenvolvimento do backend.
- **Python**: Linguagem principal utilizada para o desenvolvimento.
- **Pandas** e **NumPy**: Para manipulação e limpeza de dados.
- **Matplotlib** e **Seaborn**: Para visualização de dados.
- **Scikit-learn**: Para treinamento e avaliação do modelo de Machine Learning (Random Forest).
- **SQLite**: Banco de dados utilizado para armazenar informações dos pacientes.


## Como Rodar o Projeto

### 1. Clone o repositório

git clone https://github.com/usuario/stroke_prediction_dashboard.git
cd stroke_prediction_dashboard

### 2. Crie e ative um ambiente virtual

python3 -m venv venv
source venv/bin/activate  # Para Linux/Mac
venv\Scripts\activate     # Para Windows

### 3. Configure o banco de dados

python manage.py migrate

### 4. Inicie o servidor de desenvolvimento

python manage.py runserver

### 5. Acesse o Painel de Visualização de Dados

Após iniciar o servidor, abra um navegador e vá para o endereço http://127.0.0.1:8000/. Você pode visualizar gráficos e dados relacionados ao risco de AVC.

### 6. Carregue novos dados

Para adicionar novos dados de pacientes, acesse a página de upload em http://127.0.0.1:8000/upload/ e faça o upload de um arquivo CSV com as colunas gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, e stroke.
