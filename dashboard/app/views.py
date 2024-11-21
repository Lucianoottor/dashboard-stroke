import csv
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from time import sleep
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from .models import Patient
from .forms import UploadFileForm
import os

matplotlib.use('Agg')  # Usar 'Agg' backend para evitar problemas de thread no macOS


# Função para treinar o modelo
def train_model():
    # Carregar os dados do banco de dados
    df = pd.DataFrame(list(Patient.objects.values()))

    # Mapear variáveis categóricas para numéricas
    gender_mapping = {'Male': 0, 'Female': 1, np.nan: 2}
    ever_married_mapping = {'No': 0, 'Yes': 1, np.nan: 2}
    work_type_mapping = {'Never_worked': 0, 'Govt_job': 1, 'Self-employed': 2, 'children': 3, 'Private': 4, np.nan: 5}
    Residence_type_mapping = {'Rural': 0, 'Urban': 1, np.nan: 2}
    smoking_status_mapping = {'smokes': 0, 'formerly smoked': 1, 'unknown': 2, 'never smoked': 3, np.nan: 4}

    df['gender'] = df['gender'].map(gender_mapping)
    df['ever_married'] = df['ever_married'].map(ever_married_mapping)
    df['work_type'] = df['work_type'].map(work_type_mapping)
    df['Residence_type'] = df['Residence_type'].map(Residence_type_mapping)
    df['smoking_status'] = df['smoking_status'].map(smoking_status_mapping)

    # Dividir os dados em variáveis independentes (X) e variável dependente (y)
    features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
    X = df[features]
    Y = df['stroke']

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.4)

    # Treinar o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy, X_train


# Treinar o modelo e obter o resultado
model, accuracy, X_train = train_model()


# Função de visualização de dados
def home(request):
    return render(request, "data_visualization.html")


# Função para upload de arquivo CSV
def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['file']
            if not csv_file.name.endswith('.csv'):
                return render(request, 'upload.html', {'form': form, 'error': 'This is not a CSV file'})

            # Processar e salvar os dados no banco de dados
            df = pd.read_csv(csv_file)
            df.dropna(inplace=True)
            df.drop(columns='id', inplace=True)

            for _, row in df.iterrows():
                Patient.objects.create(
                    gender=row['gender'],
                    age=row['age'],
                    hypertension=row['hypertension'],
                    heart_disease=row['heart_disease'],
                    ever_married=row['ever_married'],
                    work_type=row['work_type'],
                    Residence_type=row['Residence_type'],
                    avg_glucose_level=row['avg_glucose_level'],
                    bmi=row['bmi'],
                    smoking_status=row['smoking_status'],
                    stroke=row['stroke'],
                )
            return HttpResponseRedirect('../')  # Redirect após o upload bem-sucedido
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})


# Função para listar os pacientes
def patient_list(request):
    patients = Patient.objects.all()  # Obter todos os pacientes
    return render(request, 'patient_list.html', {'patients': patients})

# Função para gerar o gráfico de distribuição de stroke (sim/não)
def generate_stroke_percentage_graph():
    patients = Patient.objects.all()
    df = pd.DataFrame(list(patients.values()))

    # Calcular a porcentagem de 'stroke' (sim/não)
    stroke_counts = df['stroke'].value_counts()
    stroke_labels = ['Não Stroke', 'Stroke']
    stroke_sizes = [stroke_counts.get(0, 0), stroke_counts.get(1, 0)]

    # Criar o gráfico de pizza
    fig, ax = plt.subplots()
    ax.pie(stroke_sizes, labels=stroke_labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax.axis('equal')  # Garantir que o gráfico seja um círculo

    return fig

# Função para gerar o gráfico de distribuição de stroke por idade
def generate_pizza():
    patients = Patient.objects.all()
    df = pd.DataFrame(list(patients.values()))
    dfworktype = df['work_type']
    dfsmoketype = df['smoking_status']
    dfgender = df['gender']

    # Contagem dos valores
    count_worktype = dfworktype.value_counts()
    count_smoketype = dfsmoketype.value_counts()
    count_gender = dfgender.value_counts()

    # Explodir o primeiro setor
    explode_worktype = [0.1 if i == 0 else 0 for i in range(len(count_worktype))]
    explode_smoketype = [0.1 if i == 0 else 0 for i in range(len(count_smoketype))]
    explode_gender = [0.1 if i == 0 else 0 for i in range(len(count_gender))]

    # Criar a figura com 1 linha e 3 colunas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Gráfico de Work Type
    axes[0].pie(count_worktype, labels=count_worktype.index, autopct='%1.1f%%', startangle=90, shadow=True,
                explode=explode_worktype)
    axes[0].set_title('Proporções de Work Type')

    # Gráfico de Smoking Status
    axes[1].pie(count_smoketype, labels=count_smoketype.index, autopct='%1.1f%%', startangle=90, shadow=True,
                explode=explode_smoketype)
    axes[1].set_title('Proporções de Smoking Status')

    # Gráfico de Gender
    axes[2].pie(count_gender, labels=count_gender.index, autopct='%1.1f%%', startangle=90, shadow=True,
                explode=explode_gender)
    axes[2].set_title('Proporções de Gender')

    # Ajuste o layout para não sobrepor os gráficos
    plt.tight_layout()

    # Exibir os gráficos
    plt.show()

    return fig

def generate_histograms():
    patients = Patient.objects.all()
    df = pd.DataFrame(list(patients.values()))
    # Criar a figura com 1 linha e 3 colunas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Gráfico de distribuição de Age
    sns.histplot(data=df, x="age", kde=True, ax=axes[0])
    axes[0].set_title('Age Distribution')

    # Gráfico de distribuição de Average Glucose Level
    sns.histplot(data=df, x="avg_glucose_level", kde=True, ax=axes[1])
    axes[1].set_title('Average Glucose Level Distribution')
    axes[1].set_xlabel('Glucose Level (On Average)')

    # Gráfico de distribuição de BMI
    sns.histplot(data=df, x="bmi", kde=True, ax=axes[2])
    axes[2].set_title('BMI Distribution')
    axes[2].set_xlabel('BMI')

    # Ajuste o layout para não sobrepor os gráficos
    plt.tight_layout()

    # Exibir os gráficos
    plt.show()

    return fig


def graph_stroke_age():
    # Retrieve patient data from the database
    patients = Patient.objects.all()
    df = pd.DataFrame(list(patients.values()))

    # Create the figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Graph 1: Stroke Rate by Age
    age_stroke_rate = df.groupby('age')['stroke'].mean().reset_index()
    sns.lineplot(x="age", y="stroke", data=age_stroke_rate, color="blue", linewidth=2.5, dashes=False, ax=axes[0])
    axes[0].set_title('Average Stroke Rate Among Multiple Ages')
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Stroke Rate')

    # Graph 2: Glucose Level Distribution by Stroke
    sns.boxplot(x='stroke', y='avg_glucose_level', data=df, ax=axes[1])
    axes[1].set_title('Distribution of Glucose Levels by Stroke')
    axes[1].set_xlabel('Stroke (0=no, 1=yes)')
    axes[1].set_ylabel('Glucose Level')

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Return the combined figure object
    return fig

# Função para gerar e retornar o gráfico em formato de imagem diretamente
def generate_graph(request, graph_type):
    # Escolher o gráfico com base no tipo
    if graph_type == 'stroke_percentage':
        fig = generate_stroke_percentage_graph()
    elif graph_type == 'pizza_graphs':
        fig = generate_pizza()
    elif graph_type == 'histograms':
        fig = generate_histograms()
    elif graph_type == 'stroke_age':
        fig = graph_stroke_age()
    else:
        return HttpResponse("Tipo de gráfico inválido.", status=400)

    # Gerar o gráfico diretamente na resposta HTTP
    response = HttpResponse(content_type='image/png')
    fig.savefig(response, format='png')
    response['Content-Disposition'] = 'inline; filename="graph.png"'

    return response


# Página principal para exibição
def visualize_data(request):
    return render(request, 'data_visualization.html')



# Função para a página de previsão do modelo
def modelo(request):
    if request.method == "POST":
        # Receber dados do formulário
        gender = request.POST['gender']
        age = int(request.POST['age'])
        hypertension = request.POST['hypertension']
        heart_disease = request.POST['heart_disease']
        ever_married = request.POST['ever_married']
        work_type = request.POST['work_type']
        Residence_type = request.POST['Residence_type']
        avg_glucose_level = float(request.POST['avg_glucose_level'])
        bmi = float(request.POST['bmi'])
        smoking_status = request.POST['smoking_status']

        # Mapear valores categóricos para numéricos
        columns_map = {
            'gender': {'Male': 0, 'Female': 1, 'Other': 2},
            'hypertension': {'Yes': 0, 'No': 1},
            'heart_disease': {'Yes': 0, 'No': 1},
            'ever_married': {'Yes': 0, 'No': 1},
            'work_type': {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4},
            'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3},
            'Residence_type': {'Urban': 0, 'Rural': 1}
        }

        # Substituir valores categóricos por valores numéricos
        gender = columns_map['gender'].get(gender)
        hypertension = columns_map['hypertension'].get(hypertension)
        heart_disease = columns_map['heart_disease'].get(heart_disease)
        ever_married = columns_map['ever_married'].get(ever_married)
        work_type = columns_map['work_type'].get(work_type)
        smoking_status = columns_map['smoking_status'].get(smoking_status)
        Residence_type = columns_map['Residence_type'].get(Residence_type)

        # Preparar os dados para a previsão
        new_data = pd.DataFrame([[age, hypertension, heart_disease, smoking_status]],
                                columns=['age', 'hypertension', 'heart_disease', 'smoking_status'])

        # Normalizar os dados de entrada
        scaler = StandardScaler()
        new_data_scaled = scaler.fit_transform(X_train)

        # Prever o risco de stroke
        prediction = model.predict(new_data_scaled)

        # Resultado
        result = "Alto risco de Stroke" if prediction[0] == 1 else "Baixo risco de Stroke"
        return render(request, 'resultado.html', {'result': result})

    return render(request, 'modelo.html', {'score': accuracy})
