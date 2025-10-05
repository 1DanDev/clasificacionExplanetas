# clasificador_exoplanetas.py

# ====================
# PASO 1: IMPORTS Y CONFIG
# ====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("🔧 Configuración completada")

# ====================
# PASO 2: CARGA DE DATOS
# ====================
df = pd.read_csv('cumulative_2025.10.04_13.19.42.csv', comment='#')
print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# ====================
# PASO 3: EXPLORACIÓN INICIAL
# ====================
print("\n📊 DISTRIBUCIÓN DE CLASES:")
print(df['koi_disposition'].value_counts())

# ====================
# PASO 4: ANÁLISIS DE FEATURES
# ====================
features = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_score', 
           'koi_impact', 'koi_duration', 'koi_steff', 'koi_srad', 'koi_smass']

print(f"\n🔍 ANALIZANDO {len(features)} CARACTERÍSTICAS:")
print("Valores nulos por característica:")
print(df[features].isnull().sum())

print("\n📈 ESTADÍSTICAS BÁSICAS:")
print(df[features].describe())

# ====================
# PASO 5: DEFINIR TARGET Y FEATURES
# ====================
print("\n🎯 DEFINICIÓN DEL PROBLEMA:")

# Opción 1: Clasificación binaria (Confirmados vs No confirmados)
df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
print(f"Target binario - Confirmados: {(df['target'] == 1).sum()}, No confirmados: {(df['target'] == 0).sum()}")

# Seleccionar características con menos valores nulos
selected_features = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_score']
X = df[selected_features]
y = df['target']

print(f"\n✅ Características seleccionadas: {selected_features}")

# ====================
# PASO 6: PREPROCESAMIENTO
# ====================
print("\n🔧 PREPROCESANDO DATOS...")

# Manejar valores nulos
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print(f"Forma de X después del preprocesamiento: {X_scaled.shape}")

# ====================
# PASO 7: DIVISIÓN TRAIN/TEST
# ====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Mantener proporción de clases
)

print(f"\n📊 DIVISIÓN DE DATOS:")
print(f"Train: {X_train.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")
print(f"Proporción de clases en train: {np.mean(y_train):.3f}")
print(f"Proporción de clases en test: {np.mean(y_test):.3f}")

# ====================
# PASO 8: ENTRENAMIENTO DEL MODELO
# ====================
print("\n🤖 ENTRENANDO MODELO...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

print("✅ Modelo entrenado!")

# ====================
# PASO 9: EVALUACIÓN
# ====================
print("\n📊 EVALUANDO MODELO...")

# Predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades para clase 1

# Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 EXACTITUD (Accuracy): {accuracy:.3f}")

print("\n📋 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
print("🎭 MATRIZ DE CONFUSIÓN:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ====================
# PASO 10: VISUALIZACIÓN
# ====================
print("\n📈 CREANDO VISUALIZACIONES...")

# 1. Matriz de confusión visual
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')

# 2. Importancia de características
plt.subplot(1, 3, 2)
feature_importance = model.feature_importances_
feature_names = selected_features
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Importancia de Características')
plt.xlabel('Importancia')

# 3. Distribución de probabilidades
plt.subplot(1, 3, 3)
plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='No Confirmados', bins=20)
plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Confirmados', bins=20)
plt.title('Distribución de Probabilidades')
plt.xlabel('Probabilidad de ser Confirmado')
plt.legend()

plt.tight_layout()
plt.show()

print("\n✨ ¡ANÁLISIS COMPLETADO!")