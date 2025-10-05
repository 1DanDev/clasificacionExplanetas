# Instalar librerías si no las tienes
# !pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd  # ¡ESTA LÍNEA FALTABA!
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")