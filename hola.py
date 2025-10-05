import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class ExoplanetDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def generar_datos_ejemplo(self, n_samples=2000):
        """
        Genera datos de ejemplo similares a los de exoplanetas
        ya que no tenemos el dataset original
        """
        np.random.seed(42)
        
        datos = {
            'ixol_score': np.concatenate([
                np.random.normal(15, 5, n_samples//2),  # No confirmados
                np.random.normal(75, 15, n_samples//2)  # Confirmados
            ]),
            'ixol_prad': np.concatenate([
                np.random.normal(25, 10, n_samples//2), # No confirmados
                np.random.normal(8, 4, n_samples//2)    # Confirmados
            ]),
            'ixol_period': np.concatenate([
                np.random.normal(50, 20, n_samples//2), # No confirmados
                np.random.normal(15, 8, n_samples//2)   # Confirmados
            ]),
            'ixol_teq': np.concatenate([
                np.random.normal(1200, 300, n_samples//2), # No confirmados
                np.random.normal(800, 200, n_samples//2)   # Confirmados
            ]),
            'ixol_depth': np.concatenate([
                np.random.normal(0.5, 0.2, n_samples//2), # No confirmados
                np.random.normal(0.1, 0.05, n_samples//2) # Confirmados
            ])
        }
        
        df = pd.DataFrame(datos)
        
        # Crear variable objetivo (1: Confirmado, 0: No confirmado)
        # Basado en combinación de características
        prob_confirmado = (
            (df['ixol_score'] > 50).astype(int) * 0.4 +
            (df['ixol_prad'] < 20).astype(int) * 0.3 +
            (df['ixol_period'] < 30).astype(int) * 0.2 +
            (df['ixol_teq'] < 1000).astype(int) * 0.1
        )
        
        df['confirmado'] = (prob_confirmado + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)
        
        return df
    
    def entrenar_modelo(self, df):
        """Entrena el modelo con los datos"""
        print("🔧 Entrenando modelo de detección de exoplanetas...")
        
        # Separar características y objetivo
        X = df[['ixol_score', 'ixol_prad', 'ixol_period', 'ixol_teq', 'ixol_depth']]
        y = df['confirmado']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Escalar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Métricas
        print("\n📊 EVALUACIÓN DEL MODELO:")
        print("=" * 50)
        print(f"Precisión: {self.model.score(X_test_scaled, y_test):.3f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📈 Matriz de Confusión:")
        print(f"Verdaderos Negativos: {cm[0,0]}")
        print(f"Falsos Positivos: {cm[0,1]}")
        print(f"Falsos Negativos: {cm[1,0]}")
        print(f"Verdaderos Positivos: {cm[1,1]}")
        
        self.is_trained = True
        print("\n✅ Modelo entrenado exitosamente!")
        
        return X_test_scaled, y_test, y_pred, y_proba
    
    def predecir_individual(self):
        """Predicción para un solo candidato"""
        if not self.is_trained:
            print("❌ Primero debes entrenar el modelo!")
            return
        
        print("\n" + "="*60)
        print("🔭 PREDICCIÓN DE CANDIDATO A EXOPLANETA")
        print("="*60)
        
        try:
            # Ingreso de datos
            print("\nIngresa los valores del candidato:")
            score = float(input("Ixol_score (0-100): "))
            prad = float(input("Ixol_prad (Radio en radios terrestres): "))
            period = float(input("Ixol_period (Período orbital en días): "))
            teq = float(input("Ixol_teq (Temperatura de equilibrio en K): "))
            depth = float(input("Ixol_depth (Profundidad del tránsito): "))
            
            # Preparar datos
            X_nuevo = np.array([[score, prad, period, teq, depth]])
            X_nuevo_scaled = self.scaler.transform(X_nuevo)
            
            # Predicciones
            prediccion = self.model.predict(X_nuevo_scaled)[0]
            probabilidad = self.model.predict_proba(X_nuevo_scaled)[0]
            prob_confirmado = probabilidad[1]
            
            # Mostrar resultados
            self._mostrar_resultado_detallado(prediccion, prob_confirmado, 
                                            [score, prad, period, teq, depth])
            
        except ValueError:
            print("❌ Error: Ingresa valores numéricos válidos")
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
    
    def _mostrar_resultado_detallado(self, prediccion, prob_confirmado, datos):
        """Muestra el resultado de la predicción de forma detallada"""
        print("\n" + "="*50)
        print("📊 RESULTADO DEL ANÁLISIS")
        print("="*50)
        print(f"Probabilidad de ser exoplaneta: {prob_confirmado:.2%}")
        
        # Análisis de características
        score, prad, period, teq, depth = datos
        
        print(f"\n🔍 ANÁLISIS DE CARACTERÍSTICAS:")
        
        factores_positivos = []
        factores_negativos = []
        
        if score > 50:
            factores_positivos.append("Ixol_score ALTO (señal fuerte)")
        else:
            factores_negativos.append("Ixol_score BAJO (señal débil)")
            
        if prad < 20:
            factores_positivos.append("Radio planetario NORMAL")
        else:
            factores_negativos.append("Radio MUY GRANDE (posible estrella)")
            
        if period < 30:
            factores_positivos.append("Período orbital CORTO (típico de planetas)")
        else:
            factores_negativos.append("Período orbital LARGO (atípico)")
            
        if teq < 1000:
            factores_positivos.append("Temperatura MODERADA")
        else:
            factores_negativos.append("Temperatura MUY ALTA")
        
        # Mostrar factores
        if factores_positivos:
            print("✅ Factores que favorecen confirmación:")
            for factor in factores_positivos:
                print(f"   • {factor}")
                
        if factores_negativos:
            print("❌ Factores que sugieren falso positivo:")
            for factor in factores_negativos:
                print(f"   • {factor}")
        
        # Clasificación final
        print(f"\n🎯 CLASIFICACIÓN FINAL:")
        if prob_confirmado >= 0.8:
            print("✅ EXOPLANETA CONFIRMADO (Confianza: ALTA)")
            print("   Recomendación: Priorizar observaciones de seguimiento")
        elif prob_confirmado >= 0.6:
            print("🟡 CANDIDATO FUERTE (Confianza: MEDIA-ALTA)")
            print("   Recomendación: Observaciones adicionales recomendadas")
        elif prob_confirmado >= 0.4:
            print("🟠 POSIBLE FALSO POSITIVO (Confianza: BAJA)")
            print("   Recomendación: Análisis más detallado necesario")
        elif prob_confirmado >= 0.2:
            print("🔴 PROBABLE FALSO POSITIVO (Confianza: MUY BAJA)")
            print("   Recomendación: Revisar datos de entrada")
        else:
            print("❌ NO ES EXOPLANETA (Confianza: ALTA)")
            print("   Recomendación: Descartar candidato")
    
    def probar_ejemplos_predefinidos(self):
        """Prueba con ejemplos predefinidos"""
        if not self.is_trained:
            print("❌ Primero debes entrenar el modelo!")
            return
            
        ejemplos = [
            [85, 12, 18, 750, 0.08],   # Muy probable exoplaneta
            [25, 35, 65, 1500, 0.6],    # Muy probable falso positivo
            [60, 15, 25, 900, 0.1],     # Candidato fuerte
            [45, 28, 45, 1100, 0.4],    # Posible falso positivo
            [10, 50, 80, 1800, 0.8]     # Claramente no exoplaneta
        ]
        
        descripciones = [
            "Señal fuerte, radio pequeño, período corto",
            "Señal débil, radio grande, período largo", 
            "Señal media, características moderadas",
            "Señal media-baja, características atípicas",
            "Señal muy débil, características estelares"
        ]
        
        print("\n" + "="*60)
        print("🧪 PRUEBA CON EJEMPLOS PREDEFINIDOS")
        print("="*60)
        
        for i, (ejemplo, desc) in enumerate(zip(ejemplos, descripciones), 1):
            print(f"\n--- Ejemplo {i}: {desc} ---")
            
            X_ejemplo = np.array([ejemplo])
            X_ejemplo_scaled = self.scaler.transform(X_ejemplo)
            
            prediccion = self.model.predict(X_ejemplo_scaled)[0]
            prob_confirmado = self.model.predict_proba(X_ejemplo_scaled)[0][1]
            
            self._mostrar_resultado_detallado(prediccion, prob_confirmado, ejemplo)
    
    def mostrar_importancia_caracteristicas(self):
        """Muestra la importancia de cada característica"""
        if not self.is_trained:
            print("❌ Primero debes entrenar el modelo!")
            return
            
        caracteristicas = ['Ixol_score', 'Ixol_prad', 'Ixol_period', 'Ixol_teq', 'Ixol_depth']
        importancia = self.model.feature_importances_
        
        print("\n📊 IMPORTANCIA DE CARACTERÍSTICAS:")
        print("=" * 40)
        for car, imp in sorted(zip(caracteristicas, importancia), 
                             key=lambda x: x[1], reverse=True):
            print(f"{car:<15}: {imp:.3f} ({imp*100:.1f}%)")
    
    def guardar_modelo(self, nombre_archivo='modelo_exoplanetas.pkl'):
        """Guarda el modelo entrenado"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, nombre_archivo)
            print(f"✅ Modelo guardado como {nombre_archivo}")
        else:
            print("❌ No hay modelo entrenado para guardar")
    
    def cargar_modelo(self, nombre_archivo='modelo_exoplanetas.pkl'):
        """Carga un modelo previamente entrenado"""
        try:
            datos = joblib.load(nombre_archivo)
            self.model = datos['model']
            self.scaler = datos['scaler']
            self.is_trained = True
            print(f"✅ Modelo cargado desde {nombre_archivo}")
        except FileNotFoundError:
            print("❌ Archivo de modelo no encontrado")

def main():
    """Función principal con menú interactivo"""
    detector = ExoplanetDetector()
    
    while True:
        print("\n" + "="*60)
        print("🌌 SISTEMA DE DETECCIÓN DE EXOPLANETAS")
        print("="*60)
        print("1. Entrenar modelo con datos de ejemplo")
        print("2. Predecir candidato individual")
        print("3. Probar ejemplos predefinidos") 
        print("4. Mostrar importancia de características")
        print("5. Guardar modelo entrenado")
        print("6. Cargar modelo existente")
        print("7. Salir")
        
        opcion = input("\nSelecciona una opción (1-7): ")
        
        if opcion == "1":
            # Generar datos y entrenar
            df = detector.generar_datos_ejemplo()
            detector.entrenar_modelo(df)
            
        elif opcion == "2":
            detector.predecir_individual()
            
        elif opcion == "3":
            detector.probar_ejemplos_predefinidos()
            
        elif opcion == "4":
            detector.mostrar_importancia_caracteristicas()
            
        elif opcion == "5":
            detector.guardar_modelo()
            
        elif opcion == "6":
            detector.cargar_modelo()
            
        elif opcion == "7":
            print("👋 ¡Hasta luego! Que encuentres muchos exoplanetas 🪐")
            break
            
        else:
            print("❌ Opción no válida. Por favor, elige 1-7")

if __name__ == "__main__":
    main()