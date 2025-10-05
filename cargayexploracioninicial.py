# Cargar el dataset
df = pd.read_csv('cumulative_2025.10.04_13.19.42.csv', comment='#')

print("=== INFORMACIÓN BÁSICA ===")
print(f"Dimensiones: {df.shape}")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print("\nPrimeras 5 filas:")
print(df.head())

print("\n=== TIPOS DE DATOS ===")
print(df.dtypes)