def utility_function(X, Y, alpha, beta):
    return X**alpha * Y**beta

# Ejemplo de cálculo de utilidad
X = 5  # Cantidad de bien X
Y = 3  # Cantidad de bien Y
alpha = 0.5  # Parámetro alpha
beta = 0.8  # Parámetro beta

utilidad = utility_function(X, Y, alpha, beta)
print(f"La utilidad del consumidor con {X} unidades de bien X y {Y} unidades de bien Y es {utilidad:.2f}")
