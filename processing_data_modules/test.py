import matplotlib.pyplot as plt
import numpy as np
import math

# Definición de la función a minimizar
def f(x):
    return (x-2)**2 + 1

# Función para generar la lista de números de Fibonacci
def fibonacci_numbers(n):
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

# Parámetros
a = 0
b = 4
N = 5  # Número total predefinido de iteraciones

# Generar números Fibonacci (necesitamos hasta F(N+1))
fib = fibonacci_numbers(N+1)  # Ejemplo: si N=5, obtenemos [1, 1, 2, 3, 5, 8]

# Lista para almacenar resultados de cada iteración
iterations = []

# Proceso iterativo (se hacen N-2 iteraciones; en este caso, 3 iteraciones)
for k in range(1, N-1):
    L = b - a
    ratio1 = fib[N - k - 1] / fib[N - k + 1]
    ratio2 = fib[N - k] / fib[N - k + 1]
    x1 = a + ratio1 * L
    x2 = a + ratio2 * L
    f1 = f(x1)
    f2 = f(x2)
    
    # Imprimir datos de la iteración y la decisión tomada
    print(f"Iteración {k}: Intervalo = [{a:.3f}, {b:.3f}]")
    print(f"  x₁ = {x1:.3f}  -->  f(x₁) = {f1:.4f}")
    print(f"  x₂ = {x2:.3f}  -->  f(x₂) = {f2:.4f}")
    
    if f1 > f2:
        print(f"  Decisión: f(x₁) ({f1:.4f}) > f(x₂) ({f2:.4f}) → descartar la parte izquierda; actualizar a = x₁\n")
        a = x1
        decision = "Descartar izquierda (a = x₁)"
    else:
        print(f"  Decisión: f(x₁) ({f1:.4f}) <= f(x₂) ({f2:.4f}) → descartar la parte derecha; actualizar b = x₂\n")
        b = x2
        decision = "Descartar derecha (b = x₂)"
    
    # Guardar datos de la iteración
    iterations.append({
        'iter': k,
        'interval': (a, b),
        'x1': x1,
        'x2': x2,
        'f(x1)': f1,
        'f(x2)': f2,
        'decision': decision
    })

# Estimación final del mínimo: se toma el punto medio del intervalo final
x_min_est = (a + b) / 2

# --- Graficar el proceso de optimización ---
x_vals = np.linspace(-0.5, 4.5, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x)", color="navy")

# Dibujar los límites del intervalo inicial en rojo
plt.axvline(x=0, color="red", linestyle=":", linewidth=2, label="Intervalo Inicial")
plt.axvline(x=4, color="red", linestyle=":", linewidth=2)

# Colores para cada iteración
colors = ["orange", "magenta", "green"]

for i, data in enumerate(iterations):
    int_a, int_b = data['interval']
    x1_val = data['x1']
    x2_val = data['x2']
    
    # Dibujar líneas verticales para los límites del intervalo de la iteración
    plt.axvline(x=int_a, color=colors[i], linestyle="--", linewidth=2, label=f"Iter. {data['iter']} - Límite Izq")
    plt.axvline(x=int_b, color=colors[i], linestyle="--", linewidth=2, label=f"Iter. {data['iter']} - Límite Der")
    
    # Graficar los puntos de evaluación
    plt.scatter([x1_val, x2_val], [f(x1_val), f(x2_val)], color=colors[i], s=70, zorder=5)
    # Etiquetas para los puntos (rotadas y en negro para mejor legibilidad)
    plt.text(x1_val, f(x1_val) + 0.2, f"x₁={x1_val:.3f}", color="black", rotation=45, ha="center", va="bottom", fontsize=10)
    plt.text(x2_val, f(x2_val) + 0.2, f"x₂={x2_val:.3f}", color="black", rotation=45, ha="center", va="bottom", fontsize=10)

# Sombrear el intervalo final obtenido tras las iteraciones
plt.axvspan(a, b, color='lightblue', alpha=0.3, label=f"Intervalo Final [{a:.3f}, {b:.3f}]")

# Mostrar la estimación final del mínimo en la gráfica
plt.scatter([x_min_est], [f(x_min_est)], color="black", s=100, zorder=6, marker='x', label=f"Est. mínimo = {x_min_est:.3f}")

plt.title("Método de Búsqueda por Fibonacci: Optimización en 3 Iteraciones")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Resumen Numérico ---
print("Resumen Numérico del Método de Fibonacci:")
for data in iterations:
    print(f"Iteración {data['iter']}:")
    print(f"  Intervalo: [{data['interval'][0]:.3f}, {data['interval'][1]:.3f}]")
    print(f"  x₁ = {data['x1']:.3f}, f(x₁) = {data['f(x1)']:.4f}")
    print(f"  x₂ = {data['x2']:.3f}, f(x₂) = {data['f(x2)']:.4f}")
    print(f"  Decisión: {data['decision']}")
    print("---------------------------------------------------")
print(f"Estimación final del mínimo: x = {x_min_est:.3f}, f(x) = {f(x_min_est):.4f}")
