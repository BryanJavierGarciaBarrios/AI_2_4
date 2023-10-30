pip install pgmpy

from pgmpy.models import DecisionNetwork

# Crear una instancia de la red de decisión
decision_network = DecisionNetwork()

# Definir las variables de decisión y sus valores
decision_network.add_node("Lluvia", ["Si", "No"])
decision_network.add_node("Traje", ["Si", "No"])
decision_network.add_node("Llevar_Paraguas", ["Si", "No"])

# Definir la relación entre las variables de decisión
decision_network.add_edge("Lluvia", "Llevar_Paraguas")
decision_network.add_edge("Traje", "Llevar_Paraguas")

# Asignar una tabla de decisión a la variable de decisión "Llevar_Paraguas"
decision_network.add_cpds(
    "Llevar_Paraguas",
    {
        "Lluvia": {"Si": 0.9, "No": 0.2},
        "Traje": {"Si": 0.7, "No": 0.1},
        "Llevar_Paraguas": {"Si": 1.0, "No": 0.0},
    }
)

# Verificar la consistencia de la red de decisión
assert decision_network.check_model()

# Imprimir la tabla de decisión de "Llevar_Paraguas"
print("Tabla de decisión de 'Llevar_Paraguas':")
print(decision_network.get_cpds("Llevar_Paraguas"))
