pip install pgmpy

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference import DBNInference

# Crear un modelo de red bayesiana dinámica
model = DBN()

# Definir las variables en el día 0
cloudy_0 = 'Cloudy0'
temp_0 = 'Temp0'

# Definir las variables en el día 1
cloudy_1 = 'Cloudy1'
temp_1 = 'Temp1'

# Definir las variables en el día 2
cloudy_2 = 'Cloudy2'
temp_2 = 'Temp2'

# Agregar variables a la red
model.add_edge(cloudy_0, cloudy_1)
model.add_edge(cloudy_1, cloudy_2)
model.add_edge(temp_0, temp_1)
model.add_edge(temp_1, temp_2)

# Definir las distribuciones de probabilidad condicional (CPD)
cpd_cloudy_0 = TabularCPD(variable=cloudy_0, variable_card=2, values=[[0.5], [0.5]])
cpd_temp_0 = TabularCPD(variable=temp_0, variable_card=2, values=[[0.2], [0.8]])

cpd_cloudy_1 = TabularCPD(variable=cloudy_1, variable_card=2, values=[[0.7, 0.3], [0.3, 0.7]],
                         evidence=[cloudy_0], evidence_card=[2])
cpd_temp_1 = TabularCPD(variable=temp_1, variable_card=2, values=[[0.4, 0.7], [0.6, 0.3]],
                       evidence=[temp_0], evidence_card=[2])

cpd_cloudy_2 = TabularCPD(variable=cloudy_2, variable_card=2, values=[[0.8, 0.2], [0.2, 0.8]],
                         evidence=[cloudy_1], evidence_card=[2])
cpd_temp_2 = TabularCPD(variable=temp_2, variable_card=2, values=[[0.1, 0.4], [0.9, 0.6]],
                       evidence=[temp_1], evidence_card=[2])

# Agregar las CPD al modelo
model.add_cpds(cpd_cloudy_0, cpd_temp_0, cpd_cloudy_1, cpd_temp_1, cpd_cloudy_2, cpd_temp_2)

# Verificar si el modelo es válido
model.check_model()

# Realizar inferencia en el modelo
inference = VariableElimination(model)
query_result = inference.query(variables=[cloudy_0, cloudy_1, cloudy_2],
                               evidence={temp_0: 1, temp_1: 0, temp_2: 1})
print(query_result)

# Realizar inferencia en la secuencia de tiempo
dbn_inference = DBNInference(model)
time_slice_0 = (cloudy_0, temp_0)
time_slice_1 = (cloudy_1, temp_1)
time_slice_2 = (cloudy_2, temp_2)
evidence = {time_slice_0: 0, time_slice_1: 1}
query_result = dbn_inference.query(time_slice_2, evidence)
print(query_result)
