### Objective Function

Let $x_{(l,k,j)}$ be the number of tickets sold for OD pair $l$, departure time $k$, and ticket type $j$ (where $j = f$ for Eco_flex, $j = l$ for Eco_lite). Let $x_o[l]$ be the number of outside option passengers for OD $l$.

The objective is to maximize total revenue:

$$
\max \quad \sum_{(l,k,j)} \text{avg\_price}[(l,k,j)] \cdot x_{(l,k,j)}
$$

where the sum is over all ticket options for the specified flights.

### Constraints

#### 1. Capacity Constraints

For each flight (OD pair $l$, departure time $k$), the total capacity consumed by ticket sales cannot exceed the flight capacity:

$$
2 \cdot x_{(l,k,f)} + 1 \cdot x_{(l,k,l)} \leq \text{flight\_capacity}
$$

#### 2. Balance Constraints

For each OD pair $l$:

$$
\text{ratio\_0\_list}[l] \cdot x_o[l] + \sum_{k} \sum_{j} \text{ratio\_list}[(l,k,j)] \cdot x_{(l,k,j)} = \text{avg\_pax}[l]
$$

where the sum is over all flights and ticket types for OD $l$.

#### 3. Scale Constraints

For each ticket option \((l,k,j)\):

$$
\frac{x_{(l,k,j)}}{\text{value\_list}[(l,k,j)]} - \frac{x_o[l]}{\text{value\_0\_list}[l]} \leq 0
$$

#### 4. Nonnegativity Constraints

$$
x_{(l,k,j)} \geq 0, \quad x_o[l] \geq 0
$$

### Retrieved Information

```python
avg_price = {
    '(AC,23:00,f)': '1269.77', '(AC,23:00,l)': '485.58',
    '(AC,19:05,f)': '1285.64', '(AC,19:05,l)': '499.36',
    '(BA,15:40,f)': '1139.8', '(BA,15:40,l)': '441.86',
    '(BA,18:50,f)': '1138.59', '(BA,18:50,l)': '429.78',
    '(CA,16:55,f)': '1285.73', '(CA,16:55,l)': '527.21',
    '(BA,09:05,f)': '1123.69', '(BA,09:05,l)': '419.78',
    '(CA,07:40,f)': '1285.82', '(CA,07:40,l)': '558.55'
}
value_list = {
    '(AC,23:00,f)': 1.947126283, '(AC,23:00,l)': 0.574427934,
    '(AC,19:05,f)': 2.329088053, '(AC,19:05,l)': 0.476091969,
    '(BA,15:40,f)': 0.897742508, '(BA,15:40,l)': 0.506369319,
    '(BA,18:50,f)': 1.788650781, '(BA,18:50,l)': 0.756699229,
    '(CA,16:55,f)': 1.329173046, '(CA,16:55,l)': 0.769156234,
    '(BA,09:05,f)': 2.600413693, '(BA,09:05,l)': 1,
    '(CA,07:40,f)': 0.865902564, '(CA,07:40,l)': 0.666898117
}
ratio_list = {
    '(AC,23:00,f)': 1.0, '(AC,23:00,l)': 1.0,
    '(AC,19:05,f)': 1.0, '(AC,19:05,l)': 1.0,
    '(BA,15:40,f)': -2.932359218, '(BA,15:40,l)': 22.58777776,
    '(BA,18:50,f)': 8.734367009, '(BA,18:50,l)': -161.8249472,
    '(CA,16:55,f)': 1.0, '(CA,16:55,l)': 1.0,
    '(BA,09:05,f)': -9.210371325, '(BA,09:05,l)': -3.045572418,
    '(CA,07:40,f)': 1.0, '(CA,07:40,l)': 1.0
}
value_0_list = {'AC': 0.024033093, 'BA': 0.133469692, 'CA': 0.126816434}
ratio_0_list = {'AC': 1.0, 'BA': 0.012916147, 'CA': 1.0}
avg_pax = {'CA': '4807.43', 'BA': '33210.71', 'AC': '4812.5'}
flight_capacity = 187
```

### Generated Code

```python
import gurobipy as gp
from gurobipy import GRB

avg_price = {
    '(AC,23:00,f)': 1269.77, '(AC,23:00,l)': 485.58,
    '(AC,19:05,f)': 1285.64, '(AC,19:05,l)': 499.36,
    '(BA,15:40,f)': 1139.8, '(BA,15:40,l)': 441.86,
    '(BA,18:50,f)': 1138.59, '(BA,18:50,l)': 429.78,
    '(CA,16:55,f)': 1285.73, '(CA,16:55,l)': 527.21,
    '(BA,09:05,f)': 1123.69, '(BA,09:05,l)': 419.78,
    '(CA,07:40,f)': 1285.82, '(CA,07:40,l)': 558.55
}
value_list = {
    '(AC,23:00,f)': 1.947126283, '(AC,23:00,l)': 0.574427934,
    '(AC,19:05,f)': 2.329088053, '(AC,19:05,l)': 0.476091969,
    '(BA,15:40,f)': 0.897742508, '(BA,15:40,l)': 0.506369319,
    '(BA,18:50,f)': 1.788650781, '(BA,18:50,l)': 0.756699229,
    '(CA,16:55,f)': 1.329173046, '(CA,16:55,l)': 0.769156234,
    '(BA,09:05,f)': 2.600413693, '(BA,09:05,l)': 1,
    '(CA,07:40,f)': 0.865902564, '(CA,07:40,l)': 0.666898117
}
ratio_list = {
    '(AC,23:00,f)': 1.0, '(AC,23:00,l)': 1.0,
    '(AC,19:05,f)': 1.0, '(AC,19:05,l)': 1.0,
    '(BA,15:40,f)': -2.932359218, '(BA,15:40,l)': 22.58777776,
    '(BA,18:50,f)': 8.734367009, '(BA,18:50,l)': -161.8249472,
    '(CA,16:55,f)': 1.0, '(CA,16:55,l)': 1.0,
    '(BA,09:05,f)': -9.210371325, '(BA,09:05,l)': -3.045572418,
    '(CA,07:40,f)': 1.0, '(CA,07:40,l)': 1.0
}
value_0_list = {'AC': 0.024033093, 'BA': 0.133469692, 'CA': 0.126816434}
ratio_0_list = {'AC': 1.0, 'BA': 0.012916147, 'CA': 1.0}
avg_pax = {'CA': 4807.43, 'BA': 33210.71, 'AC': 4812.5}
flight_capacity = 187

model = gp.Model("sales_based_lp")
x = model.addVars(avg_price.keys(), lb=0, name="x")
x_o = model.addVars(value_0_list.keys(), lb=0, name="x_o")

# Objective
model.setObjective(gp.quicksum(avg_price[key] * x[key] for key in avg_price.keys()), GRB.MAXIMIZE)

# Capacity constraints
flight_pairs = [
    ('AC', '23:00'), ('AC', '19:05'),
    ('BA', '15:40'), ('BA', '18:50'), ('BA', '09:05'),
    ('CA', '16:55'), ('CA', '07:40')
]
for l, k in flight_pairs:
    key_f = f'({l},{k},f)'
    key_l = f'({l},{k},l)'
    if key_f in x and key_l in x:
        model.addConstr(2 * x[key_f] + 1 * x[key_l] <= flight_capacity, name=f"capacity_{l}_{k}")

# Balance constraints
for l in ratio_0_list.keys():
    temp = gp.LinExpr()
    for key in ratio_list.keys():
        if key.startswith(f'({l},'):
            temp += ratio_list[key] * x[key]
    model.addConstr(ratio_0_list[l] * x_o[l] + temp == avg_pax[l], name=f"balance_{l}")

# Scale constraints
for key in value_list.keys():
    l = key[1:3]  # e.g., 'AC'
    l = l.replace(',', '')
    for od in value_0_list.keys():
        if od in key:
            model.addConstr(x[key] / value_list[key] - x_o[od] / value_0_list[od] <= 0, name=f"scale_{key}")

model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(v.varName, v.x)
    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found.")
```

---

**Note:**  
- The variables $x_{(l,k,j)}$ correspond to each ticket type for each specified flight.
- The capacity constraint uses 2 units for Eco_flex and 1 unit for Eco_lite, as specified.
- All required parameters are included above.
- The code is ready to run with Gurobi and will solve the SBLP for the specified flights and constraints.