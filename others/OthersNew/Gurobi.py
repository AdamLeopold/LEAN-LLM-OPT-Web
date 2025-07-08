import gurobipy as gp
from gurobipy import GRB

# Data
p = {
    '(AB,11:20,f)': 1140.3, '(AB,11:20,l)': 429.26,
    '(AB,17:05,f)': 1145.54, '(AB,17:05,l)': 498.89,
    '(AB,19:10,f)': 1138.14, '(AB,19:10,l)': 450.8,
    '(AB,22:45,f)': 1134.68, '(AB,22:45,l)': 428.56
}
v = {
    '(CB,7:40,f)': 0, '(CB,7:40,l)': 0, '(CB,10:45,f)': 0, '(CB,10:45,l)': 0,
    '(AB,11:20,f)': 0.342687, '(AB,11:20,l)': 0.24677,
    '(CB,12:35,f)': 0, '(CB,12:35,l)': 0,
    '(AB,12:40,f)': 0.063743, '(AB,12:40,l)': 0.041724,
    '(AB,12:55,f)': 0.063743, '(AB,12:55,l)': 0.041724,
    '(AB,13:55,f)': 0.063743, '(AB,13:55,l)': 0.041724,
    '(CB,14:15,f)': 0, '(CB,14:15,l)': 0,
    '(AB,16:05,f)': 0.063743, '(AB,16:05,l)': 0.041724,
    '(CB,16:55,f)': 0, '(CB,16:55,l)': 0,
    '(AB,17:05,f)': 0.063743, '(AB,17:05,l)': 0.041724,
    '(CB,17:40,f)': 0, '(CB,17:40,l)': 0,
    '(AB,18:00,f)': 0.193098, '(AB,18:00,l)': 0.112429,
    '(CB,18:30,f)': 0, '(CB,18:30,l)': 0,
    '(AB,19:10,f)': 0.193098, '(AB,19:10,l)': 0.112429,
    '(AB,22:45,f)': 0.139137252, '(AB,22:45,l)': 0.299098
}
r = {
    '(CB,7:40,f)': 0, '(CB,7:40,l)': 0, '(CB,10:45,f)': 0, '(CB,10:45,l)': 0,
    '(AB,11:20,f)': 0.67, '(AB,11:20,l)': 1.0,
    '(CB,12:35,f)': 0, '(CB,12:35,l)': 0,
    '(AB,12:40,f)': 1.0, '(AB,12:40,l)': 1.0,
    '(AB,12:55,f)': 1.0, '(AB,12:55,l)': 1.0,
    '(AB,13:55,f)': 1.0, '(AB,13:55,l)': 1.0,
    '(CB,14:15,f)': 0, '(CB,14:15,l)': 0,
    '(AB,16:05,f)': 1.0, '(AB,16:05,l)': 1.0,
    '(CB,16:55,f)': 0, '(CB,16:55,l)': 0,
    '(AB,17:05,f)': 1.0, '(AB,17:05,l)': 1.0,
    '(CB,17:40,f)': 0, '(CB,17:40,l)': 0,
    '(AB,18:00,f)': 0.74, '(AB,18:00,l)': 1.0,
    '(CB,18:30,f)': 0, '(CB,18:30,l)': 0,
    '(AB,19:10,f)': 0.74, '(AB,19:10,l)': 1.0,
    '(AB,22:45,f)': 0.0, '(AB,22:45,l)': 0.65
}
v_0 = {'CB': 0, 'AB': 1}
r_0 = {'CB': 0, 'AB': 1.41}
d = {'CB': 4807.43, 'AB': 38965.86}
c_f = 1.2
C = 187

# Set of all ticket options
ticket_keys = list(v.keys())
flight_keys = set((k.split(',')[0][1:], k.split(',')[1][:-1]) for k in ticket_keys)
od_keys = ['AB', 'CB']

model = gp.Model("sales_based_lp")

# Decision variables
x = model.addVars(ticket_keys, lb=0, name="x")
x_o = model.addVars(od_keys, lb=0, name="x_o")
y = model.addVars(flight_keys, vtype=GRB.BINARY, name="y")

# Objective
model.setObjective(gp.quicksum(p.get(k, 0) * x[k] for k in ticket_keys if k in p), GRB.MAXIMIZE)

# Flight selection constraint
model.addConstr(gp.quicksum(y[k] for k in flight_keys) == 3, name="select_3_flights")

# Capacity constraints
for (l, k) in flight_keys:
    model.addConstr(
        c_f * x.get(f"({l},{k},f)", 0) + x.get(f"({l},{k},l)", 0) <= C * y[(l, k)],
        name=f"capacity_{l}_{k}"
    )

# Balance constraints
for l in od_keys:
    model.addConstr(
        r_0[l] * x_o[l] + gp.quicksum(r.get(k, 0) * x[k] for k in ticket_keys if l in k) == d[l],
        name=f"balance_{l}"
    )

# Scale constraints
for k in ticket_keys:
    l = k.split(',')[0][1:]
    model.addConstr(
        x[k] * v_0[l] - x_o[l] * v[k] <= 0,
        name=f"scale_{k}"
    )

model.optimize()

if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(v.varName, v.x)
    print("Optimal objective value:", model.objVal)
else:
    print("No optimal solution found.")
