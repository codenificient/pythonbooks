## Data
employees = {'Alice' : 100000,
'Bob' : 99817,
'Carol' : 122908,
'Frank' : 88123,
'Eve' : 93121}

top_earners = [(x,y) for x, y in employees.items() if y >= 100000]
print(top_earners)