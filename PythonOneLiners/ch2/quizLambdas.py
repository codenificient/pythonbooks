## Data
txt = ['lambda functions are anonymous functions.',
'anonymous functions dont have a name.',
'functions are objects in Python.']

print([(True, line) if 'anonymous' in line else (False, line) for line in txt])
