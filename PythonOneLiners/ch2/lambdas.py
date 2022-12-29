## Data
txt = ['lambda functions are anonymous functions.',
'anonymous functions dont have a name.',
'functions are objects in Python.']

# One liners
mark = map(lambda  s: (True, s) if 'anonymous' in s else (False, s), txt)

print(list(mark))