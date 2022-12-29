x,y = True, False

print(x or y)
print(x and y)

print(not y)
print(not x)

print(x and not y)
print(not x and y or x)


# if conditions evaluating to False

if None or 0 or 0.0 or '' or [] or {} or set():
    print("Dead Code")