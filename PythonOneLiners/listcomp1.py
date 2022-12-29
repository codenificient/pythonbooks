# using list comprehensions in several lines

squares = []
for i in range(10):
    squares.append(i**2)

print(squares)

# doing the same thing in one line

print([i**2 for i in range(12)])
