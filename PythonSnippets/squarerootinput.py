import cmath

num = 1+2j

num1 = eval(input('Enter a number: '))

num_sqrt = cmath.sqrt(num)
num1_sqrt = cmath.sqrt(num1)

print('The square root of {0} is {1:0.2f} + {2:0.2f}j'.format(num, num_sqrt.real,num_sqrt.imag))
print('The square root of {0} is {1:.2f}'.format(num1, num1_sqrt))
