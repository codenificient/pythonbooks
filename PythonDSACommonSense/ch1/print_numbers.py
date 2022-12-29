def print_numbers_version_one():
    number = 2
    while number <= 100:
        if number % 2 == 0:
            print(number)
        number += 1

def print_numbers_version_two():
    number = 2
    while number <= 100:
        print(number)
        number += 2

if __name__ == '__main__':
    print_numbers_version_one()
    print_numbers_version_two()