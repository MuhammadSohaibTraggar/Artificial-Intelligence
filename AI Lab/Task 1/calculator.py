def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    result = 0
    for i in range(abs(b)):
        result += a
    if b < 0:
        result = -result
    return result

def divide(a, b):
    if b == 0:
        return "Cannot divide by zero!"
    quotient = 0
    remainder = abs(a)
    
    while remainder >= abs(b):
        remainder -= abs(b)
        quotient += 1
    
    if (a < 0 and b > 0) or (a > 0 and b < 0):
        quotient = -quotient
    return quotient
def modulus(a, b):
    if b == 0:
        return "Cannot modulus by zero!"
    remainder = abs(a)
    while remainder >= abs(b):
        remainder -= abs(b)
    if a < 0:
        remainder = -remainder
    return remainder

def dynamic_calculator():
    str1 = input("Welcome to  the dynamic calculator. To Start press enter --->")
    num1 = int(input("Enter the first number: "))
    operator = input("Enter operator (+, -, *, /, %): ")
    num2 = int(input("Enter the second number: "))
    if operator == '+':
        result = add(num1, num2)
    elif operator == '-':
        result = subtract(num1, num2)
    elif operator == '*':
        result = multiply(num1, num2)
    elif operator == '/':
        result = divide(num1, num2)
    elif operator == '%':
        result = modulus(num1, num2)
    else:
        result = "Invalid operator!"
    print("Result:", result)
dynamic_calculator()
