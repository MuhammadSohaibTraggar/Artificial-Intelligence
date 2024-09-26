
string = "Hello, I am Muhammad Sohaib! I love coding, and I am learning Python  at the Superior University."
print("The original string is:", "\n", string)
punctuation_list = ''''!()-[]{};:'"\,<>./?@#$%^&*_~'''
for char in punctuation_list:
    string = string.replace(char, "")
print("\nThe string after removing punctuation will be like: \n", string)
