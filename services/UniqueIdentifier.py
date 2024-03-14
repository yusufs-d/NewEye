import random
import string

def generate_unique_identifier():
    letters = string.ascii_uppercase
    digits = string.digits
    return ''.join(random.choice(letters) for _ in range(2)) + \
           ''.join(random.choice(digits) for _ in range(3))

# Example usage:
identifier = generate_unique_identifier()
print(identifier)
