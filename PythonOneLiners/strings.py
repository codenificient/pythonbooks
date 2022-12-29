# python strings

singleString = 'yes'
doubleQuote = "We Kam"
multiQuote = """What this is"""

## Most important String Methods
y = "    this is lazy\t\n  "
print(y.strip()) # remove whitespace: 'this is lazy'

# Lower Case
print("Dr.Dre".lower())

# UpperCase
print("attention".upper())


# finding substrings
print("smartphone".startswith("smart")) # if a string starts with a substring

print("smartphone".endswith("phone"))  # if a string ends with a substring

print("another".find("other"))  # returns the initial index where substring is found

print("ear" in "earth") # returns a boolean

# replacing substrings
print("cheat".replace("ch", "shr")) # replaces a string with another, whether it makes sense or not

# joining strings
print("*".join(["F", "B", "I"])) # adds a star in between the provided letters


print(",".join(["W", "T", "H", "?"]))

# Measuring strings
print("String Length: ",len("rasdahmgladas"))

class returnQuotes:
    print(singleString)
    print(doubleQuote)
    print(multiQuote)

returnQuotes()

