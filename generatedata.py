# Python file to generate some data
# buckets will be of size 100...
import random, string
# read inputs..
rows, buckets = map(int, input().split(" "))
if rows % 31:
    print("Please make rows a multiple of 31...")
    exit(0)
letters = string.ascii_lowercase
print(rows, buckets)
for i in range(rows):
    # print, number (ID), string (name/value), actual value...
    print(i, ''.join(random.choice(letters) for i in range(10)), random.randint(0, buckets*100))