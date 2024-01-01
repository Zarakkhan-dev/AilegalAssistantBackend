sentence = "hello this is 299 uis here"

for i in range(255, 300):
    if str(i) in sentence:
        print(f"Number {i} found in the sentence")
    else:
        print(f"Number {i} not found in the sentence")
