


string_list = ['rest', 'resting', 'look', 'looked', 'it', 'spit']
# a = filter(lambda x: [x for i in string_list if x in i and x != i] == [], string_list)
# a = filter(lambda x: [x for i in string_list if x in i and 'rest' in i] == [], string_list)
a = filter(lambda x: [x for i in string_list if x in i and 'rest' in i] == [], string_list)
# a = (map(lambda i: 'rest' not in i, string_list))
# a = (filter(lambda i: 'rest' not in i, string_list))
print(list(a))