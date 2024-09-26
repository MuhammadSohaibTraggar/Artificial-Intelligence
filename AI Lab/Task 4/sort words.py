Words=["Sohaib","Abbas","Huzaifa","Ahmad","Hassan"]
for i in range(0,len(Words)):
    for j in range(0,len(Words)):
        if Words[j]> Words[i]:
            temp = Words[i]
            Words[i]=Words[j]
            Words[j]= temp
print(Words)