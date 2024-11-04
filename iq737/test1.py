sval = 117077860
freq = str(sval)
length = len(freq)
if length >= 9:
    print(freq[0:3]+'.'+freq[3:6]+'.'+freq[6:9])
    print(length)
elif length >= 8:
    print(freq[0:2]+'.'+freq[2:5]+'.'+freq[5:8])
    print(length)
elif length >=7:
    print(freq[0:1]+'.'+freq[1:4]+'.'+freq[4:7])
    print(length)