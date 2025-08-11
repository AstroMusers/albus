r = 11

for i in range(r):
    for j in range(r):
        for k in range(r):
            ID = str(i).zfill(2) + str(j).zfill(2) + str(k).zfill(2)
            print(f"Processing ID: {ID}")