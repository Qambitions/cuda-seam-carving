
from PIL import Image
from numpy import asarray
import sys
img = Image.open(sys.argv[1])

data = asarray(img)

with open(sys.argv[2], "w") as o:
    o.write('P3 \n')
    o.write(f'{data.shape[1]}\n{data.shape[0]}\n255\n')
    for x in data:
        for y in x:
            for z in y:
                o.write(f"{z}\n")