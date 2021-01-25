# coverting bit depth from 32 to 24
from PIL import Image

fault = "IR"

for i in range(1,101):
    im = Image.open("D:/Atik/pythonScripts/WCNN/Dataset/cnnData/cwtNeeemd/fakeNtrain/{}/fakeFig_{}.png".format(fault, i)).convert('RGB')
    im.save("D:/Atik/pythonScripts/WCNN/Dataset/cnnData/cwtNeeemd/fakeNtrain/{}/fakeFig_{}.png".format(fault, i))