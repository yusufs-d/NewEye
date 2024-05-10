from services.Speak import buildSounds


sounds = []
with open("/Users/yusufs/Desktop/NewEyeRepo/NewEye/sample_model/labelmap.txt","r") as objects:
    for obj in objects:
        obj = obj.strip()
        if not obj.startswith("???"):
            sounds.append((f"{obj} 11 o'clock",f"{obj}11"))
            sounds.append((f"{obj} 12 o'clock",f"{obj}12"))
            sounds.append((f"{obj} 1 o'clock",f"{obj}1"))

buildSounds(sounds,1.5)