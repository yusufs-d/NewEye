import os
import shutil
from services.JsonControl import JsonControl

def reset():
        
        jsonObj = JsonControl()

        jsonObj.reset_control()
        jsonObj.reset_imageCounter()
        jsonObj.reset_objectCounter()
        jsonObj.reset_rowCounter()
        jsonObj.setExcelPath("object-distance.xlsx")

        if os.path.exists(os.path.join("services","ScreenShots")):
            try:
                shutil.rmtree(os.path.join("services","ScreenShots"))
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))

        if os.path.exists(os.path.join("services","object-distance.xlsx")):
            os.remove(os.path.join("services","object-distance.xlsx"))

        if os.path.exists(os.path.join("services","info.txt")):
            os.remove(os.path.join("services",'info.txt'))