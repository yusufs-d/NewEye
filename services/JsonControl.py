import json
import os
# Important reminder!! Do not make any change on the 'control.json' file by hand.


# Class for the control.json file operations.
# Purpose of the control.json : This json file stores just one value called 'control'
# This control value indicates the status of the program.
# 0 means the program did not take any input from user
# 1 means the program has taken before extraction inputs
# control > 1 indicates number of scenes completed.
# For example: The user typed all of the inputs and checked 12 scene and terminated the program.
# In this situation, the control value would be 12. If the user start the program again the program will
# be continued from scene 12. 
class JsonControl:

    def __init__(self):

        currentDir = os.getcwd()
        self.jsonDir = os.path.join(currentDir,"services","control.json")


        # Opens json file and stores as dictionary in the 'data' variable
        with open(self.jsonDir,'r') as jfile:
            self.data = json.load(jfile)             
        

    def check_control(self):
        """Returns control value as integer"""
        return self.data["control"]
    
    def check_objectCounter(self):
        """Returns objectCounter value as integer"""
        return self.data["objectCounter"]

    def check_rowCounter(self):
        """Returns rowCounter value as integer"""
        return self.data["rowCounter"]

    def check_imageCounter(self):
        """Returns imageCounter value as integer"""
        return self.data["imageCounter"]
    
    def checkExcelPath(self):
        return self.data["excelPath"]
    
    def setExcelPath(self,path):
        self.data["excelPath"] = path
        
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)
    
    def increase_control(self):
        """Increases the control value"""
        self.data["control"] += 1
        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)

    def increase_objectCounter(self):
        """Increases the objectCounter value"""
        self.data["objectCounter"] += 1

        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)

    def increase_rowCounter(self):
        """Increases the rowCounter value"""
        self.data["rowCounter"] += 1

        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)

    def increase_imageCounter(self):
        """Increases the imageCounter value"""
        self.data["imageCounter"] += 1

        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)

    def decrease_control(self):
        """Decreases the control value"""
        if self.check_control() > 0 :
            self.data["control"] -= 1

        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile) 

    def decrease_objectCounter(self):
        """Decreases the objectCounter value"""
        if self.check_objectCounter() > 0:
            self.data["objectCounter"] -= 1

        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile) 

    def decrease_rowCounter(self):
        """Decreases the rowCounter value"""
        if self.check_rowCounter() > 2:
            self.data["rowCounter"] -= 1

        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile) 

    def decrease_imageCounter(self):
        """Decreases the imageCounter value"""
        if self.check_imageCounter() > 0:
            self.data["imageCounter"] -= 1

        """Finally, overwrites the value to control.json file"""
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile) 
    

    def reset_control(self):
        """
            Sets 0 to control value. Be careful to use this function!!\n
            You may lose your progress
        """
        self.data["control"] = 0

        #Finally, overwrites the control value to control.json file
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)

        
    def reset_objectCounter(self):
        """
            Sets 0 to objectCounter value. Be careful to use this function!!\n
            You may lose your progress
        """
        self.data["objectCounter"] = 0

        #Finally, overwrites the control value to control.json file
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)

        
    def reset_rowCounter(self):
        """
            Sets 2 to rowCounter value. Be careful to use this function!!\n
            You may lose your progress
        """
        self.data["rowCounter"] = 2

        #Finally, overwrites the control value to control.json file
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)

        
    def reset_imageCounter(self):
        """
            Sets 0 to imageCounter value. Be careful to use this function!!\n
            You may lose your progress
        """
        self.data["imageCounter"] = 0

        #Finally, overwrites the control value to control.json file
        with open(self.jsonDir,'w') as jfile:
            json.dump(self.data,jfile)



    