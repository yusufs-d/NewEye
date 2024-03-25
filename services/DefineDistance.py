from services.ExcelControl import ExcelControl
from services.JsonControl import JsonControl
import os
import cv2


class DefineDistance:

    def __init__(self, directory: str):
        self.directory = directory
        self.excel = ExcelControl()
        self.files = os.listdir(self.directory)
        self.files_sorted = sorted(self.files, key=lambda filename: int(filename.split("_")[1].split(".")[0]))
        self.jsonObj = JsonControl() 
        


    def start_process(self):
        isBackward = False

        with open(os.path.join(os.getcwd(),"services","info.txt"),"r") as file:
            lines = file.readlines()


        while self.jsonObj.check_objectCounter() < len(lines):

            line = lines[self.jsonObj.check_objectCounter()]
            if line.startswith("***"):
                if isBackward:
                    self.jsonObj.decrease_imageCounter()
                    self.jsonObj.decrease_objectCounter()
                    cv2.destroyAllWindows()
                    continue
                else:
                    self.jsonObj.increase_imageCounter()
                    self.jsonObj.increase_objectCounter()
                    cv2.destroyAllWindows()
                    continue
            
            else:
                line_splitted = line.split(",")
                object_name = line_splitted[0]
                rect_area = line_splitted[1]
                region = line_splitted[2]

                try:
                    image_path = os.path.join(self.directory,self.files_sorted[self.jsonObj.check_imageCounter()])

                    image = cv2.imread(image_path)
                except:
                    print("End of the images!")
                    self.jsonObj.increase_control()
                    self.excel.close_excel()
                    break

                if image is None:
                    print("Error! Unable to load image", image_path)
                    continue

                print("\n****************************************\nEnter an input\n1- Close\n2- Not Close\nb- Back to the previous object\nd- Delete the object\nq- Quit the program\n****************************************\n")


                cv2.imshow(self.files_sorted[self.jsonObj.check_imageCounter()],image)

                print(f"Enter an input for {object_name}:")

                key = cv2.waitKey(0)

                if key == ord("1"):
                    print(f"{object_name} is close")
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"A",object_name.split(".")[0])
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"B",str(rect_area))
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"C",str(region))
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"D","1")
                    self.jsonObj.increase_rowCounter()
                    self.jsonObj.increase_objectCounter()
                    isBackward = False

                elif key == ord("2"):
                    print(f"\n{object_name} is not close\n")
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"A",object_name.split(".")[0])
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"B",str(rect_area))
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"C",str(region))
                    self.excel.add_info_to_table(self.jsonObj.check_rowCounter(),"D","2")
                    self.jsonObj.increase_rowCounter()
                    self.jsonObj.increase_objectCounter()
                    isBackward = False

                elif key == ord("b"):
                    if self.jsonObj.check_objectCounter() <= 0:
                        print("\nThere is no previous object!\n")
                        continue

                    else:
                        self.jsonObj.decrease_objectCounter()
                        if self.jsonObj.check_rowCounter() > 2:
                            self.jsonObj.decrease_rowCounter()
                        isBackward = True
                
                elif key == ord("d"):
                    self.jsonObj.increase_objectCounter()
                    print("\n Object deleted! Press b to cancel\n")
                    isBackward = False
                    continue

                elif key == ord("q"):
                    print("\nProgram terminated!\n")
                    isBackward = False
                    break
        if len(lines) == self.jsonObj.check_objectCounter():
            self.jsonObj.increase_control()


        self.excel.close_excel()
                
                

        



