from ExcelControl import ExcelControl
import os
import cv2


class DefineDistance:

    def __init__(self, directory: str):
        self.directory = directory
        self.excel = ExcelControl()
        self.files = os.listdir(self.directory)
        self.files_sorted = sorted(self.files)

    def start(self):
        counter = 0
        while counter<len(self.files_sorted):
            image_name = self.files_sorted[counter]
            # Load the image
            image = cv2.imread(os.path.join(self.directory, image_name))

            # Check if the image was loaded successfully
            if image is None:
                print("Error: Unable to load image ", image_name)
                continue
            
            print("Press '1' for tag the object as not close\nPress '2' for tag the object as close\nPress 'd' for delete the object\nPress 'b' to go backt to the previous image")

            # Display the image
            cv2.imshow(image_name, image)

            with open("info.txt","r+") as file:
                for line in file:
                    if line.startswith("***"):
                        break
                    else:
                        print(line)
            # Wait for user input
            key = cv2.waitKey(0)

            # Check if the user pressed '1', '2', or '3'
            if key == ord('1'):
                counter+=1
            elif key == ord('2'):
                print("User pressed '2' for image:", image_name)
                counter+=1
            elif key == ord('d'):
                print("User pressed 'd' for image:", image_name)
                counter+=1
            elif key == ord('b'):
                print("User pressed 'b' for image:", image_name)
                if counter > 0:
                    counter-=1
                else:
                    print("There is no previous image!")

            elif key == ord('q'):
                print("Program terminated!")
                break
            else:
                print("Invalid input.")

        # Close the window
        cv2.destroyAllWindows()
  
        print("End of the images!")

    def start2(self):
        image_counter = 0
        object_counter = 0
        with open("info.txt","r") as file:
            lines = file.readlines()

        while object_counter < len(lines):
            line = lines[object_counter]
            if line.startswith("***"):
                cv2.destroyAllWindows()
                image_counter +=1 
                object_counter +=1
                continue
            else:
                line_splitted = line.split(",")
                object_name = line_splitted[0]
                rect_area = line_splitted[1]
                region = line_splitted[2]

                image_path = os.path.join(self.directory,self.files_sorted[image_counter])

                image = cv2.imread(image_path)

                if image is None:
                    print("Error! Unable to load image", image_path)
                    continue

                print("\n****************************************\nEnter an input\n1- Close\n2- Not Close\nb- Back to the previous object\nd- Delete the object\nq- Quit the program\n****************************************\n")


                cv2.imshow(self.files_sorted[image_counter],image)

                print(f"Enter an input for {object_name}:")

                key = cv2.waitKey(0)

                if key == ord("1"):
                    print(f"{object_name} is close")
                    self.excel.add_info_to_table("A",object_name.split(".")[0])
                    self.excel.add_info_to_table("B",str(rect_area))
                    self.excel.add_info_to_table("C",str(region))
                    self.excel.add_info_to_table("D","1")
                    self.excel.increase_row()
                    object_counter += 1
                elif key == ord("2"):
                    print(f"\n{object_name} is not close\n")
                    self.excel.add_info_to_table("A",object_name.split(".")[0])
                    self.excel.add_info_to_table("B",str(rect_area))
                    self.excel.add_info_to_table("C",str(region))
                    self.excel.add_info_to_table("D","2")
                    self.excel.increase_row()
                    object_counter += 1
                elif key == ord("b"):
                    if object_counter <= 0:
                        print("\nThere is no previous object!\n")
                        continue
                    else:
                        object_counter -= 1
                        self.excel.decrease_row()
                
                elif key == ord("d"):
                    object_counter+=1
                    print("\n Object deleted! Press b to cancel\n")
                    continue

                elif key == ord("q"):
                    print("\nProgram terminated!\n")
                    break
                
        print("\nEnd of the images!\n")
                
                

        





dd = DefineDistance("/Users/yusufs/Desktop/NewEye/ScreenShots")
dd.start2()
