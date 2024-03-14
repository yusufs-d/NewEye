import os
import openpyxl # Module  for the excel operations

class ExcelControl:
    """
    This class is used for the export to excel process.
    It exports the scene names to the excel file

    """
    
    def __init__(self):


        # Workbook() takes one, non-optional, argument
        # which is the filename that we want to create.
        self.workbook = openpyxl.Workbook()
        self.path = "object-distance.xlsx"
        self.table_row = 2

        if os.path.exists("object-distance.xlsx"): # Check the excel file if exists
            self.path = "object-distance-2.xlsx"

        # The workbook object is then used to add new
        # worksheet via the add_worksheet() method.
        self.worksheet = self.workbook.active # Select worksheet as default
        self.worksheet.title = "Objects-Distances" # Set worksheet title

        # It creates Scene Table with Scene Name and Tag Columns
        self.create_table()

    
    def create_table(self):
        
        # Use the worksheet object to set values
        self.worksheet["A1"].value = 'ObjectName'
        self.worksheet["B1"].value = 'Size'
        self.worksheet["C1"].value = 'Region'
        self.worksheet["D1"].value = 'isClose'
        self.workbook.save(self.path) # Finally, save the excel file

    
    def add_info_to_table(self,cell,info):
        # Args: filename-> Filename to add excel table
        self.worksheet[f"{cell}{self.table_row}"].value = info # Set cell value
        self.workbook.save(self.path) # Finally, save the excel file
        
    def increase_row(self):
        self.table_row += 1

    def decrease_row(self):
        if self.table_row > 2:
            self.table_row -= 1
        else:
            return
        

