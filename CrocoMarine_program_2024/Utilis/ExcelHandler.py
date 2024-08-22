import pandas as pd

class ExcelHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []

    def add_data(self, row) -> None:
        """
        Add a row of data to the internal data list.

        Args:
            row (list): Row of data to add.
        """
        self.data.append(row)

    def save(self )-> None:
        """
        Save the internal data to the Excel file.
        """
        df = pd.DataFrame(self.data, columns=['Current Frame', 'X Bound, Left', 'X Bound, Right', 'Y Bound, Upper', 'Y Bound, Lower'])
        df.to_excel(self.file_path, index=False)