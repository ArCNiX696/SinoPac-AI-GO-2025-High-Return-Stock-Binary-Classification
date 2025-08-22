import os

file_path = "/home/user/documents/report.pdf"
file_name = os.path.basename(file_path)
print(f"Filename: {file_name}") # Output: Filename: report.pdf


file_path = "/home/user/documents/report.pdf"
directory_name = os.path.dirname(file_path)
print(f"Directory: {directory_name}") # Output: Directory: /home/user/documents