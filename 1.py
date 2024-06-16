# Open the text file for reading
file_path = 'D:/User/Desktop/03-11-09-50-26.log'
with open(file_path, 'r') as file:
    lines = file.readlines()
 # Filter out lines that do not contain "Loss"
loss_lines = [line for line in lines if 'Loss' in line]
 # Write the filtered lines back to the file
with open(file_path, 'w') as file:
    for line in loss_lines:
        file.write(line)