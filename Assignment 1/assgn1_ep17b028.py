import sys

#Accept name of file as command line
netlist_filename = sys.argv[1]

#Implementation of opening netlist file with exception handling to display error message in case of incorrect filename
try:
  f = open(netlist_filename, "r")
except IOError or NameError:
  print ('There is no file named', netlist_filename)

lines = f.readlines()
f.close()

#Print the original netlist file
print("The original netlist file is\n")
for l in lines:
    print("%s" %l)

#Initialise variables for indices of .circuit and .end. We assign values such that if .circuit or .end are not found, or if they appear in reverse order, we reject
index_dotcircuit = -1
index_dotend = 1

desired_cols = 4

#Empty list which will be appended later
section = []

#For loop to identify the locations of .circuit and .end
for dummy in range(len(lines)):
    if(lines[dummy] == '.circuit\n'):
        index_dotcircuit = dummy
# If .end is at end of file the array entry in lines is .end but if there is junk after .end then it will have .end\n
    if(lines[dummy] == '.end\n' or lines[dummy] == '.end'):
        index_dotend = dummy

#Here we are checking if the file has .circuit and .end, and whether they appear in order
if((index_dotcircuit*index_dotend) < 0 or index_dotcircuit>=index_dotend):
    print("This is a malformed input file")
else:

    #Create an array for storing in reverse order
    rows, cols = (index_dotend-1-index_dotcircuit, desired_cols)
    parse = [None]*rows
    for i in range(rows):
        parse[i] = [None]*cols
    
    for i in range(index_dotcircuit+1,index_dotend):
        section.append(lines[i])
        #First we bsplit the string by spaces. It is possible that there may be comments but they will occur after the fourth column
        temp = lines[i].split(" ")
        #If there are no comments the fourth column will contain \n at the end. We do not want that
        temp2 = temp[cols-1].split("\n")
        temp[cols-1] = temp2[0]
        
        for j in range(cols):
            parse[i-index_dotcircuit-1][j] = temp[j]
             
    print("\nThe section within .circuit and .end is")
    for w in section:
        print("%s" %w)
    print("\nParsed is")
    for h in range(rows):
        for k in range(cols):
            print(parse[h][k], end=" ")
        print("\n")
