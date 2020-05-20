#This assignment takes in a .netlist file, parses it, and solves for node voltages for single frequency ac circuits and DC

#BRIEF DESCRIPTION OF THE ALGORITHM FOR SECOND ASSIGNMENT(SOLVING FOR NODE VOLTAGES),
#IS GIVEN ON LINES 88-107

#import sys
import numpy as np
import cmath
import sympy
import scipy.signal as sp
import matplotlib.pyplot as plt
sympy.init_printing(use_latex='mathjax')
s = sympy.symbols('s')

#From line 6 to line 84 is the first assignment to parse the list check conditions etc etc.
#Second Assignment starts from line 88

def string_split(word):
    return [char for char in word]

#Accept name of file as command line
netlist_filename = 'ckt2.netlist'

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

desired_cols = 6

#Empty list which will be appended later
section = []
flag_ac_presence = 0
#For loop to identify the locations of .circuit and .end
for dummy in range(len(lines)):
    if(lines[dummy] == '.circuit\n'):
        index_dotcircuit = dummy
# If .end is at end of file the array entry in lines is .end but if there is junk after .end then it will have .end\n
    if(lines[dummy] == '.end\n' or lines[dummy] == '.end'):
        index_dotend = dummy
        
    if(lines[dummy].split(" ")[0] =='.ac'):
        angfreq = float(lines[dummy].split(" ")[2])
        flag = 1

#Here we are checking if the file has .circuit and .end, and whether they appear in order
if((index_dotcircuit*index_dotend) < 0 or index_dotcircuit>=index_dotend):
    print("This is a malformed input file")
else:

    #Create an array for storing in reverse order
    rows_parse, cols_parse = (index_dotend-1-index_dotcircuit, desired_cols)
    parse = [None]*rows_parse
    for i in range(rows_parse):
        parse[i] = [None]*cols_parse
    
    for i in range(index_dotcircuit+1,index_dotend):
        section.append(lines[i])
        #First we split the string by spaces. It is possible that there may be comments but they will occur after the fourth column
        temp = lines[i].split(" ")
        #If there are no comments the fourth column will contain \n at the end. We do not want that
        temp2 = temp[3].split("\n")
        temp[3] = temp2[0]
        
        dummynew = string_split(temp[0])
        
        if(dummynew[0]=="V" or dummynew[0]=="I" or dummynew[0]=="O"):
            if(temp[3]=="dc"):
                for l in range(5):                
                    parse[i-index_dotcircuit-1][l] = temp[l]
            else:
                for l in range(6):
                    parse[i-index_dotcircuit-1][l] = temp[l]
        else:
            for l in range(4):
                parse[i-index_dotcircuit-1][l] = temp[l]


#Start of Second Assignment code
#A brief description of the algorithm:

#Create classes for components: Resistor, Capacitor, inductor, and ac and dc voltage current and voltage sources.
#Depending on what the first letter of the first element of a row of the parsed matrix is, instantiate the necessary objects.
#Create a list of objects
#Create a dictionary for node names and node indices(numbers), keeping in mind no repitition etc

#We will solve for node voltages for DC and AC sources separately(principle of superposition)
#ie first zero the AC sources and solve for DC, then zero the DC sources and solve for AC.

#One thing to keep in mind for DC is that inductor behaves as short circuit, and any amount of current can pass through it.
#This issue can be solved by treating inductor as a 0V DC battery.
#No issue in case of capacitor as it is simply open circuit(the corresponfing matrix elements are zero, indicating no connection)


#It is clear that the size of the matrix depends on whether we are solving for AC or DC.
#In case of DC, len of M is number of nodes+no of DC sources+no of inductors
#In case of AC len of M is number of nodes+number of AC sources

#Finally we print voltages of nodes for DC and AC separately, indicating Vpp and phase in case of AC 

class resistor():
    def __init__(self, entry):
        self.type = "Resistor"
        self.component_name = entry[0]
        self.value = float(entry[3])
        self.node_in = entry[1]
        self.node_out = entry[2]
    
class capacitor():
    def __init__(self, entry):
        self.type = "Capacitor"
        self.component_name = entry[0]
        self.value = float(entry[3])
        self.node_in = entry[1]
        self.node_out = entry[2]
     
class inductor():
    def __init__(self, entry):
        self.type = "Inductor"
        self.component_name = entry[0]
        self.value = float(entry[3])
        self.node_in = entry[1]
        self.node_out = entry[2]
   
    
class ac_current_source():
    def __init__(self, entry):
        self.type = "AC Current Source"
        self.component_name = entry[0]
        self.value = cmath.rect(float(entry[4]),float(entry[5])*cmath.pi/180.0)
        self.node_in = entry[1]
        self.node_out = entry[2]
        
class dc_current_source():
    def __init__(self, entry):
        self.type = "DC Current Source"
        self.component_name = entry[0]
        self.value = float(entry[4])
        self.node_in = entry[1]
        self.node_out = entry[2]
        
class ac_voltage_source():
    def __init__(self, entry):
        self.type = "AC Voltage Source"
        self.component_name = entry[0]
        self.value = cmath.rect(float(entry[4]),float(entry[5])*cmath.pi/180.0)
        self.node_in = entry[1]
        self.node_out = entry[2]
        
class dc_voltage_source():
    def __init__(self, entry):
        self.type = "DC Voltage Source"
        self.component_name = entry[0]
        self.value = float(entry[4])
        self.node_in = entry[1]
        self.node_out = entry[2]

class opamp():
    def __init__(self, entry):
        self.type = "Opamp"
        self.component_name = entry[0]
        self.node_plus = entry[1]
        self.node_minus = entry[2]
        self.node_out = entry[3]
        self.ideality = entry[4].split("\n")[0]
        
        if(self.ideality=="Nonideal"):
            self.value = float(entry[5])
        

#Create a list of objects-
list_components = [None]*rows_parse

#Number of DC and AC voltage sources is essential in determining the size of M, as discussed earlier.
#Inititalise to zero and increment dependending on the instance.

number_dc_voltagesrc = 0
number_ac_voltagesrc = 0

number_dc_currentsrc = 0
number_ac_currentsrc = 0

number_opamp = 0

#Number of inductors is also essential in determining size of M, as discussed previously
#We are treating inductors(in case of DC), as 0V DC voltage sources

number_inductor = 0
for i in range(rows_parse):
    temp3 = string_split(parse[i][0])
    
    if(temp3[0]=='R'):
        list_components[i] = resistor(parse[i])
    
    elif(temp3[0]=='C'):
        list_components[i] = capacitor(parse[i])
        
    elif(temp3[0]=='L'):
        list_components[i] = inductor(parse[i])
        number_inductor +=1
        
    elif(temp3[0]=='V' and parse[i][3]=="dc"):
        list_components[i] = dc_voltage_source(parse[i])
        number_dc_voltagesrc += 1

    elif(temp3[0]=='V' and parse[i][3]=="ac"):
        list_components[i] = ac_voltage_source(parse[i])
        number_ac_voltagesrc += 1
    
    elif(temp3[0]=='I' and parse[i][3]=="dc"):
        list_components[i] = dc_current_source(parse[i])

    elif(temp3[0]=='I' and parse[i][3]=="ac"):
        list_components[i] = ac_current_source(parse[i])

    elif(temp3[0]=='O'):
        list_components[i] = opamp(parse[i])
        number_opamp += 1


#Creating a Table of distinct nodes present in the circuit.
node_table = {"GND":0}
f = 1
for k in range(1,3):
    for i in range(rows_parse):
        if(parse[i][k]!="GND" and parse[i][k] not in node_table):
            node_table.update({parse[i][k]:f})
            f = f+1

#Solving for DC

#Create M and b of the desired size. Initialise all elements to zero
len_M_0 = len(node_table)+number_dc_voltagesrc+number_inductor+number_opamp
M_0 = np.zeros((len_M_0,len_M_0))
b_0 = np.zeros(len_M_0)

dummy1 = len(node_table)
dummy2 = len(node_table)+number_dc_voltagesrc
dummy3 = len(node_table)+number_dc_voltagesrc+number_inductor

for i in range(rows_parse):
    
    if(list_components[i].type=="DC Voltage Source"):
        M_0[dummy1][node_table[list_components[i].node_in]] = 1
        M_0[dummy1][node_table[list_components[i].node_out]] = -1
        b_0[dummy1] = list_components[i].value
        M_0[node_table[list_components[i].node_in]][dummy1] = 1
        M_0[node_table[list_components[i].node_out]][dummy1] = -1
        dummy1 = dummy1+1
    
    elif(list_components[i].type=="Resistor"):
        M_0[node_table[list_components[i].node_in]][node_table[list_components[i].node_in]]+=1/list_components[i].value
        M_0[node_table[list_components[i].node_out]][node_table[list_components[i].node_out]]+=1/list_components[i].value
        M_0[node_table[list_components[i].node_out]][node_table[list_components[i].node_in]]+=-1/list_components[i].value
        M_0[node_table[list_components[i].node_in]][node_table[list_components[i].node_out]]+=-1/list_components[i].value

    elif(list_components[i].type=="DC Current Source"):
        b_0[node_table[list_components[i].node_in]] = -list_components[i].value
        b_0[node_table[list_components[i].node_out]] = list_components[i].value

    elif(list_components[i].type=="Inductor"):
        M_0[dummy2][node_table[list_components[i].node_in]] = 1
        M_0[dummy2][node_table[list_components[i].node_out]] = -1
        b_0[dummy2] = 0
        M_0[node_table[list_components[i].node_in]][dummy2] = 1
        M_0[node_table[list_components[i].node_out]][dummy2] = -1
        dummy2 = dummy2+1

    elif(list_components[i].type=="Opamp"):
        if(list_components[i].ideality=="Ideal"):
            M_0[dummy3][node_table[list_components[i].node_plus]] = 1
            M_0[dummy3][node_table[list_components[i].node_minus]] = -1
            b_0[dummy3] = 0
            M_0[node_table[list_components[i].node_out]][dummy3] = -1
        else:
            M_0[dummy3][node_table[list_components[i].node_plus]] = 1
            M_0[dummy3][node_table[list_components[i].node_minus]] = -1
            M_0[dummy3][node_table[list_components[i].node_out]] = -list_components[i].value
            M_0[node_table[list_components[i].node_out]][dummy3] = -1
            b_0[dummy3] = 0
        dummy3+=1

M_0[0] = np.zeros(len_M_0)
M_0[0][0] = 1

if (np.linalg.det(M_0)==0):
    flag_dc_soln = 1

else:
    x_0 = np.linalg.solve(M_0,b_0)
    flag_dc_soln = 0

#Solving for AC
    
len_M_1 = len(node_table)+number_ac_voltagesrc+number_opamp
M_1 = sympy.Matrix.zeros(len_M_1)
b_1 = sympy.zeros(len_M_1,1)

dummy4 = len(node_table)
dummy5 = len(node_table)+number_ac_voltagesrc

for i in range(rows_parse):
    
    if(list_components[i].type=="AC Voltage Source"):
        M_1[dummy4,node_table[list_components[i].node_in]] = 1
        M_1[dummy4,node_table[list_components[i].node_out]] = -1
        #V = sympy.symbols(list_components[i].component_name)
        b_1[dummy4] = 1
        M_1[node_table[list_components[i].node_in],dummy4] = 1
        M_1[node_table[list_components[i].node_out],dummy4] = -1
        dummy4 = dummy4+1
    
    elif(list_components[i].type=="Resistor"):
        M_1[node_table[list_components[i].node_in],node_table[list_components[i].node_in]]+=1/list_components[i].value
        M_1[node_table[list_components[i].node_out],node_table[list_components[i].node_out]]+=1/list_components[i].value
        M_1[node_table[list_components[i].node_out],node_table[list_components[i].node_in]]+=-1/list_components[i].value
        M_1[node_table[list_components[i].node_in],node_table[list_components[i].node_out]]+=-1/list_components[i].value

    elif(list_components[i].type=="AC Current Source"):
        b_1[node_table[list_components[i].node_in]] = -sympy.symbols(list_components[i].component_name)
        b_1[node_table[list_components[i].node_out]] = sympy.symbols(list_components[i].component_name)

    elif(list_components[i].type=="Inductor"):
        M_1[node_table[list_components[i].node_in],node_table[list_components[i].node_in]]+=1/((list_components[i].value)*s)
        M_1[node_table[list_components[i].node_out],node_table[list_components[i].node_out]]+=1/((list_components[i].value)*s)
        M_1[node_table[list_components[i].node_out],node_table[list_components[i].node_in]]+=-1/((list_components[i].value)*s)
        M_1[node_table[list_components[i].node_in],node_table[list_components[i].node_out]]+=-1/((list_components[i].value)*s)

    elif(list_components[i].type=="Capacitor"):
        M_1[node_table[list_components[i].node_in],node_table[list_components[i].node_in]]+=1*((list_components[i].value)*s)
        M_1[node_table[list_components[i].node_out],node_table[list_components[i].node_out]]+=1*((list_components[i].value)*s)
        M_1[node_table[list_components[i].node_out],node_table[list_components[i].node_in]]+=-1*((list_components[i].value)*s)
        M_1[node_table[list_components[i].node_in],node_table[list_components[i].node_out]]+=-1*((list_components[i].value)*s)

    elif(list_components[i].type=="Opamp"):
        if(list_components[i].ideality=="Ideal"):
            M_1[dummy5,node_table[list_components[i].node_plus]] = 1
            M_1[dummy5,node_table[list_components[i].node_minus]] = -1
            b_1[dummy5] = 0
            M_1[node_table[list_components[i].node_out]][dummy3] = -1
        else:
            M_1[dummy5,node_table[list_components[i].node_plus]] = 1
            M_1[dummy5,node_table[list_components[i].node_minus]] = -1
            M_1[dummy5,node_table[list_components[i].node_out]] = -list_components[i].value
            M_1[node_table[list_components[i].node_out],dummy5] = -1
            b_1[dummy5] = 0
        dummy5+=1

M_1[0,:] = sympy.zeros(1,len_M_1)
M_1[0,0] = 1
        
x_1 = (M_1**-1)*b_1

print(node_table)
print("Node voltages for DC are:\n")

if(flag_dc_soln==0):
    for w in node_table:
        print(w, end=" ")
        print(x_0[node_table[w]])
else:
        print("No unique soln for DC\n")

print(M_0)
print(b_0)

print("\nNode voltages for AC are:\n")

print(M_1)
print(b_1)
print('\n')

print('\n')
print('\n')
print('\n')

H = x_1[node_table['n5']]
print(H)
#H = (1/s)*H
w = np.logspace(0,8,801)
ss = 1.j*w
hf = sympy.lambdify(s,H,'numpy')
v = hf(ss)
plt.plot(w,np.abs(v))
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.show()

n,d = sympy.fraction(H)
n_sp,d_sp=(np.array(sympy.Poly(j,s).all_coeffs(),dtype=float) for j in (n,d))
H1 = sp.lti(n_sp,d_sp)


'''
for w in node_table:
    print(w, end=" ")
    print(x_1[node_table[w]])
'''