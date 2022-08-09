import random
import csv

class CreateBinaryStringDataset:
	def CreateBinaryStringsAndWriteToFile(self,numberofStrings,filename,isFixedLen = True): 
		random.seed(1)
		strLen = 50  
		rows = []
		for i in range(0,numberofStrings):
			if(not isFixedLen):
				strLen = random.randint(1,50)  
			outputStr = ""
			inputStr = ""
			for j in range(0,strLen):
				val = random.randint(0,1)
				inputStr += str(val)
				xorOutputStr = "" 
				xorVal = 0
			for k in range(0,strLen):
				xorVal = xorVal^(int(inputStr[k]))
				xorOutputStr += str(xorVal) 
			#print("input=%s ,output =%s" %(inputStr,xorOutputStr))
			row = [inputStr,xorOutputStr]
			rows.append(row)  
		csvHandle = open(filename,'w+')
		csvWriter = csv.writer(csvHandle)
		csvWriter.writerows(rows)
		csvHandle.close()


#print("Fixed Len Data Set=")
createBinaryStringDatasetObj = CreateBinaryStringDataset()
createBinaryStringDatasetObj.CreateBinaryStringsAndWriteToFile(100000,"FixedLengthDataSet.csv",True)
#print("Variable Len Data Set=")
createBinaryStringDatasetObj.CreateBinaryStringsAndWriteToFile(100000,"VariableLengthDataset.csv",False)



