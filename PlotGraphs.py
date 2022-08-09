from matplotlib import pyplot as plt
import io
import re
import numpy as np

class PlotGraph:

	def read_output_data_and_prepare_plot(self):
		count = 0
		iterations =[]
		losses = []		
		with open('TrainLog.txt') as f:
			next(f)
			for line in f:
				#if count >=4:
					#break
				result = re.search('Iteration:(.*) batch loss :=(.*)', line)
				iterations.append(int(result.group(1)))	
				losses.append(float(result.group(2)))
				count+=1
		#print(iterations)
		#print(losses)
		plt.plot(iterations,losses)
		#plt.xticks(np.arange(iterations[0],iterations[-1],1))
		plt.xlabel("Iterations")
		plt.ylabel("Batch Losses")
		plt.show()


g = PlotGraph()
g.read_output_data_and_prepare_plot()
