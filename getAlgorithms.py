class getAlgorithms:
	def __init__(self):
		algorithms = ['bar plot',
		'histogram',
		'scatter plot',
		'violin plot',		
		]
		self.algo_list = [{'label':i,'value':i} for i in algorithms]

	def getAlgos(self):
		return self.algo_list