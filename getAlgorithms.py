class getAlgorithms:
	def __init__(self):
		algorithms = [
		{'label': 'Visualization: Box Plot','value':'boxPlot'},
		{'label': 'Visualization: 2D Scatter Plot','value':'scatter2D'},
		{'label': 'Visualization: 3D Scatter Plot','value':'scatter3D'},
		{'label': 'Visualization: Histogram','value':'histogram'},				

		]
		self.algo_list = algorithms

	def getAlgos(self):
		return self.algo_list