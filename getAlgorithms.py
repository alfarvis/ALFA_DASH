class getAlgorithms:
	def __init__(self):
		algorithms = [
		{'label': 'Visualization: Bar Plot','value':'barPlot'},
		{'label': 'Visualization: Box Plot','value':'boxPlot'},
		{'label': 'Visualization: Histogram','value':'histogram'},	
		{'label': 'Visualization: Pie Chart','value':'pieChart'},				
		{'label': 'Visualization: Violin Plot','value':'violinPlot'},
		{'label': 'Visualization: 2D Scatter Plot','value':'scatter2D'},
		{'label': 'Visualization: 3D Scatter Plot','value':'scatter3D'},
		{'label': 'Visualization: Bland Altman','value':'blandAltman'},
		
		{'label': 'Visualization ML: tSNE','value':'tsne_VP'},
		{'label': 'Visualization ML: PCA','value':'pca_VP'},
		{'label': 'Visualization ML: Isomap','value':'isomap_VP'},
		{'label': 'Visualization ML: Diffusion Map','value':'dfm_VP'},
		{'label': 'Visualization ML: LLE','value':'lle_VP'},
		
		{'label': 'Machine Learning: SVM','value':'svm_VP'},
		{'label': 'Machine Learning: Logistic Regression','value':'logreg_VP'},
		{'label': 'Machine Learning: Decision Trees','value':'dt_VP'},
		{'label': 'Machine Learning: Random Forest','value':'rf_VP'},
		{'label': 'Machine Learning: AutoML','value':'automl_VP'},
		{'label': 'Machine Learning: Artificial Neural Networks','value':'ann_VP'},

		{'label': 'Clustering: kMeans','value':'kMeans_VP'},
		{'label': 'Clustering: Hierarchical','value':'hierarchical_VP'},
		{'label': 'Clustering: Spectral','value':'spectral_VP'},
		
		{'label': 'Statistics: ttest','value':'ttest_VP'},
		{'label': 'Statistics: anova','value':'anova_VP'},
		{'label': 'Statistics: Wilkoxon rank-sum','value':'wilkoxon_VP'},		
		{'label': 'Statistics: correlation','value':'ttest_VP'},
		{'label': 'Statistics: Matthews correlation coefficient','value':'mcc_VP'},
		{'label': 'Statistics: Univariate ROC curve','value':'roc_curve_VP'}, #done
		{'label': 'Statistics: Univariate PR curve','value':'pr_curve_VP'}, #done

		

		]
		self.algo_list = algorithms

	def getAlgos(self):
		return self.algo_list