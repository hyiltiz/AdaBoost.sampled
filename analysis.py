import numpy as np 
import adaboost as ab 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
import sys
import matplotlib
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def error_vs_classifier_ratio(data_npy = 'breast-cancer', n_ratios = 20, T = int(1e2), seed = 0, loglevel = 0):
	bc_ratios, errors = np.linspace(0.05, 1, n_ratios, endpoint = True), []
	for i in range(n_ratios):
		print 'working on iteration number ' + str(i+1) + ' out of ' + str(n_ratios)
		g, test_error = ab.adaBoost(data_npy,(0,bc_ratios[i]),T, seed, loglevel)
		errors.append(test_error)
	pp = PdfPages(data_npy + '_err_vs_ratio_plots.pdf')
	error_vs_ratio_fig = plt.figure()
	plt.plot(bc_ratios, errors)
	plt.title('Final Ensemble Error as a Function of Percentage of Stumps Used')
	plt.xlabel('Percent Classifiers Used')
	plt.ylabel('Error')
	plt.show()
	pp.savefig(error_vs_ratio_fig)
	pp.close()

if __name__ == '__main__':
	error_vs_classifier_ratio(sys.argv[1])
