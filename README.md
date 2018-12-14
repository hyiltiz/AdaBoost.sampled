# AdaBoost.sampled

# Data sets
1. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#breast-cancer
2. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#mushrooms
3. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#diabetes

# Possible replacement for 'mushrooms'

4. https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna

# Literature Review
1. Weak Learner to subsamples: https://www.sciencedirect.com/science/article/pii/S0377042707001343
2. Analysis of Adaboost Variants: https://www.hindawi.com/journals/jece/2015/835357/
3. Seminal paper on Adaboost by Freund and Schapire: https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf
4. Parametrized Adaboost, penalizes misclassification of already correctly classified samples: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6778011
5. Real Adaboost : https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1999-ML-Improved%20boosting%20algorithms%20using%20confidence-rated%20predictions%20(Schapire%20y%20Singer).pdf
6. Gentle Adaboost, Newton stepping at each step instead of exact optimization: https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1999-ML-Improved%20boosting%20algorithms%20using%20confidence-rated%20predictions%20(Schapire%20y%20Singer).pdf
7. Modest Adaboost, better generalization error than Gentle Adaboost: http://graphicon.ru/html/2005/proceedings/papers/vezhnevetz_vezhnevetz.pdf
8. Margin pruning boost, reduces overfitting of Gentle Adaboost: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9069/90691P/A-new-method-for-solving-overfitting-problem-of-gentle-AdaBoost/10.1117/12.2050093.full?SSO=1
9. Penalized Adaboost, improves generalization error of Gentle Adaboost :https://www.jstage.jst.go.jp/article/transinf/E98.D/11/E98.D_2015EDP7069/_article 
