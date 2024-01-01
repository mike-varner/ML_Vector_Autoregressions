"""
Description: This script use US Macroeconomic data to calcuate a Vector Autoregression.
            This VAR is calcuated using the LASSO machine learning model. Additionally,
            impulse reseponse functions are calcualted.
Author: Michael Varner
Last Edited: October 23, 2016
"""
#Import Packages
import numpy
import pandas
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.linear_model import LassoCV

"""
Stage 1: Import and Clean Data
"""
#import data
qe_data = pandas.read_csv('/Users/Mike/Desktop/Bates Files/Senior Year/Thesis/QE_Data/qe_data.csv', low_memory=False)
#remove date variable
date = pandas.DataFrame(qe_data, columns=['date'])
x = qe_data.drop('date', 1)
#create list of exogeneous variable names 
exogeneous_var_names = ['money_supply', 'tax_rev', 'federal_debt', 'gov_exp', 'broad_index', 'financial_stress', 'industrial_production']
#remove exogeneous variables
endogeneous = x.drop(exogeneous_var_names, 1)
#create a list of the endogeneous variable names
endogeneous_variable_names = list(endogeneous.columns.values)
#create dataframe of exogeneous variables
exogeneous = x.drop(endogeneous_variable_names, 1)

#set the maximum number of lags that the model can consider for variable selection and estimation
#I have arbitrairly chose twice as many lags as endogeneous variables
max_lags = int(len(endogeneous_variable_names)*2+1)
lags = (range(max_lags))[1:]

#create lagged values of the endogeneous variables
for p in lags:
    for var in endogeneous_variable_names:
        endogeneous["L{0}_{1}".format(p , var)] = endogeneous[var].shift(p)

#add back in the exogeneous variables
#prepared_data = endogeneous.add(exogeneous)

#remove missing values created by creating lags
#this might not be what I want, but come back to it later
clean_data = endogeneous.dropna(how='any') 

"""
Stage 2: Compute Matricies Using LASSO
"""
####Calculate the Y matrix (KxT)
Y = clean_data[endogeneous_variable_names].transpose()
Y.as_matrix()

###Calculate B Matrix (k*(kp+1))####
#define the LASSO with 10 KFold CV, to calculate coefficients and intercepts
lasso = linear_model.LassoCV(cv=10, n_jobs=-1 , normalize=True)
'''
#fit the LASSO only using lagged values of the endogeneous variables
qe_lasso = lasso.fit(clean_data.drop(endogeneous_variable_names, 1).as_matrix() , clean_data.qe.as_matrix())
#strip the coefficients from the LASSO
#add the intercept to the beginning of the array
qe_lasso_coeff = numpy.hstack((qe_lasso.intercept_, qe_lasso.coef_))
'''

fed_funds_lasso = lasso.fit(clean_data.drop(endogeneous_variable_names, 1).as_matrix() , clean_data.fed_funds.as_matrix())
fed_funds_lasso_coeff = numpy.hstack((fed_funds_lasso.intercept_, fed_funds_lasso.coef_))
unemployment_lasso = lasso.fit(clean_data.drop(endogeneous_variable_names, 1).as_matrix() , clean_data.unemployment.as_matrix())
unemployment_lasso_coeff = numpy.hstack((unemployment_lasso.intercept_, unemployment_lasso.coef_))
treasury_bill_lasso = lasso.fit(clean_data.drop(endogeneous_variable_names, 1).as_matrix() , clean_data.treasury_bill.as_matrix())
treasury_bill_lasso_coeff = numpy.hstack((treasury_bill_lasso.intercept_, treasury_bill_lasso.coef_))
qe_lasso = lasso.fit(clean_data.drop(endogeneous_variable_names, 1).as_matrix() , clean_data.qe.as_matrix())
qe_lasso_coeff = numpy.hstack((qe_lasso.intercept_, qe_lasso.coef_))

#initialize the first row of the B matrix as fed_funds coefficients
B = fed_funds_lasso_coeff
#append rows of coefficients 
B = numpy.vstack([unemployment_lasso_coeff,B])
B = numpy.vstack([qe_lasso_coeff,B])
B = numpy.vstack([treasury_bill_lasso_coeff,B])
B = pandas.DataFrame(B)

'''
B = fed_funds_lasso_coeff
for var in endogeneous_variable_names:
    globals()['lasso_'+str(var)] = lasso.fit(clean_data.drop(endogeneous_variable_names, 1).as_matrix(), clean_data[var].as_matrix())
    globals()['coeff_lasso_'+str(var)] = numpy.hstack((globals()['lasso_'+str(var)].intercept_, 'lasso_'+str(var)].coef_))
    B_loop = numpy.vstack(globals()['coeff_lasso_'+str(var)],B])
'''

####Calculate the Z matrix ((Kp+1)xT)####
#swap the rows and colummns of the clean_data matrix 
transpose_of_clean = clean_data.drop(endogeneous_variable_names, 1).transpose()
#number of observatiosn, or columns
T = float(len(transpose_of_clean.columns))
#create a (1xT) row of 1s to add into the Z matrix
row_of_ones = numpy.ones((T,), dtype=np.int)
#add row of ones as the first column in transpose_of_clean
Z = numpy.vstack((row_of_ones,transpose_of_clean.as_matrix()))
Z = pandas.DataFrame(Z)

####Calculate the Gamma matrix ((Kp+1)x(Kp+1))
Gamma = numpy.matmul(Z,Z.transpose())/T
Gamma = pandas.DataFrame(Gamma)

####Calculate the residual matrix U (KxT)
Y_hat = numpy.matmul(B,Z) #should be (KxT)
U = pandas.DataFrame(Y-Y_hat)

####Calculate Sigma matrix (TxT)
one_over_T = float(1/T)
Sigma = pandas.DataFrame(numpy.matmul(U,U.transpose())*one_over_T)



######################Compute IRs######################################

#orthogonalize Sigma by imposing off diagonals are 0
diag_sigma = pandas.DataFrame(numpy.diag(np.diag(Sigma)))
#calculate cholesky decomposistion of sigma
chol_of_sigma = pandas.DataFrame(numpy.linalg.cholesky(diag_sigma))
#define the inverse of the cholesky decomposistion of sigma to use below
inverse_chol = numpy.linalg.inv(chol_of_sigma)

#define Q matrix as the inverse of the cholesky decomposistion of Sigma
Q = pandas.DataFrame(inverse_chol)
'''Check Matricies
#verify that this works correctly by checking (diag_sigma=chol_of_sigma' * chol_of_sigma)
    check_chol = pandas.DataFrame(numpy.matmul(chol_of_sigma.transpose(),chol_of_sigma))

#need the Q matrix from cocharane book, should equal the identity matrix
#Q*Sigma*Q' = I
identity_matrix = pandas.DataFrame(numpy.matmul(numpy.matmul(Q,diag_sigma),Q.transpose()))
'''#pg.362 of Lutekephol might help



######Create the C(L) matrix and the n_t error matrix from Cochrane#####
#change B matrix because Cochrane uses different notation B_L (33x4)
B_L = B.transpose()

#C_L = BxQ^(-1) 
C_L = pandas.DataFrame(numpy.matmul(B_L,numpy.linalg.inv(Q)))

#check to see if E(n_t n_t') = I ; where n_t = Q x e_t
#n_t = pandas.DataFrame(numpy.matmul(Q,U))
#check_transformed_errors = pandas.DataFrame(numpy.matmul(n_t,n_t.transpose()))

#create impulse column vector (TX1) or 36x1
impulse = pandas.DataFrame(numpy.vstack((1,numpy.zeros(((3),1)))))

'''
#create the point estimates for the first 5 periods
point_estiamte_1 = numpy.matmul(numpy.linalg.matrix_power(C_L, 1),impulse)
point_estiamte_2 = numpy.matmul(numpy.linalg.matrix_power(C_L, 2),impulse)
point_estiamte_3 = numpy.matmul(numpy.linalg.matrix_power(C_L, 3),impulse)
point_estiamte_4 = numpy.matmul(numpy.linalg.matrix_power(C_L, 4),impulse)
point_estiamte_5 = numpy.matmul(numpy.linalg.matrix_power(C_L, 5),impulse)
'''













