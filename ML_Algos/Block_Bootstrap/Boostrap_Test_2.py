import numpy
import pandas

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

#create data without lags to use in variance/covariance calculation below
clean_data_with_out_lags = endogeneous.dropna(how='any')


#set the maximum number of lags that the model can consider for variable selection and estimation
#I have arbitrairly chose twice as many lags as endogeneous variables
max_lags = int(len(endogeneous_variable_names)+1)
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



#Define the "VAR IRF" algo as a function that takes in data and outputs an IRF
def Elastic_Net_Imp_QE_Resp_Unemp(clean_data):
    #Import Packages
    import numpy
    import pandas
    from sklearn.cross_validation import KFold
    from sklearn.grid_search import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import ElasticNet
    from sklearn import linear_model
    from sklearn.linear_model import ElasticNetCV

    """
    Stage 2: Compute Matricies Using LASSO
    """
    ####Calculate the Y matrix (KxT) or (4x545)
    Y = clean_data[endogeneous_variable_names].transpose()
    Y.as_matrix()

    '''Calculate B using OLS row by row
    '''
    #define the model 
    elastic = linear_model.ElasticNetCV(l1_ratio=0.5, n_alphas=100, fit_intercept=True, normalize=True, tol=0.0001, cv=10, n_jobs=-1, selection='cyclic')

    #fit a regression using all data except current value
    fed_funds_elastic = elastic.fit(clean_data.drop(endogeneous_variable_names, 1) , clean_data.fed_funds.as_matrix())
    fed_funds_elastic_coeff = fed_funds_elastic.coef_
    unemployment_elastic = elastic.fit(clean_data.drop(endogeneous_variable_names, 1) , clean_data.unemployment.as_matrix())
    unemployment_elastic_coeff = unemployment_elastic.coef_
    treasury_bill_elastic = elastic.fit(clean_data.drop(endogeneous_variable_names, 1) , clean_data.treasury_bill.as_matrix())
    treasury_bill_elastic_coeff = treasury_bill_elastic.coef_
    qe_elastic = elastic.fit(clean_data.drop(endogeneous_variable_names, 1), clean_data.qe.as_matrix())
    qe_elastic_coeff = qe_elastic.coef_

    B = treasury_bill_elastic_coeff
    #append rows of coefficients 
    B = numpy.vstack([qe_elastic_coeff,B])
    B = numpy.vstack([unemployment_elastic_coeff,B])
    B = numpy.vstack([fed_funds_elastic_coeff,B])
    B = pandas.DataFrame(B)

    #####PUT INTO SPACE STATE NOTATION#####
    #create a new coefficient matrix which is (npxnp) or (16x16)
    I_4 = numpy.identity(12)
    zero_columns = numpy.zeros((12,4))
    add_on = numpy.column_stack((I_4,zero_columns))

    #append add_on to to B to get B_Space_State
    B_Space_State = pandas.DataFrame(numpy.append(B,add_on, axis=0))

    #calculate the variance/covariance matrix using numpy
    Sigma = pandas.DataFrame(numpy.cov(clean_data_with_out_lags,rowvar=False))

    #orthogonalize Sigma by imposing off diagonals are 0
    diag_sigma = pandas.DataFrame(numpy.diag(numpy.diag(Sigma)))
    #calculate cholesky decomposistion of sigma
    chol_of_sigma = pandas.DataFrame(numpy.linalg.cholesky(diag_sigma))
    #define the inverse of the cholesky decomposistion of sigma to use below
    inverse_chol = numpy.linalg.inv(chol_of_sigma)

    ####Calculate Space-State version of the Q matrix
    zero_columns = numpy.zeros((12,4))
    zeros_16x12 = numpy.zeros((16,12))
    #append add_on to to Sigma to get Q_Space_State
    Q_Space_State = pandas.DataFrame(numpy.append(inverse_chol,zero_columns, axis=0))
    Q_Space_State = pandas.DataFrame(numpy.column_stack((Q_Space_State,zeros_16x12)))

    
    #Calculate IRs
    #create impulse column vector (TX1)
    e_feddunds = pandas.DataFrame(numpy.vstack((1,numpy.zeros(((15),1)))))
    #create impulse column vector (TX1) for "qe"
    sd_of_qe = clean_data["qe"].std()
    e_qe = pandas.DataFrame(numpy.vstack((numpy.vstack((numpy.zeros((2, 1)),sd_of_qe)),numpy.zeros((13,1)))))
    e_0 = e_qe

    #run shock e_0 through system
    Q_e = pandas.DataFrame(numpy.matmul(Q_Space_State,e_0))

    #create the point estimates (QBQE_0)
    for p in range(1,21):
        globals()['point_estimate_{}'.format(p)] = pandas.DataFrame(numpy.matmul(Q_Space_State, numpy.matmul(numpy.linalg.matrix_power(B_Space_State, p),Q_e)))

    #append point estimates into a matrix
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((Q_e,point_estimate_1)))
    
    #for p in range(2,20):
    #    impulse_qe_var = numpy.column_stack((impulse_qe_var,"point_estimate_"+str(p)))
        
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_2)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_3)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_4)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_5)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_6)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_7)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_8)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_9)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_10)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_11)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_12)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_13)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_14)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_15)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_16)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_17)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_18)))
    impulse_qe_var = pandas.DataFrame(numpy.column_stack((impulse_qe_var,point_estimate_19)))

    
    #pull out the 3rd row to get response of unemployment
    imp_qe_response_unemp = pandas.DataFrame(impulse_qe_var.loc[1:1].transpose())

    #remove the 0th impact estimate, becuase it has no variation 
    imp_qe_response_unemp = imp_qe_response_unemp.drop(imp_qe_response_unemp.index[[0]])
    
    #convert back to numpy array
    imp_qe_response_unemp = imp_qe_response_unemp.as_matrix()
    #put into Pandas Series
        #imp_qe_response_unemp = pandas.Series(imp_qe_response_unemp[:,0])    
    
    imp_qe_response_unemp = imp_qe_response_unemp[:,0]
    indexing_numbers = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 ]    
    final_output = pandas.Series(imp_qe_response_unemp, index=indexing_numbers)
    
    return final_output


#calculate point estimates using the whole data set
point_estimates = Elastic_Net_Imp_QE_Resp_Unemp(clean_data)
point_estimates = point_estimates.to_frame()
#rename the column as Point_Estimates
point_estimates = point_estimates.rename(columns = {0 : 'Point_Estimates'})
    
#Define the Stationary Bootstrap using the Cleaned Data
from arch.bootstrap import StationaryBootstrap
bs_block = StationaryBootstrap(12, clean_data)

#compute confidence intervals using bias-correction at 95%
confidence_interval_1 = bs_block.conf_int(Elastic_Net_Imp_QE_Resp_Unemp, 1000, method='bc', size=.95, tail='two', sampling='nonparametric')
confidence_interval_2 = pandas.DataFrame(confidence_interval_1, index=['Lower','Upper'])
confidence_interval_3 = confidence_interval_2.transpose()

#combine the point estimates and the confidence intervals
#REMEMBER THAT 0 PERIOD IS EXACTLY 0 WITH NO VARIANCE!! WE HAD TO REMOVE IT TO BOOTSTRAP
#ADD THE 0th ESTIMATE BACK IN 
FINAL_IR = pandas.concat([point_estimates.reset_index(drop=True), confidence_interval_3], axis=1)
print(FINAL_IR)