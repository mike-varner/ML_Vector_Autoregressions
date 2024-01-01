set more off
clear
use "/Users/Mike/Desktop/Work Folder/Research With Shea Winter 2016/Updated US.dta"
log using "/Users/Mike/Desktop/Bates Files/Senior Year/Thesis/Algo_Check/Algo_Check" , replace
tsset date
	
	/*Infaltion Rate*/
	dfgls cpi
	dfuller cpi, lags(13) regress trend
	gen cpi2 = cpi - date* -.000111 
	dfuller D.cpi2, lags(13) regress trend	
	/*reject a 2nd unit root*/
	
	/*U3 Unemployment Rate*/
	dfgls unemp
	dfuller unemp, lags(4) regress trend
	dfuller D.unemp, lags(4) regress trend
	/*reject a 2nd unit root*/
	

var D.unemp D.cpi2, lags(1)
matlist e(Sigma)
summ D.unemp D.cpi2
drop if date==1
corr D.unemp D.cpi2, cov
summ D.unemp D.cpi2 


irf create robustcheck1, step(20) set (/Users/Mike/Desktop/Bates Files/Senior Year/Thesis/Algo_Check/Algo_Check ,replace) 
irf table oirf, impulse(D.unemp) response(D.cpi2)
irf table oirf, impulse(D.unemp) response(D.unemp)
irf table oirf, impulse(D.cpi2) response(D.cpi2)
irf table oirf, impulse(D.cpi2) response(D.unemp)

gen inflation = D.cpi2
gen unemployment = D.unemp

keep unemployment inflation date
	 
cd "/Users/Mike/Desktop/Bates Files/Senior Year/Thesis/Algo_Check"
/*save as a .csv for use in python*/
export delimited toy_data, replace

log close
