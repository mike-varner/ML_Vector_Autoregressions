# ML_Vector_Autoregressions
Linear machine learning methods (LASSO, Ridge, and Elastic-Net) to estimate Vector Autoregresssions (VARs) and compute Impulse Response Functions (IRFs). VARs have been particularily popular in empirical dynamic macroeconomics as they are easy to estimate and interpret. These types of models are used both for structural inference as well as forcasting. While these models are relatively easy to program (primarily Stata and Matlab) and decipher, they come with the high cost of requiring many identifying assumptions. Additionally, these models almost exclusively use Ordinary Least Squates for parameter estimation. This project proposes three new flavors of VARs in an attempt to resolve some of these known issues and potentially provide more accurate forcasts.

## Motivation/Background
This project was initially developed as part of [Vector Autoregressions with Machine Learning](https://scarab.bates.edu/honorstheses/216/) (an Honors Thesis in Economics by Mike Varner Bates College class of 2017). Variable definitions, parameter selection methods, and notation are defined therein. This project generally follows [Lutkepohl (2007)](https://www.amazon.com/New-Introduction-Multiple-Time-Analysis/dp/3540401725/ref=asc_df_3540401725/?tag=hyprod-20&linkCode=df0&hvadid=312006100296&hvpos=1o1&hvnetw=g&hvrand=9933111550468506554&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9031948&hvtargid=pla-525627263170&psc=1).

## Similar Projects
Will Nicholson has created a similar VAR estimation scheme here: [Big VAR](http://www.wbnicholson.com/BigVAR.html). They leverage the LASSO's variable selection feature in creating their models. Their method is programmed in R whereas this project solely relies on Python. 

## Built With

* [Pandas](https://pandas.pydata.org/) - Data manipulation and Linear Algebra
* [Numpy](https://www.numpy.org/) - Data manipulation and Linear Algebra
* [Scikit-Learn](https://scikit-learn.org/stable/) - Machine Learning parameter estimation 
* [Arch](https://github.com/bashtage/arch) - Bootstrapping methods

## Authors

* **Mike Varner** - *Initial Development* - [link](https://github.com/regmonkeyols/)

## Acknowledgments

* This project would not have been possible without the support of Nathan Tefft

#End of File
