############################################
### README FILE 			 ###
### Supplement to Allocation of Wealth   ###
### 	Within and Across Goals, R Script###
###	Guide				 ###
############################################

R is an open-source statistical coding language, freely available
from the R project, it is downloadable here: 
	cran.r-project.org


This script requires the following libraries:
	ggplot2
	RColorBrewer
	Rsolnp

To install the libraries (this need only be done once), in the R
console, run the command: install.packages("<NAME OF PACKAGE>").

The script requires the csv files, available in this folder:
	Example Goal Details.csv
	Capital Market Expectations.csv
	Correlations - Kitchen Sink.csv

After downloading, you must input the directory of these files to
point R to their location. You can find where in the script by 
searching for the .:. symbol, which indicates the user needs to 
change the code at this location.

The outputs of the script are not immediately visible. After running,
in the R console type these variable names to return the outputs:
	optimal_weights_A	# This is the optimal investment
				# allocation for goal A.
	optimal_weights_B	# For goal B.
	optimal_weights_C	# For goal C.
	optimal_weights_D	# For goal D.
	optimal_mv_weights_A  	# This is the optimal investment
			      	# allocation for MV constrained goal A
	optimal_mv_weights_B	# For goal B.
	optimal_mv_weights_C	# For goal C.
	optimal_mv_weights_D	# For goal D.
	
	optimal_goal_weights	# The optimal across-goal allocation
	optimal_aggregate_portfolio	# The optimal agg portfolio
	optimal_goal_weights_mv	# The optimal across-goal allcoation
				# given MV constraints
