# MSC24-WBM

Final version of the Simple Waterbalance Model (SWBM) including:
* Snow implementation
* Influence of LAI and Temperature on $\beta_0$
* Running over all grid cells at the same time using numpy arrays instead of looping to save time

  LAI data:
  * spanning from 2000-2018
  * for the last 5 years we repeat the last 5 existig years
  * missing data for some pixels (water)