import trainning_script as ts
import preprocessing_script as ps
from data_processing import data_pre_processing_script
from data_processing import data_prep as dp

dp.windows = True
dp.cuda = True

#to prepare the data
data_pre_processing_script.main()