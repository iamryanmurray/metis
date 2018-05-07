import pandas as pd
import glob


def import_all():
	path = './data/'
	allFiles = glob.glob(path +'/*.csv')

	frame = pd.DataFrame()
	list_ = []

	for file_ in allFiles:
	    df = pd.read_csv(file_,index_col=None,header=0,low_memory=False)
	    list_.append(df)
	    
	frame = pd.concat(list_)

	frame.to_pickle('./data/all_data.pkl')