from pymongo import MongoClient
import pandas as pd


def import_from_mongo():
	#loads data from mongo into dataframe and returns a list of space 
	#separated titles and a list of full transcripts
	client = MongoClient()
	ted = client.ted_database
	df = pd.DataFrame(list(ted.ted_transcripts.find()))
	heads = list(df.title)
	heads = [h.split('_') for h in heads]
	heads = [' '.join(h) for h in heads]

	desc = list(df.transcript)

	return heads,desc
