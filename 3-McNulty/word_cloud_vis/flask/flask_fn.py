from one_new_comment import *


def test_new_comment(username,flair,comment):

	with open('scaler_object.pkl', 'rb') as f:
	    scaler = pickle.load(f)

	with open('text_bow_object.pkl', 'rb') as f:
	    text_bow = pickle.load(f)

	with open('link_bow_object.pkl', 'rb') as f:
	    link_bow = pickle.load(f)

	with open('random_forest_model_full.pkl', 'rb') as f:
	    rf = pickle.load(f)

	authors = pd.read_pickle('author_counts.pkl')


	test_input = {'author':username,'author_flair_text':flair,'body':comment}


	test_df = pd.DataFrame([test_input])

	test_clean = clean_df(test_df,authors)

	test_input = pd.DataFrame(bow_and_scale(test_clean,text_bow,link_bow))

	prediction = rf.predict_proba(test_input)

	if prediction[0][1] > 0.10:
		return "This post is likely to be downvoted, you may want to reconsider posting it."
	else:
		return "This post is unlikely to be downvoted, post away!"