# Reddit Comment Predictive Analysis - Upvoted or Downvoted

In this project, I analyzed comments from [Reddit's Seattle subreddit](https://reddit.com/r/Seattle) made between 2014 and 2017 to develop a model which can predict if a new comment will be upvoted or downvoted.

I used Google's [BigQuery](https://cloud.google.com/bigquery/) API to scrape the relevant comments, only focusing on comments with a score above +10 or below -10.  This gave me a dataset of approximately 140,000 comments, with about a 6:1 ratio of Upvoted:Downvoted comments.

I extracted meta-features like average word length and sentiment analysis of the comments and cleaned the text by removing punctuation and stopwords before converting the text to a "Bag of Words" model with n-grams from 1-3 words long.  This 