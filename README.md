# EntityMatching
1. Code - The code for my solution is implemented at :
2. Solution outline - My solution has five steps: (1) Read data, (2) perform blocking, (3) feature engineering, (4) training model on train.csv, and (5) push output to output.csv file.
3. Read data - As the sample solution does, I am reading through 2554 lines in the left table times 22074 lines in the right table through a total of 56376996 pairs.
4. Perform blocking - I modified my entity matching algorithm to perform blocking by category -- I found that while there are actual matches filtered out, a lot of insane matches that occur based on products from different categories are reduced. Also, since brands can make many products, it seems better to get a false match of a product that is at least in the same industry. 
5. Feature engineering - Modifying the feature engineering from Jaccard similarity to cosine similarity seemed to make a massive difference in reducing false matches. Levenshtein distance does seem to perform the best for comparing the strings in the rows. 
6. Model Training - For this part, I tried applying a RandomizedSearchCV while trainin & optimizing hyperparameters, yet this made output.csv return 6 times as many values which I knew to be false matches. GridSearch did not work any better with the randomforest classifier, so I kept it as is. 
7. Push Output - Here, any predicted value of 1, meaning a predicted match, is pushed to the output.csv table after finishing through the algorithm.
