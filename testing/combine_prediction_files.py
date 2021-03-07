import os
import glob
import pandas as pd
# this file is to combine the separate predicted .csv files
#set working directory
os.chdir("./predicted/")

# find all csv files in working directory
all_predicted_files = [i for i in glob.glob('*.{}'.format('csv'))]

#combine all files in the list
combined_predictions = pd.concat([pd.read_csv(f) for f in all_predicted_files])
#export to csv
combined_predictions.to_csv("../combined_predicted_election_tweets.csv", index=False, encoding='utf-8-sig')