import pandas as pd
import time

df = pd.read_json('../data/webmasters_questions.json', orient='index')

df['QuestionID'] = df.index

mydf = df[['QuestionID', 'title', 'dups']]

#print(mydf)

newdf = pd.DataFrame(columns=['Q1', 'Q2', 'Dup'])
print(len(mydf.index))

count = 0

for i in range(0, mydf['QuestionID'].size):
    tic = time.process_time()
    for j in range(i + 1, mydf['QuestionID'].size):
        #print(mydf.iloc[i, 0] , mydf.iloc[j, 2])
        newdf.at[count,'Q1'] = mydf.iloc[i, 1]
        newdf.at[count,'Q2'] = mydf.iloc[j, 1]
        if mydf.iloc[i, 0] in mydf.iloc[j, 2]:
            newdf.at[count, 'Dup']='Y'
        else:
            newdf.at[count, 'Dup']='N'
        count=count+1
        #print(i,j)
        #print(count)
        #print(newdf)
        #print(newdf.iloc[[count]])
    toc = time.process_time()
    print(toc-tic)

print(newdf.head())

        #print(mydf.loc[j,'dups'])


    # for j in range(i+1, mydf['QuestionID'].size):
    #     if mydf.loc[i,'QuestionID'] in mydf.loc[j,'dups'] :
    #         print('FOUND !')
    #     else :
    #         pass

# print(df.iloc[0])

#mydf = df[['title', 'dups']]

#mydf.rename(columns={list(mydf)[0]:'col1_new_name'}, inplace=True)

#print(list(mydf))

#with open('../data/test_data.json', 'r') as f:
    #json_text = f.read()

# Decode the JSON string into a Python dictionary.
#english_questions = json.loads(json_text)

#print(english_questions)

# englishDf = pd.DataFrame.from_dict(json_normalize(english_questions), orient='columns')

# print(englishDf.iloc[0])

#df = pd.DataFrame.from_dict(english_questions,orient='index')