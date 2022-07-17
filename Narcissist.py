from re import T
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from pandas.core.indexes.base import Index
from sklearn.model_selection import train_test_split

import streamlit as st
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from streamlit.state.session_state import SessionState
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import sys
import PIL



scaler = StandardScaler()
st.image("./narcissistic-personality-disorder-therapists.jpg")
st.title('The Narcissistic Personality Inventory')
st.markdown("The Narcissistic Personality Inventory (NPI) has grown to be one of the most extensively used personality tests for non-clinical levels of trait narcissism.")
st.markdown('Please beware that this is not a diagnosis, if you get a high score that doesn’t mean that you have clinical narcissism.')
st.markdown("When you’re ready please click (Start).")
st.markdown("Note: The test is timed, as soon as you finish please click (Submit).")
if 'start_time' not in st.session_state:
    st.session_state.start_time = 0
if 'end_time' not in st.session_state:
    st.session_state.end_time = 0
if 'age' not in st.session_state:
    st.session_state.age = 0
if 'gender' not in st.session_state:
    st.session_state.gender = 0



data = pd.read_csv('./data.csv')


data = data[data['age']<100]
X = data.drop(columns="score")
y = data["score"]
# X = scaler.fit_transform(X)
Train_x, test_x, Train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=101)
lr = LinearRegression()
lr.fit(Train_x,Train_y)
y_pred = lr.predict(test_x)
st.caption("Below are the accuracy results of the model: ")
st.write("The MAE of the model is: ", mean_absolute_error(y_pred,test_y))
st.write("The RMSE of the model is: ", mean_squared_error(test_y, y_pred, squared=False))
st.write("The R2 of the model is: ",r2_score(test_y,y_pred))


df = data[1:]
npi_score = data.iloc[:,0]
avg_score = (npi_score).mean()

male_score = (npi_score[data.iloc[:,42]==1].mean())
female_score = (npi_score[data.iloc[:,42]==2].mean())

d = pd.read_csv('/Volumes/XTRM-Q/Narcissist/data.csv')
col_names = ['score', 'influencing', 'modesty', 'easy_to_bait', 
'compliments', 'world_ruler', 'avoid_responsibility',
 'self_attention', 'success', 'special', 'leader',
'assertive', 'authoritarian', 'manipulator',
'insisting_respect', 'likes_self_body', 'understands_people',
'takes_responsibility', 'wants_to_achieve', 'likes_to_look_at_self_body',
'shows_off', 'confident_in_actions', 'independent', 'good_storyteller',
'recieve_stuff_from_ppl', 'gets_whats_deserves', 'recieve_compliment',
'will_to_power', 'fashionista', 'likes_mirrors', 'center_of_attention',
'lives_their_ways', 'being_authority', 'being_leader', 'goes_to_be_great',
'missionary', 'born_leader', 'biography', 'ppl_notice_look', 'more_capable',
 'extraordinary', 'elapse', 'gender', 'age']
ind = [1,1,1,2,2,1,2,1,2,2,1,1,1,1,2,1,2,2,2,2,1,2,2,1,1,2,1,2,1,1,1,2,1,1,2,1,1,1,1,2]
d = d.set_axis(col_names, axis=1, inplace=False)
for i, column in enumerate(d.iloc[:,1:41]):
    if i==42:
        break
    c = d[column]
    for num_i in range(len(c)):
        if c[num_i] == ind[i]:
            c[num_i] = 1
        else:
            c[num_i] = 0
    d[column] = c



import warnings
warnings.filterwarnings('ignore')
mean = d.groupby(by=["age", "gender"],as_index=False).mean()

df = d.iloc[:,1:41]
age = d['age']
gender = d['gender']
df['age'] = age
df['gender'] = gender
df = df.groupby(by=['age', 'gender']).sum()




def dis():
    sums_by_age = pd.DataFrame(df.sum())
    sns.set_context('paper')
    f, ax = plt.subplots(figsize = (6,15))
    sns.set_color_codes('pastel')
    clrs = ['grey' if (x < 5000) else 'red' for x in sums_by_age[0] ]
    sns.barplot(x = 0, y =list(sums_by_age.index),data = sums_by_age,
                edgecolor = 'black',  palette=clrs)

    ax.legend(ncol = 2, loc = 'lower right')
    sns.despine(left = True, bottom = True)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


def display_answer(ans):
    if ans > avg_score:
        st.title(f'You scored : {ans} which is higher than the average ')
    else:
             st.title(f'You scored : {ans} which is lower than the average ')
    
    st.write("Scores of people your gender and age group: ")
    f = sns.displot(data["score"],kde=True)
    
    st.pyplot(f)
    print("Score Mean: ",data["score"].mean())
    values = (male_score,female_score)
    labels = ['Male','Female']
    st.write("Males had higher scores on average")
    fig = px.bar(x = labels, y = values)
    st.plotly_chart(fig)
    
    fig2 = px.scatter(data , x = data['age'],y=data['score'],color = "gender")
    st.plotly_chart(fig2)
    age = st.session_state.age
    ages = []
    for i in range(age-3,age+4):
        ages.append(i)
    
    d = data[(data["gender"] == np.int64(st.session_state.gender)) & (data['age'].isin(ages))]
    figg = px.scatter(d,y='score',color='age')
    figg
     
    st.subheader(f'The average score of people your gender and age group is: {d["score"].mean()}')
    
    
    
    





def get_answers():
    
    lines = []
    file = open("/Volumes/XTRM-Q/Narcissist/quest.txt","r")
    lines = file.readlines()
    answers = []
    st.session_state.age = st.slider("Please select your age:",14,100)
    st.session_state.gender = st.radio("Please select your gender, (1-Male) (2-Female)",('1','2'))
    placeholder = st.empty()
    isClicked = placeholder.button('start')
    if isClicked:
        placeholder.empty()
        st.session_state.start_time = time.time()

    with st.form(key='my-form'):
        st.header("Now, Start answering the following questions: ")
        for line in lines :
            answers.append(st.radio(line,('1','2')))
        
        if st.form_submit_button("Submit"):
            st.session_state.end_time = time.time()
    answers.append(st.session_state.end_time - st.session_state.start_time)
    answers.append(st.session_state.gender)
    answers.append(st.session_state.age)     

    
    
    aa = lr.predict([answers])
    
    
    display_answer(aa)
    answers.append(aa)
    return answers
        





col_names2 = ['age','gender','score', 'influencing', 'modesty', 'easy_to_bait', 
'compliments', 'world_ruler', 'avoid_responsibility',
 'self_attention', 'success', 'special', 'leader',
'assertive', 'authoritarian', 'manipulator',
'insisting_respect', 'likes_self_body', 'understands_people',
'takes_responsibility', 'wants_to_achieve', 'likes_to_look_at_self_body',
'shows_off', 'confident_in_actions', 'independent', 'good_storyteller',
'recieve_stuff_from_ppl', 'gets_whats_deserves', 'recieve_compliment',
'will_to_power', 'fashionista', 'likes_mirrors', 'center_of_attention',
'lives_their_ways', 'being_authority', 'being_leader', 'goes_to_be_great',
'missionary', 'born_leader', 'biography', 'ppl_notice_look', 'more_capable',
 'extraordinary', 'elapse']

comp = ['influencing', 'modesty', 'easy_to_bait', 'compliments', 'world_ruler','avoid_responsibility',
 'self_attention', 'success', 'special', 'leader','assertive', 'authoritarian', 'manipulator',
'insisting_respect', 'likes_self_body', 'understands_people',
'takes_responsibility', 'wants_to_achieve', 'likes_to_look_at_self_body',
'shows_off', 'confident_in_actions', 'independent', 'good_storyteller',
'recieve_stuff_from_ppl', 'gets_whats_deserves', 'recieve_compliment',
'will_to_power', 'fashionista', 'likes_mirrors', 'center_of_attention',
'lives_their_ways', 'being_authority', 'being_leader', 'goes_to_be_great',
'missionary', 'born_leader', 'biography', 'ppl_notice_look', 'more_capable',
 'extraordinary']
def show():
    l = get_answers()
    
    age = l[42]
    gender = int(l[41])
    a = mean[(mean['age'] == age)] 
    b = a[a['gender']==gender]
    
    h = {}
    length = len(col_names2)

    traits = []
    for i in range(0,40):
        if int(l[i]) == ind[i]:
            traits.append(1)
        else:
            traits.append(0)
    
    for i in range(0,1):
        h[i] = {}
        for j in range(0,length):
            h[i][j] = col_names2[j]
    
    arr = []
    arr.append(age)
    arr.append(gender)
    arr.append(l[43][0])
    for i in range(0,40):
        arr.append(traits[i])
    arr.append(l[40])
    b.loc[1] = arr
    source = ["Data", "User"]
    b['source'] = source
    
    
    temp = pd.DataFrame(b)
    st.subheader("This is how each of your narcissitic traits compare to people your exact age and gender: ")
    plt.figure(figsize= (20,40))
    for i in enumerate(comp):
        plt.subplot(10,4,i[0]+1)
        sns.barplot(x=i[1],y='source',data=temp)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    
    

show()








