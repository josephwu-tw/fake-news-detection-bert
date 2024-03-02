# setup

import __path__ as path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
import string


# functions

def date_col_cleaner(column):
    try:
        # Convert value to datetime object
        return pd.to_datetime(column) 
    except:
        # If incompatible with datetime, conver to NaN
        return np.NaN

def text_cleaner(text):
    url = re.compile(r'https?://\S+')
    html = re.compile(r'<.*?>')
    name = re.compile(r'@\S+')
    hashtag = re.compile(r'[#]')
    exclamation = re.compile(r'[!]')
    question = re.compile(r'[?]')
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    text = url.sub(r' url ', text)
    text = html.sub(r' html ', text)
    text = name.sub(r' atsymbol ', text)
    text = hashtag.sub(r' hashtag ', text)
    text = exclamation.sub(r' exclamation ', text)
    text = question.sub(r' question ', text)
    text = text.translate(str.maketrans('','',string.punctuation))
    text = emoji_pattern.sub(r' emoji ', text)
    
    return text

# data import

fake = pd.read_csv(path.data_dir + 'Fake.csv').dropna()
true = pd.read_csv(path.data_dir + 'True.csv').dropna()

# adding label

fake['label'] = 1
true['label'] = 0


# merge data and remove duplicates

df = pd.concat([fake, true], axis = 0, ignore_index = True)
df.drop_duplicates(inplace = True)

# date column change type

df['date'] = df['date'].apply(date_col_cleaner)
df = df.dropna()

# # show fake and real pie chart

# fig = go.Figure(data = [go.Pie(labels =['Fake', 'True'],
#                                values = list(df['label'].value_counts()),
#                                textinfo = 'label+percent',)])

# fig.show()

# show subject count bar chart

df = pd.get_dummies(df, columns= ['subject'], prefix='', prefix_sep='', dtype = int)

# subject = df.subject.value_counts().sort_values(ascending = True)
# subject_ax = subject.plot(kind = 'barh', title = 'Counts by Subject')

# plt.show()

# # show date count bar chart

# date = pd.DataFrame(df.date.dt.to_period('M').value_counts(sort = False))
# date_ax = date.plot.bar(title = 'Counts by Date',
#                         legend = False,
#                         rot = 10)

# for i, t in enumerate(date_ax.get_xticklabels()):
#     if (i % 6) != 0:
#         t.set_visible(False)

# plt.savefig(path.plt_dir + 'date count.jpg')

# merge title and text

df['text'] = 'titlestart ' + df['title'] + ' titleend ' + df['text']
del df['title']

# reformat the text

df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda text: text_cleaner(text))

df['text'] = df['text'].apply(lambda text: ' '.join(text.strip().split()))

# adding text feature columns

df['url'] = df['text'].apply(lambda text: text.count(' url '))
df['html'] = df['text'].apply(lambda text: text.count(' html '))
df['atsymbol'] = df['text'].apply(lambda text: text.count(' atsymbol '))
df['hashtag'] = df['text'].apply(lambda text: text.count(' hashtag '))
df['exclamation'] = df['text'].apply(lambda text: text.count(' exclamation '))
df['question'] = df['text'].apply(lambda text: text.count(' question '))
df['emoji'] = df['text'].apply(lambda text: text.count(' emoji '))

number = re.compile(r'\d+')
df['number'] = df['text'].apply(lambda text: number.sub(r' number ', text).count(' number '))

# show text features bar chart

text_feature = df.iloc[:,11:19]

# for col in list(text_feature.columns):
#     data = text_feature[col].value_counts(bins = [i for i in range(-1,11)]).sort_index().to_frame().set_index(pd.Index([i for i in range(10)]+['>10']))
#     ax = data.plot(kind = 'bar',
#                    title = col.upper() + ' Counts in Text',
#                    rot = 0)
#     plt.savefig(path.plt_dir + col.lower() + ' count.jpg')

# sorting by date

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by = 'date', ascending = True).reset_index(drop = True)
df = df.drop(columns = 'date')

# output dataset

df.to_csv(path.data_dir + 'clean_dataset.csv', index = False)

