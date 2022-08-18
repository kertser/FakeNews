# Kill the warnings:
import warnings
warnings.filterwarnings(action='ignore')

from nicegui import ui
import numpy as np # linear algebra
import pandas as pd # data processing
from textblob import TextBlob, Word
from statistics import mean
import scipy.stats
from scipy.stats import pearsonr,spearmanr

import joblib
import config

Emotions = ['Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']
emLoad = {'Anger':100*1/8,'Anticipation':100*1/8,'Disgust':100*1/8,'Fear':100*1/8,'Joy':100*1/8,'Sadness':100*1/8,'Surprise':100*1/8,'Trust':100*1/8}
wordsData = pd.read_excel(config.wordsData_url, index_col=0)
wordsData = wordsData[wordsData.columns.intersection(['English Word']+[emotion for emotion in Emotions])]

# Load the latest classifier:
#model = joblib.load("model.pkl")
model = joblib.load("model1a.pkl")

#%% --- Feature Construction ---
def feature_wordsCount(df_row, Sentence, df):
    # count the unique words in the Sentence and calculate the ratio
    uniqueWords = len(set(Sentence.words))
    totalWords = len((Sentence.words))
    df.at[df_row,'uniqe_words_ratio']=uniqueWords/totalWords

def feature_nounPolarity(df_row, Sentence, df):
    # Add feature for sum of polarity index into the dataset
    # df_row is an index of the row in the dataframe
    #Sentence = TextBlob(fake_news_full_df['text'][df_row]).correct()
    try:
        df.at[df_row,'nounPolarity'] = mean([TextBlob(nounS).sentiment.polarity for nounS in Sentence.noun_phrases])
    except:
        df.at[df_row,'nounPolarity'] = 0 # No nouns found

def feature_nounSubjectivity(df_row, Sentence, df):
    # Add feature for sum of subjectivity index into the dataset
    # df_row is an index of the row in the dataframe
    #Sentence = TextBlob(fake_news_full_df['text'][df_row]).correct()
    try:
        df.at[df_row,'nounSubjectivity'] = mean([TextBlob(nounS).sentiment.subjectivity for nounS in Sentence.noun_phrases])
    except:
        df.at[df_row,'nounSubjectivity'] = 0 # No nouns found

def feature_sentenceSentiment(df_row, Sentence, df):
    # Entire phrase sentiment analysis
    # df_row is an index of the row in the dataframe
    #Sentence = TextBlob(fake_news_full_df['text'][df_row]).correct()
    polarity, subjectivity = Sentence.sentiment
    df.at[df_row,'sentencePolarity'] = polarity
    df.at[df_row,'sentenceSubjectivity'] = subjectivity
    df.at[df_row,'meanPolarity_per_sentence'] = mean([sentence.polarity for sentence in Sentence.sentences])
    df.at[df_row,'meanSubjetivity_per_sentence'] = mean([sentence.subjectivity for sentence in Sentence.sentences])

def feature_Emotions(df_row, Sentence, df):
    # Insert the emotional count per words into dataset
    # df_row is an index of the row in the dataframe
    # WordsData is the English dataset, one-hot-encoded for emotions

    # Reset emotions for the selected row
    for emotion in Emotions:
        df.at[df_row,emotion]=0

    for word in [Word(word).singularize().lemmatize() for word in Sentence.words if word in wordsData.index]:
        try:
            for emotion in set(wordsData.columns[(wordsData[wordsData.index == word].values == 1)[0]].tolist()):
                df.at[df_row,emotion]+=1
        except:
            pass # no emotonal load for that specific word

def frequency_Analysis(df_row, Sentence, df):
    # Emotional load converting to frequency and amplitude
    # df_row is an index of the row in the dataframe

    #Sentence = TextBlob(fake_news_full_df['text'][df_row]).correct()
    data1 = np.array([sentence.polarity for sentence in Sentence.sentences]) # Sentence polarity
    data2 = np.array([sentence.subjectivity for sentence in Sentence.sentences]) # Sentence subjectivity
    sentence_timing = [len(sentence.words) for sentence in Sentence.sentences] # Sentence timing

    #Frequency Analysis:
    ps1 = np.abs(np.fft.fft(data1))**2
    ps2 = np.abs(np.fft.fft(data2))**2

    time_step = 1 / np.average(sentence_timing)
    freqs1 = np.fft.fftfreq(data1.size, time_step)
    freqs2 = np.fft.fftfreq(data2.size, time_step)

    MaxPolarityFrequency = round(max(freqs1),2) # Feature
    MaxSubjectivityFrequency = round(max(freqs2),2) # Feature

    df.at[df_row,'MaxPolarityFrequency'] = MaxPolarityFrequency
    df.at[df_row,'MaxSubjectivityFrequency'] = MaxSubjectivityFrequency

def correlation_and_entropy(df_row,Sentence,df):
    # Test for mutual correlation of sentences polarity and subjectivity
    # df_row is an index of the row in the dataframe

    #Sentence = TextBlob(fake_news_full_df['text'][df_row]).correct()
    data1 = np.array([sentence.polarity for sentence in Sentence.sentences]) # Sentence polarity
    data2 = np.array([sentence.subjectivity for sentence in Sentence.sentences]) # Sentence subjectivity

    # Peason correlation between polarity and subjectivity - Feature
    try:
        corrP, _ = pearsonr(data1, data2)
    except:
        corrP = 0 # less than 2 elements for correlation
    # Spearman correlation between polarity and subjectivity - Feature
    try:
        corrS, _ = spearmanr(data1, data2)
    except:
        corrS = 0 # less than 2 elements for correlation

    # Calculate entropy of words in the sentence
    p_data = pd.DataFrame(Sentence.words).value_counts()
    try:
        entropy = scipy.stats.entropy(p_data)
    except:
        entropy = 0 # No data for entropy calculation

    df.at[df_row,'corrP'] = corrP
    df.at[df_row,'corrS'] = corrS
    df.at[df_row,'entropy'] = entropy

def construct_Features(indexRange,df,correct=True):
    # Construct the features
    for row in indexRange:
        #print(f'Constructing features for row #{row} out of {len(df)}:')
        try:
            if correct:
                Sentence = TextBlob(df['text'][row]).correct()
            else:
                Sentence = TextBlob(df['text'][row])

            feature_wordsCount(row,Sentence,df)
            feature_nounPolarity(row, Sentence,df)
            feature_nounSubjectivity(row, Sentence,df)
            feature_sentenceSentiment(row, Sentence,df)
            feature_Emotions(row, Sentence, df)
            frequency_Analysis(row, Sentence, df)
            correlation_and_entropy(row, Sentence, df)
        except:
            print(f'row #{row} contains some bugs, skipping')
def detect():
    #ui.notify('Wait for the AI to finish the analysis', close_button='OK')
    ui.colors(primary='#555') # Make all gray

    testDF = pd.DataFrame(columns=['text', 'uniqe_words_ratio', 'nounPolarity', 'nounSubjectivity',
                                   'sentencePolarity', 'sentenceSubjectivity', 'meanPolarity_per_sentence',
                                   'meanSubjetivity_per_sentence', 'Anger', 'Anticipation', 'Disgust',
                                   'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'MaxPolarityFrequency',
                                   'MaxSubjectivityFrequency', 'corrP', 'corrS', 'entropy'])
    testDF.at[0,'text'] = str(textInput.value)
    construct_Features(range(1),testDF,correct=True)

    emLoad['Anger'] = testDF['Anger'][0]
    emLoad['Anticipation'] = testDF['Anticipation'][0]
    emLoad['Disgust'] = testDF['Disgust'][0]
    emLoad['Fear'] = testDF['Fear'][0]
    emLoad['Joy'] = testDF['Joy'][0]
    emLoad['Sadness'] = testDF['Sadness'][0]
    emLoad['Surprise'] = testDF['Surprise'][0]
    emLoad['Trust'] = testDF['Trust'][0]
    sumEmotions = sum(emLoad.values())

    if sumEmotions != 0:
        #table.visible = True
        #chart.visible = True

        chart.options.series[0].data[0]['y'] = emLoad['Anger']/sumEmotions * 100
        chart.options.series[0].data[1]['y'] = emLoad['Anticipation']/sumEmotions * 100
        chart.options.series[0].data[2]['y'] = emLoad['Disgust']/sumEmotions * 100
        chart.options.series[0].data[3]['y'] = emLoad['Fear']/sumEmotions * 100
        chart.options.series[0].data[4]['y'] = emLoad['Joy']/sumEmotions * 100
        chart.options.series[0].data[5]['y'] = emLoad['Sadness']/sumEmotions * 100
        chart.options.series[0].data[6]['y'] = emLoad['Surprise']/sumEmotions * 100
        chart.options.series[0].data[7]['y'] = emLoad['Trust']/sumEmotions * 100
        chart.update()

        table.options.rowData[0].value = str(round(emLoad['Anger']/sumEmotions * 100,1)) + '%'
        table.options.rowData[1].value = str(round(emLoad['Anticipation'] / sumEmotions * 100,1)) + '%'
        table.options.rowData[2].value = str(round(emLoad['Disgust'] / sumEmotions * 100,1)) + '%'
        table.options.rowData[3].value = str(round(emLoad['Fear'] / sumEmotions * 100,1)) + '%'
        table.options.rowData[4].value = str(round(emLoad['Joy'] / sumEmotions * 100,1)) + '%'
        table.options.rowData[5].value = str(round(emLoad['Sadness'] / sumEmotions * 100,1)) + '%'
        table.options.rowData[6].value = str(round(emLoad['Surprise'] / sumEmotions * 100,1)) + '%'
        table.options.rowData[7].value = str(round(emLoad['Trust'] / sumEmotions * 100,1)) + '%'
        table.update()

        classifier = model.predict(testDF.drop(['text'], axis=1).astype(float))[0]

        if classifier == True:
            ui.notify('The news sentence is probably True', close_button='OK', position='center')
        else:
            ui.notify('The news sentence is probably Fake', close_button='OK', position='center')

    else:
        #table.visible = False
        #chart.visible = False
        ui.notify('The data is insufficient', close_button='OK', position='center')

    ui.colors() # Reset colors

def clear():
   textInput.value=''

#%% --- Main Frame ---
ui.colors()

# Main Window:
with ui.column().classes('no-warp'):
    with ui.row().classes('no-wrap'):
        with ui.card().classes('bg-yellow-300 h-80'):
            with ui.column().classes('w-full -my-3'):
                ui.markdown('### Fake News Detector\nBy means of Emotional Analysis (just for lulz)').classes('self-start self-center')
                textInput = ui.input(
                    label='Input text and press EVALUATE to start',
                    placeholder='Let''s see whether it is Fake?',
                ).classes('w-96 max-h-52 -my-3').props('type=textarea outlined height=300px')
                with ui.row().classes('w-full justify-between no-wrap place-self-end'):
                    ui.button('Evaluate', on_click=detect)
                    ui.button('Clear Text', on_click=clear)

        with ui.card().classes('bg-yellow-300 w-56 no-wrap h-80'):
            table = ui.table({
                'columnDefs': [
                    {'headerName': 'Emotion', 'field': 'emotion'},
                    {'headerName': 'Value', 'field': 'value'},
                ],
                'rowData': [
                    {'emotion': Emotions[0], 'value': '12.5%'},
                    {'emotion': Emotions[1], 'value': '12.5%'},
                    {'emotion': Emotions[2], 'value': '12.5%'},
                    {'emotion': Emotions[3], 'value': '12.5%'},
                    {'emotion': Emotions[4], 'value': '12.5%'},
                    {'emotion': Emotions[5], 'value': '12.5%'},
                    {'emotion': Emotions[6], 'value': '12.5%'},
                    {'emotion': Emotions[7], 'value': '12.5%'}
                ],
            }).classes('-my-3')
            table.options.__setattr__('suppressHorizontalScroll', True)

    with ui.card().classes('bg-yellow-300 no-wrap h-80 w-full'):
        ui.label('Emotional Chart of the Sentence:').classes('self-center')
        chart = ui.chart({
            'title': False,
            'exporting': {'enabled': False},
            'credits': {'enabled': False},
            'chart': {'type': 'pie','height': 270},
            'plotOptions': {
                'pie': {
                    'allowPointSelect': True,
                    'cursor': 'pointer',
                    'dataLabels': {
                        'enabled': True,
                        'format': '<b>{point.name}</b>: {point.percentage:.1f} %'
                    }
                }
            },
            'tooltip': {
                'pointFormat': '{series.data.name} <b>{point.percentage:.1f}%</b>'
            },
            'series': [{
                'data':[{
                        'name': Emotions[0],
                        'y': emLoad[Emotions[0]],
                        }, {
                            'name': Emotions[1],
                            'y': emLoad[Emotions[1]],
                        }, {
                            'name': Emotions[2],
                            'y': emLoad[Emotions[2]]
                        }, {
                            'name': Emotions[3],
                            'y': emLoad[Emotions[3]]
                        }, {
                            'name': Emotions[4],
                            'y': emLoad[Emotions[4]]
                        }, {
                            'name': Emotions[5],
                            'y': emLoad[Emotions[5]]
                        }, {
                            'name': Emotions[6],
                            'y': emLoad[Emotions[6]]
                        }, {
                            'name': Emotions[7],
                            'y': emLoad[Emotions[7]]
                        }]

            }],
        })

ui.html('<p>Alpha-Numerical, Mike Kertser, 2022, <strong>v0.02</strong></p>').classes('no-wrap')

if __name__ == "__main__":
    #ui.run(title='Fake-News Tool', host='127.0.0.1', reload=False, show=True)
    ui.run(title='Fake-News Tested', reload=True, show=True)
