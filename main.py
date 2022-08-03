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
wordsData = pd.read_excel(config.wordsData_url, index_col=0)
wordsData = wordsData[wordsData.columns.intersection(['English Word']+[emotion for emotion in Emotions])]


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
async def detect():
    ui.colors(primary='#555') # Make all gray

    testDF = pd.DataFrame(columns=['text', 'uniqe_words_ratio', 'nounPolarity', 'nounSubjectivity',
                                   'sentencePolarity', 'sentenceSubjectivity', 'meanPolarity_per_sentence',
                                   'meanSubjetivity_per_sentence', 'Anger', 'Anticipation', 'Disgust',
                                   'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'MaxPolarityFrequency',
                                   'MaxSubjectivityFrequency', 'corrP', 'corrS', 'entropy'])
    testDF.at[0,'text'] = str(textInput.value)
    construct_Features(range(1),testDF,correct=True)
    classifier = model.predict(testDF.drop(['text'],axis=1).astype(float))[0]
    print(classifier)
    ui.colors() # Reset colors
    await dialog
    if classifier==True:
        ui.notify('The news sentence is probably True',close_button='OK',position='center')
    else:
        ui.notify('The news sentence is probably Fake',close_button='OK',position='center')

def clear():
   textInput.value=''

#%% --- Main Frame ---
ui.colors()

# Main Window:
with ui.row().classes('flex no-wrap'):
    with ui.card().classes('bg-yellow-300 h-96'):
        with ui.column().classes('w-full -my-5'):
            ui.markdown('### Fake News Detector\nBy means of Emotional Analysis (just for lulz)').classes('self-start self-center')
            textInput = ui.input(
                label='Input text and press EVALUATE to start',
                placeholder='Let''s see whether it is Fake?',
            ).classes('w-96 h-52 max-h-52').props('type=textarea outlined')
            with ui.row().classes('w-full justify-between no-wrap'):
                ui.button('Evaluate', on_click=detect)
                ui.button('Clear Text', on_click=clear)
    with ui.card().classes('bg-yellow-300 w-56 no-wrap h-96'):
        table = ui.table({
            'columnDefs': [
                {'headerName': 'Emotion', 'field': 'emotion'},
                {'headerName': 'Value', 'field': 'value'},
            ],
            'rowData': [
                {'emotion': Emotions[0], 'value': 0},
                {'emotion': Emotions[1], 'value': 0},
                {'emotion': Emotions[2], 'value': 0},
                {'emotion': Emotions[3], 'value': 0},
                {'emotion': Emotions[4], 'value': 0},
                {'emotion': Emotions[5], 'value': 0},
                {'emotion': Emotions[6], 'value': 0},
                {'emotion': Emotions[7], 'value': 0},
                {'emotion': 'Polarity', 'value': 0},
                {'emotion': 'Entropy', 'value': 0}
            ],
        }).classes('-my-3')
        table.options.__setattr__('suppressHorizontalScroll', True)
ui.html('<p>Alpha-Numerical, Mike Kertser, 2022, <strong>v0.01</strong></p>').classes('no-wrap')

if __name__ == "__main__":
    # Load the latest classifier:
    model = joblib.load("model.pkl")
    ui.run(title='Fake-News Tool', host='127.0.0.1', reload=False, show=True)
    #ui.run(title='Fake-News Tested', reload=True, show=True)