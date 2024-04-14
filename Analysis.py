import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import plotly.express as px

def display_genre_trend(genre_trend_data):
    genre_trend_data['year'] = genre_trend_data.index
    fig = px.line(genre_trend_data, x='year', y=genre_trend_data.columns[:-1],
                  title='Movie Popularity Trends Across Genres Over Time',
                  labels={'value': 'Movies', 'year': 'Year', 'variable': 'Genres'})
    fig.update_layout(xaxis=dict(type='category'), legend_title='Genres')
    fig.show()

def display_sentiment_analysis(sentiment_data):
    fig = px.box(sentiment_data, x='Genre', y='Sentiment Score', color='Genre',
                 title='Sentiment Analysis of Plot Summaries by Genre',
                 labels={'Sentiment Score': 'Sentiment Score', 'Genre': 'Genre'})
    fig.update_layout(yaxis=dict(title='Sentiment Score'))
    fig.show()

def filter_movies_without_plot(omdb_data):
    omdb_data = omdb_data[omdb_data['omdb_plot'] != 'N/A']
    omdb_data = omdb_data.reset_index(drop=True)
    return omdb_data

def filter_movies_without_publication_date(wikidata):
    wikidata = wikidata[wikidata['publication_date'].notnull()].copy()
    wikidata[['year', 'month', 'date']] = wikidata['publication_date'].str.split('-', expand=True)
    wikidata['year'] = wikidata['year'].astype(int)
    wikidata = wikidata[(wikidata['year'] < 2019) & (wikidata['year'] > 2000)]
    wikidata = wikidata.reset_index(drop=True)
    wikidata = wikidata[['year', 'imdb_id']]
    return wikidata

def clean_plot_text(omdb_data):
    cleaned_plot = omdb_data['omdb_plot'].str.lower().str.replace(r'[^\w\s\'-]', '')
    return cleaned_plot.tolist()

def extract_unique_genres(omdb_data):
    genres = np.concatenate(omdb_data['omdb_genres'])
    return np.unique(genres)

def display_text_length_histogram(omdb_data):
    plot_lengths = omdb_data['omdb_plot'].str.len()
    fig, ax = plt.subplots()
    plot_lengths.hist(bins=np.arange(0,5000,50), figsize=(16, 9), ax=ax)
    plt.title('Distribution of Plot Lengths')
    plt.xlabel('Characters', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(linestyle=':')
    plt.savefig('output/text_in_plot.png')
    plt.show()

def display_genre_occurrence(genres, count):
    genre_counts = pd.DataFrame({'genre':genres, 'count':np.sum(count, axis=0)})
    fig, ax = plt.subplots()
    genre_counts.plot(x='genre', y='count', kind='bar', legend=False, figsize=(16, 9), ax=ax)
    ax.set_axisbelow(True)
    ax.grid(linestyle=':')
    plt.title('Occurrences of Each Genre')
    plt.ylabel('Occurrences', fontsize=12)
    plt.xlabel('Genres', fontsize=12)
    plt.xticks(rotation=35, ha='right')
    plt.savefig('output/genre_vs_occurrence.png')
    plt.show()

def display_genre_popularity_trend(genre_trend_data):
    fig, ax = plt.subplots()
    genre_trend_data.plot(figsize=(16, 9), ax=ax)
    ax.set_axisbelow(True)
    ax.grid(linestyle=':')
    plt.title('Trends in Movie Popularity for Different Genres over Time')
    plt.ylabel('Movies', fontsize=12)
    plt.xlabel('Year', fontsize=12)
    plt.legend(title='Genres')
    plt.savefig('output/genre_trend.png')
    plt.show()

def find_best_parameters(classifier, X_train, X_test, y_train, y_test, genres):
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__min_df': (0.005, 0.01, 0.015),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__estimator__C': [0.01, 0.1, 1],
        'clf__estimator__class_weight': ['balanced', None],
    }

    grid_search_tune = GridSearchCV(classifier, parameters, cv=3, n_jobs=2, verbose=3)
    grid_search_tune.fit(X_train, y_train)

    best_classifier = grid_search_tune.best_estimator_
    predictions = best_classifier.predict(X_test)

    print(classification_report(y_test, predictions, target_names=genres))
    print()
    print("Best parameters:")
    print(grid_search_tune.best_estimator_.steps)
    print()

def print_predictions(mlb, X_test, predictions):
    target_labels = mlb.inverse_transform(predictions)
    for item, labels in zip(X_test, target_labels):
        print('{0}... => {1}\n'.format(item[0:40], ', '.join(labels)))

def main():
    omdb_data = pd.read_json('data/omdb-data.json.gz', orient='record', lines=True, encoding='utf-8')
    wikidata = pd.read_json('data/wikidata-movies.json.gz', orient='record', lines=True, encoding='utf-8')
    
    omdb_data = filter_movies_without_plot(omdb_data)
    genres = extract_unique_genres(omdb_data)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(omdb_data['omdb_genres'])
    X = clean_plot_text(omdb_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    display_genre_occurrence(genres, y)
    display_text_length_histogram(omdb_data)

    classifier = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.25, min_df=0.005, ngram_range=(1, 1))),
        ('clf', OneVsRestClassifier(LinearSVC())),
    ])

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    print('==========================================================================')
    print('Summary of Prediction Accuracy')
    print('==========================================================================')
    print(classification_report(y_test, predictions, target_names=genres))
    print('The warning means that there is no F-score to calculate for the specified\n'
        'labels, so the F-score for them are considered to be 0.0. Since we wanted\n'
        'an average of the score, we must take into account that the score of 0 was\n'
        'included in the calculation; hence scikit-learn displays the warning.\n')
    print('The reason for the missing values is that some labels in y_test (truth) do\n'
        'not appear in predictions. In other words, the labels are never predicted.')
    print('--------------------------------------------------------------------------\n')

    wikidata = filter_movies_without_publication_date(wikidata)
    genre_trend_data = pd.merge(omdb_data, wikidata, on='imdb_id')
    genre_trend_data = pd.DataFrame({
        'genre': np.concatenate(genre_trend_data['omdb_genres']),
        'year': np.repeat(genre_trend_data['year'], genre_trend_data['omdb_genres'].str.len()),
    })
    top_genres = genre_trend_data.groupby('genre')['year'].count().reset_index(name='count')
    top_genres = top_genres.sort_values('count', ascending=False).reset_index()
    top_genres = top_genres['genre'].tolist()[:10]

    genre_trend_data = genre_trend_data.groupby(['year', 'genre'])['genre'].count()
    genre_trend_data = genre_trend_data.unstack()
    genre_trend_data = genre_trend_data[top_genres]
    display_genre_trend(genre_trend_data)
    
    display_genre_trend(genre_trend_data)

if __name__ == "__main__":
    main()
