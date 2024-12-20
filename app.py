import pandas as pd
import numpy as np
np.Inf = np.inf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from IPython.display import HTML
import dash
from dash import dash_table
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
from dash import ctx
import io
import base64
from io import BytesIO
import time
import re
from collections import Counter

company_colors = {
        'Smartcat': '#905fef',
        'Lokalise': '#fc5608',
        'Crowdin': '#8c9494',
        'Phrase Localization Platform': '#00eab4',
        'Murf.ai': '#df3a93',
        'Transifex': '#076bd1'}

sentiment_class_color_map = {"Positive": "#96D289", "Negative": "#FF7BC4", "Neutral": "#6c757d"}
sentiment_scNeg_color_map = {"pos_cons": "#96D289", "neg_cons": "#FF7BC4", "neu_cons": "#6c757d"}
sentiment_scPos_color_map = {"pos_pros": "#96D289", "neg_pros": "#FF7BC4", "neu_pros": "#6c757d"}

ratings_color_map = {
    0.5: '#ff70bf',
    1: '#ff7bc4',
    1.5: '#ff85c8',
    2: '#ffa8d9',
    2.5: '#ffcae9',
    3: '#dbfdd8',
    3.5: '#cdf5b7',
    4: '#bbefa9',
    4.5: '#a9e99b',
    5: '#96d289'}

def clean_text(text, stopwords):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = text.lower().split()
    return ' '.join([word for word in words if word not in stopwords])
        
data = pd.read_csv('https://raw.githubusercontent.com/TatKhachatryan/G2ReviewsDashApp/refs/heads/main/final_data_G2_Reviews.csv')

data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
unique_products = data['product_name'].unique()

external_stylesheets = [dbc.themes.PULSE]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

server = app.server

app.title = 'Localization Tools Analysis Dashboard'

app.layout = html.Div([
    html.Div(style={
        'borderBottom': '1px solid #ffffff',
        'margin': '20px'
    }),
    html.H1("Localization Tools Analysis Dashboard", style={"text-align": "center", "margin": "auto"}),
    html.Div(style={
        'borderBottom': '1px solid #ffffff',
        'margin': '20px'
    }),

    # introduction block
    html.Div(
        children=[
            html.P(
                "This project focuses on analyzing reviews for 6 companies: Crowdin, Lokalise, Smartcat, Transifex, Murf.ai & Phrase Localization Platform.",
                style={'marginBottom': '10px'}
            ),
            html.P(
                "The data was scraped from g2.com, a popular platform for software and business solution reviews.",
                style={'marginBottom': '10px'}
            ),
            html.P(
                "The objective is to delve into the reviews to identify both positive and negative feedback, providing actionable insights and recommendations for improvement.",
                style={'marginBottom': '20px'}
            ),
            dbc.Button(
                "Click to Read the Summary Article",
                href="https://foggy-hope-121.notion.site/Localization-Tools-Reviews-Analysis-NLP-Project-15168dd6126e80d58e7bce2971854839",  # Replace with your actual Notion link
                target="_blank",  # Opens the link in a new tab
                color="success",  # Button color
                style={
                    "borderRadius": "10px",
                    "padding": "10px 20px",
                    "fontSize": "16px"
                }
            )
        ],
        style={
            'backgroundColor': '#f9f9f9',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)',
            'textAlign': 'center'
        }
    ),

    html.Div(style={'margin': '40px 0'}),
    # general stats
    html.H2("General Statistics", style={
        'backgroundColor': '#BBEFA9',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)',
        'textAlign': 'center'
    }),
    # year slider
    html.Div([
        html.H4("Select a Year Range"),
        dcc.RangeSlider(
            id='year-slicer',
            min=data['year'].min(),
            max=data['year'].max(),
            value=[2019, data['year'].max()],
            marks={year: str(year) for year in range(data['year'].min(), data['year'].max() + 1)},
            step=1
        )
    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px 0',
              'border': '1px solid black', 'borderRadius': '10px', 'backgroundColor': '#f8fff7'}),
    # company dropdown menu
    html.Div([
        html.H4("Select or de-select a Company:"),
        dcc.Dropdown(
            id='company-dropdown',
            options=[{'label': cat, 'value': cat} for cat in data['product_name'].unique()],
            value=list(data['product_name'].unique()[:]),
            multi=True)

    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px 0'}),

    html.Div([
        dcc.Graph(id='ratings-histogram', style={'flex': '1 1 50%', 'min-width': '300px', 'margin': '10px'}),
        dcc.Graph(id='ratings-percentage-bar', style={'flex': '1 1 50%', 'min-width': '300px', 'margin': '10px'}),
    ], style={'display': 'flex', 'width': '80%', 'margin': 'auto', 'padding': '20px 0',
              'justify-content': 'space-between'}),

    html.Div([
        dcc.Graph(id='review-line-chart'),
    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px 0'}),

    # sentiment analysis
    html.H2("Sentiment Analysis", style={
        'backgroundColor': '#BBEFA9',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)',
        'textAlign': 'center'
    }),

    html.Div([
        html.H4("Choose Sentiment Visualization:"),
        dcc.RadioItems(
            id='sentiment-toggle',
            options=[
                {'label': 'Sentiment Classes', 'value': 'classes'},
                {'label': 'Sentiment Scores', 'value': 'scores'}
            ],
            value='classes',  # Default value
            labelStyle={'display': 'inline-block', 'margin-right': '20px'},
        )
    ], style={'textAlign': 'center', 'margin-top': '20px'}),
    # graph placeholders for sentiment visualizations
    html.Div([
        dcc.Graph(id='sentiment-pro', style={'flex': '1 1 50%', 'min-width': '300px', 'margin': '10px'}),
        dcc.Graph(id='sentiment-cons', style={'flex': '1 1 50%', 'min-width': '300px', 'margin': '10px'}),
    ], style={'display': 'flex', 'width': '80%', 'margin': 'auto', 'padding': '20px 0',
              'justify-content': 'space-between'}),

    # topic modeling
    html.H2("Topic Modeling", style={
        'backgroundColor': '#BBEFA9',
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0px 4px 6px rgba(0,0,0,0.1)',
        'textAlign': 'center'
    }),

    html.Div([
        html.H4("Select or de-select a Company:"),
        dcc.Dropdown(
            id='company-dropdown1',
            options=[{'label': cat, 'value': cat} for cat in data['product_name'].unique()],
            value=list(data['product_name'].unique()[:]),
            multi=True)

    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px 0'}),

    html.Div([dcc.Graph(id='treemap-pro'), ], style={'width': '80%', 'margin': 'auto', 'padding': '20px 0'}),
    html.Div([dcc.Graph(id='treemap-cons'), ], style={'width': '80%', 'margin': 'auto', 'padding': '20px 0'}),

    html.H4("WordCloud Plots for both Pros and Cons Reviews", style={"text-align": "center", "margin": "auto"}),
    html.Div([
        html.H4("Select a Company:"),
        dcc.Dropdown(
            id='company-dropdown2',
            options=[{'label': cat, 'value': cat} for cat in data['product_name'].unique()],
            value="Smartcat",
            multi=False)

    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px 0'}),

    html.Div([
        html.Img(id='wordcloud-pro', style={'flex': '1 1 50%', 'min-width': '300px', 'margin': '10px'}),
        html.Img(id='wordcloud-cons', style={'flex': '1 1 50%', 'min-width': '300px', 'margin': '10px'}),
    ], style={'display': 'flex', 'width': '80%', 'margin': 'auto', 'padding': '20px 0',
              'justify-content': 'space-between'}),

    # data download button
    html.Div([
        html.Button("Download the Data", id="download-btn", n_clicks=0,
                    style={"color": "#000000", "text-decoration": "underline",
                           'border': '1px solid black', 'borderRadius': '10px',
                           'padding': '20px',
                           'backgroundColor': '#DBFDD8'}),
        dcc.Download(id="download-dataframe-xlsx")
    ], style={'text-align': 'center', 'padding': '20px', 'font-size': '30px'}),
])


@app.callback(
    Output('treemap-pro', 'figure'),
    Output('treemap-cons', 'figure'),
    Input('year-slicer', 'value'),
    Input('company-dropdown1', 'value')
)
def update_graph2(year_slicer, company_dropdown):
    # data aggregation
    df = data[(data['product_name'].isin(company_dropdown)) &
              (data['date'].dt.year >= year_slicer[0]) & (data['date'].dt.year <= year_slicer[1])]

    pros_categories = df.groupby(['product_name', 'Pros_Categories'])['Pros_Categories'].count().rename(
        'Pros_Categories_count').to_frame().reset_index()
    pros_categories.sort_values(by=['product_name', 'Pros_Categories_count'], ascending=False, inplace=True)
    pros_categories = pros_categories.reset_index(drop=True)

    pros_categories['Total_Count'] = pros_categories.groupby('product_name')['Pros_Categories_count'].transform('sum')

    pros_categories['Percentage'] = (pros_categories['Pros_Categories_count'] / pros_categories['Total_Count']) * 100

    cons_categories = df.groupby(['product_name', 'Cons_Categories'])['Cons_Categories'].count().rename(
        'Cons_Categories_count').to_frame().reset_index()
    cons_categories.sort_values(by=['product_name', 'Cons_Categories_count'], ascending=False, inplace=True)
    cons_categories = cons_categories.reset_index(drop=True)

    cons_categories['Total_Count'] = cons_categories.groupby('product_name')['Cons_Categories_count'].transform('sum')

    cons_categories['Percentage'] = (cons_categories['Cons_Categories_count'] / cons_categories['Total_Count']) * 100
    # graphs
    treemap_pros = px.treemap(
        pros_categories,
        path=['product_name', 'Pros_Categories'],
        values='Percentage',
        title='Proportional Treemap of Positive Topics by Company', height=700,
        color='product_name',
        color_discrete_map=company_colors)

    treemap_cons = px.treemap(
        cons_categories,
        path=['product_name', 'Cons_Categories'],
        values='Percentage',
        title='Proportional Treemap of Negative Topics by Company', height=700,
        color='product_name',
        color_discrete_map=company_colors)

    return treemap_pros, treemap_cons


@app.callback(
    Output('ratings-histogram', 'figure'),
    Output('ratings-percentage-bar', 'figure'),
    Output('review-line-chart', 'figure'),
    Input('year-slicer', 'value'),
    Input('company-dropdown', 'value')
)
def update_graph1(year_slicer, company_dropdown):
    df = data[(data['product_name'].isin(company_dropdown)) &
              (data['date'].dt.year >= year_slicer[0]) & (data['date'].dt.year <= year_slicer[1])]
    ratings_hist = px.histogram(df, x="rating", color='rating', color_discrete_map=ratings_color_map,
                                title=f'Total Ratings Distribution in {year_slicer[0]}-{year_slicer[-1]}')
    ratings_hist.update_layout(plot_bgcolor="#F0F0F0", paper_bgcolor="#FFFFFF")

    ratings = df.groupby(['product_name', 'rating'])['rating'].count().rename('count').to_frame().reset_index()
    ratings['Total_Count'] = ratings.groupby('product_name')['count'].transform('sum')
    ratings['Percentage'] = (ratings['count'] / ratings['Total_Count']) * 100
    ratings_percentage_bar = px.bar(ratings, x="rating", y="Percentage",
                                    color='product_name', color_discrete_map=company_colors, barmode='group',
                                    title=f"Total Rating % per Company in {year_slicer[0]}-{year_slicer[-1]}")
    ratings_percentage_bar.update_layout(plot_bgcolor="#F0F0F0", paper_bgcolor="#FFFFFF")

    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')

    reviews = df.groupby(['product_name', 'year_month'])['product_name'].count().rename('Review_Count').reset_index()
    # Convert year_month from Period to Timestamp for compatibility with plotting
    reviews['year_month'] = reviews['year_month'].dt.to_timestamp()
    review_count_line = px.line(reviews, x="year_month", y="Review_Count", color="product_name",
                                color_discrete_map=company_colors,
                                title=f"Number of Reviews in {year_slicer[0]}-{year_slicer[-1]}")
    review_count_line.update_layout(plot_bgcolor="#F0F0F0", paper_bgcolor="#FFFFFF", title_x=0.5)

    return ratings_hist, ratings_percentage_bar, review_count_line,


# Update callback to toggle between sentiment plots
@app.callback(
    Output('sentiment-pro', 'figure'),
    Output('sentiment-cons', 'figure'),
    Input('sentiment-toggle', 'value'),  # New input for sentiment toggle
    Input('year-slicer', 'value'),
    Input('company-dropdown', 'value')
)
def toggle_sentiment_plots(toggle_value, year_slicer, company_dropdown):
    df = data[(data['product_name'].isin(company_dropdown)) &
              (data['date'].dt.year >= year_slicer[0]) & (data['date'].dt.year <= year_slicer[1])]

    # generate sentiment class plots
    if toggle_value == 'classes':
        sentimentClassesPro = df.groupby(['product_name', 'sentiment_class_pros'])[
            'sentiment_class_pros'].count().rename(
            'Count').to_frame().reset_index()
        sentimentClassesPro['Total_Count'] = sentimentClassesPro.groupby('product_name')['Count'].transform('sum')
        sentimentClassesPro['Percentage'] = round(
            (sentimentClassesPro['Count'] / sentimentClassesPro['Total_Count']) * 100, 1)

        sentimentClassesCons = df.groupby(['product_name', 'sentiment_class_cons'])[
            'sentiment_class_cons'].count().rename(
            'Count').to_frame().reset_index()
        sentimentClassesCons['Total_Count'] = sentimentClassesCons.groupby('product_name')['Count'].transform('sum')
        sentimentClassesCons['Percentage'] = round(
            (sentimentClassesCons['Count'] / sentimentClassesCons['Total_Count']) * 100, 1)

        fig_classes_pro = px.bar(
            sentimentClassesPro,
            x='product_name',
            y='Percentage',
            color='sentiment_class_pros',
            color_discrete_map=sentiment_class_color_map,
            title='Sentiment Class Distribution by Companies (Pros)',
            barmode='group',
            text='Percentage'
        )
        fig_classes_pro.update_layout(yaxis_title="Percentage", xaxis_title="Product Name",
                                      plot_bgcolor="#F0F0F0", paper_bgcolor="#FFFFFF")

        fig_classes_cons = px.bar(
            sentimentClassesCons,
            x='product_name',
            y='Percentage',
            color='sentiment_class_cons',
            color_discrete_map=sentiment_class_color_map,
            title='Sentiment Class Distribution by Companies (Cons)',
            barmode='group',
            text='Percentage'
        )
        fig_classes_cons.update_layout(yaxis_title="Percentage", xaxis_title="Product Name",
                                       plot_bgcolor="#F0F0F0", paper_bgcolor="#FFFFFF")

        return fig_classes_pro, fig_classes_cons

    # generate sentiment score plots
    elif toggle_value == 'scores':
        sentiment_melted_pros = df.melt(
            id_vars=['product_name'],
            value_vars=['pos_pros', 'neg_pros', 'neu_pros'],
            var_name='Sentiment',
            value_name='Score'
        )
        sentiment_melted_cons = df.melt(
            id_vars=['product_name'],
            value_vars=['pos_cons', 'neg_cons', 'neu_cons'],
            var_name='Sentiment',
            value_name='Score'
        )
        sentimentScoresPro = sentiment_melted_pros.groupby(['product_name', 'Sentiment'])[
            'Score'].sum().to_frame().reset_index()
        sentimentScoresCons = sentiment_melted_cons.groupby(['product_name', 'Sentiment'])[
            'Score'].sum().to_frame().reset_index()

        sentimentScoresPro['Total_Score'] = sentimentScoresPro.groupby('product_name')['Score'].transform('sum')
        sentimentScoresPro['Percentage'] = round(
            ((sentimentScoresPro['Score'] / sentimentScoresPro['Total_Score']) * 100), 1)

        sentimentScoresCons['Total_Score'] = sentimentScoresCons.groupby('product_name')['Score'].transform('sum')
        sentimentScoresCons['Percentage'] = round(
            ((sentimentScoresCons['Score'] / sentimentScoresCons['Total_Score']) * 100), 1)

        fig_scores_pro = px.bar(
            sentimentScoresPro,
            x='product_name',
            y='Percentage',
            color='Sentiment',
            color_discrete_map=sentiment_scPos_color_map,
            title='Sentiment Score Distribution by Company (Pros)',
            barmode='group',
            text='Percentage'
        )
        fig_scores_pro.update_layout(yaxis_title="Percentage", xaxis_title="Product Name",
                                     plot_bgcolor="#F0F0F0", paper_bgcolor="#FFFFFF")

        fig_scores_cons = px.bar(
            sentimentScoresCons,
            x='product_name',
            y='Percentage',
            color='Sentiment',
            color_discrete_map=sentiment_scNeg_color_map,
            title='Sentiment Score Distribution by Company (Cons)',
            barmode='group',
            text='Percentage'
        )
        fig_scores_cons.update_layout(yaxis_title="Percentage", xaxis_title="Product Name",
                                      plot_bgcolor="#F0F0F0", paper_bgcolor="#FFFFFF")

        return fig_scores_pro, fig_scores_cons


@app.callback(
    Output('wordcloud-pro', 'src'),
    Output('wordcloud-cons', 'src'),
    Input('year-slicer', 'value'),
    Input('company-dropdown2', 'value')
)
def update_graph3(year_slicer, company_dropdown):
    df = data[(data['product_name'] == company_dropdown) &
              (data['date'].dt.year >= year_slicer[0]) & (data['date'].dt.year <= year_slicer[1])]

    # clean text for word clouds
    words_to_remove = ["translate", "translator", "translators", "translation", "translations", company_dropdown]
    df.loc[:, 'Cleaned_Pros'] = df['processed_Pros_NLTK'].apply(lambda x: clean_text(x, words_to_remove))
    df.loc[:, 'Cleaned_Cons'] = df['processed_Cons_NLTK'].apply(lambda x: clean_text(x, words_to_remove))

    # generate "Pros" word cloud
    pros_text = ' '.join(df['Cleaned_Pros'].dropna())
    pros_wordcloud = WordCloud(background_color='white', stopwords=words_to_remove, width=800, height=500).generate(
        pros_text)

    # create figure for Pros word cloud with title
    fig_pro, ax_pro = plt.subplots(figsize=(8, 5))
    ax_pro.imshow(pros_wordcloud, interpolation='bilinear')
    ax_pro.axis('off')  # Turn off axis
    ax_pro.set_title(f"Frequently occurring words in Pros for {company_dropdown}", fontsize=16)  # Title for Pros
    plt.tight_layout()

    # save Pros word cloud image to a BytesIO object
    pros_img = io.BytesIO()
    plt.savefig(pros_img, format='PNG')
    pros_img.seek(0)
    plt.close(fig_pro)

    # generate "Cons" word cloud
    cons_text = ' '.join(df['Cleaned_Cons'].dropna())
    cons_wordcloud = WordCloud(background_color='black', stopwords=words_to_remove, width=800, height=500).generate(
        cons_text)

    # create figure for Cons word cloud with title
    fig_cons, ax_cons = plt.subplots(figsize=(8, 5))
    ax_cons.imshow(cons_wordcloud, interpolation='bilinear')
    ax_cons.axis('off')  # Turn off axis
    ax_cons.set_title(f"Frequently occurring words in Cons for {company_dropdown}", fontsize=16)  # Title for Cons
    plt.tight_layout()

    # save Cons word cloud image to a BytesIO object
    cons_img = io.BytesIO()
    plt.savefig(cons_img, format='PNG')
    cons_img.seek(0)
    plt.close(fig_cons)

    # encode images as base64
    pros_src = 'data:image/png;base64,{}'.format(base64.b64encode(pros_img.getvalue()).decode())
    cons_src = 'data:image/png;base64,{}'.format(base64.b64encode(cons_img.getvalue()).decode())

    return pros_src, cons_src


@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("download-btn", "n_clicks"),
)
def download_excel(n_clicks):
    if n_clicks > 0:
        raw_data = data[['date', 'url', 'rating', 'product_name',
                         'sentiment_class_pros', 'sentiment_class_cons', 'Pros_Categories',
                         'Cons_Categories', 'sentiment_scores_pros', 'sentiment_scores_cons']]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            raw_data.to_excel(writer, index=False, sheet_name="Raw Data")
        output.seek(0)
        return dcc.send_bytes(output.getvalue(), "G2_Reviews_raw_data.xlsx")
    return None


if __name__ == '__main__':
    app.run_server(debug=True)
