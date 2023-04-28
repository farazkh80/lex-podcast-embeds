
import requests
import numpy as np
import regex as re
import pandas as pd
import plotly.express as px
import cohere
import streamlit as st
st.set_page_config(layout="wide")


# co = cohere.Client('') # This is your trial API key, uncomment this, comment line below
co = cohere.Client(st.secrets["co_key"])

def get_lexicon():
    url = 'https://karpathy.ai/lexicap/'
    r = requests.get(url)
    lexicon = r.text
    
    # find occurence of all strings of regex form <div><a?*+>d</a>
    res = re.findall(r'<div><a.*>1<\/a>.*<\/div>', lexicon)[0].split('</div>')[:-1]
    res = [re.sub('<div><a.*>\d*<\/a>\s*', '', eps) for eps in res]
    return res
    

def embed(texts):
    co = cohere.Client('R9T0BsZQiZFNWKjdRNJKoAhSk8OFjo5f3nWUfhWW') # This is your trial API key
    response = co.embed(
        model='large',
        texts=texts,
    )
    return response.embeddings


def pca_embeds(embeds):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(embeds)
    return pca.transform(embeds)


def read_embeds():
    df = pd.read_csv('embeds.csv')
    embeds = np.array([[x,y] for x,y in zip(df['x'].values, df['y'].values)])
    return embeds, df['episodes'].values


def save_embeds(embeds, episodes):
    df = pd.DataFrame(embeds, columns=['x', 'y'])
    df['episodes'] = episodes
    df.to_csv('embeds.csv', index=False)


def find_embed_clusters(embeds, n_clusters=10):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeds)
    return kmeans.labels_


def group_clusters(labels, episodes):
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(episodes[i])
    
    # concat the list of all cluster episodes
    for label in clusters:
        clusters[label] = ', '.join(clusters[label])

    assert len(clusters.keys()) == 10

    cluster_summary={}

    for i, l in enumerate(clusters.keys()):
        # find the top 4 keywords for each cluster by using a hashtable
        keywords = {}
        common_ignore=set(['','lex', 'fridman', 'podcast', 'and', 'or', 'the', 'of', "&", "to"])
        for word in clusters[l].split(r' '):
            word = word.strip(r',|.|\s|"|\'').lower()
            if word in common_ignore: continue
            if word not in keywords:
                keywords[word] = 0
            keywords[word] += 1


        top_keywords = sorted(keywords, key=keywords.get, reverse=True)[:3]
        cluster_summary[l] = ",".join(top_keywords)
    

    cluster_summary = [cluster_summary[i] for i in range(len(cluster_summary))]
    return cluster_summary
 

def visualize_clusters_plotly(embeds, episodes, n_clusters=10):
    labels = find_embed_clusters(embeds, n_clusters)
    # add colors
    cluster_summary = group_clusters(labels, episodes)
    colors = [cluster_summary[i] for i in labels]
    fig = px.scatter(x=embeds[:,0], y=embeds[:,1], color=colors, hover_name=episodes, width=5000, height=580)
    st.plotly_chart(fig, use_container_width=True)



if __name__=="__main__":
    st.title("Lex Fridman Podcast Episode Titles' Semantic Relevance")
    st.write("Made by [Faraz](https://twitter.com/FarazDoTAI) using [Cohere](https://cohere.ai/) embedding models and [Andrej Karpathy's Lexicap](https://karpathy.ai/lexicap/) trasncripts of [Lex Fridman Podcast](https://lexfridman.com/podcast/)")
    st.write("Code can be found [here](https://github.com/farazkh80/lex-podcast-embeds/tree/master)")
    
    # To compute and save embeddings
    # with st.spinner("Computing Embeddings..."):
        # episodes = get_lexicon()
        # embeds = embed(episodes)
        # embeds = pca_embeds(embeds) # default is 2D
        # save_embeds(embeds, episodes)
    
    # To read embeddings from csv and visualize
    with st.spinner("Visualizing Embeddings..."):
        embeds, episodes = read_embeds()
        visualize_clusters_plotly(embeds, episodes,n_clusters=10)

    
    
   