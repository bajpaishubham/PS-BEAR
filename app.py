import streamlit as st
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="PS-BEAR",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "PS-BEAR - Polar Version"
    }
)

st.title(":blue[Polar PS-BEAR]")

query = st.text_input(":blue[Enter the physics problem here]", "")
st.write(":blue[The problem entered is:]")
st.write(query)

documents = []
subtopics = []

with open('json/full_dict.json') as f:
    full_dict = json.load(f)

for topic_index in full_dict:
    topic = full_dict[topic_index]["Topic_short"]
    subtopic_dict = full_dict[topic_index]["SubTopics"]
    for subtopic_index in subtopic_dict:
        subtopics += [subtopic_dict[subtopic_index]["SubTopic"]]
        html_filename = "html/{topic}/{idx}.html".format(topic = topic, idx = subtopic_index)
        html_content = open(html_filename).read()
        soup = BeautifulSoup(html_content, 'html.parser')
        text = ""
        for para in soup.find_all('p'):
            text = text + para.get_text()

        text = text.replace('\n', '')
        documents += [text]

tfidf = TfidfVectorizer(stop_words='english')
docs_tfidf = tfidf.fit_transform(documents)

# Similarity Computation

# Random problems
# 1.276 A horizontally oriented uniform disc of mass M and radius R rotates freely about a stationary vertical axis passing through its centre. The disc has a radial guide along which can slide without friction a small body of mass m. A light thread running down through the hollow axle of the disc is tied to the body. Initially the body was located at the edge of the disc and the whole system rotated with an angular velocity ω0. Then by means of a force F applied to the lower end of the thread the body was slowly pulled to the rotation axis. Find:(a) the angular velocity of the system in its final state;(b) the work performed by the force F.
# 1.358 The velocity components of a particle moving in the xy plane of the reference frame K are equal to vx and vi,. Find the velocity v' of this particle in the frame K' which moves with the velocity V relative to the frame K in the positive direction of its x axis.
# 1.149 Two small discs of masses m1 and m2 interconnected by a weightless spring rest on a smooth horizontal plane. The discs are set in motion with initial velocities v1 and v2 whose directions are mutually perpendicular and lie in a horizontal plane. Find the total energy E of this system in the frame of the centre of inertia.
# 1.86 A ball suspended by a thread swings in a vertical plane so that its acceleration values in the extreme and the lowest position are equal. Find the thread deflection angle in the extreme position.
# 1.31 A ball starts falling with zero initial velocity on a smooth inclined plane forming an angle α with the horizontal. Having fallen the distance h, the ball rebounds elastically off the inclined plane. At what distance from the impact point will the ball rebound for the second time?
# 1.179 A rocket moves in the absence of external forces by ejecting a steady jet with velocity u constant relative to the rocket. Find the velocity v of the rocket at the moment when its mass is equal to m, if at the initial moment it possessed the mass mo and its velocity was equal to zero. Make use of the formula given in the foregoing problem.

query_tfidf = tfidf.transform([query])
cosine_similarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()

num_top_elements = 10
related_docs_indices = cosine_similarities.argsort()[-num_top_elements:][::-1]

st.write(":blue[Top matches:]")
matches = []

for i in related_docs_indices:
    st.write(subtopics[i], " - ", cosine_similarities[i])
    matches = matches + [{"subtopic": subtopics[i], "cosine_similarity": cosine_similarities[i]}]

st.write(":blue[Bar Chart:]")
st.bar_chart(matches, x = "subtopic", y = "cosine_similarity", color = "subtopic", horizontal=True)