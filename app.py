import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load data
place = pd.read_csv('clean_depok.csv')
ur = pd.read_csv('user_ratings.csv')

# Encoding User_Id dan Place_Id
user_ids = ur['user_id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

place_ids = ur['int_place_id'].unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

ur['user'] = ur['user_id'].map(user_to_user_encoded)
ur['place'] = ur['int_place_id'].map(place_to_place_encoded)
ur['rating'] = ur['rating'].values.astype(np.float32)

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)

# Mendapatkan jumlah place
num_place = len(place_encoded_to_place)

# Mengubah rating menjadi nilai float
ur['rating'] = ur['rating'].values.astype(np.float32)

# Nilai minimum rating
min_rating = min(ur['rating'])

# Nilai maksimal rating
max_rating = max(ur['rating'])

# Load the model
class Recommenders(tf.keras.Model):
  def __init__(self, num_users, num_place, embedding_size, **kwargs):
    super(Recommenders, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_place = num_place
    self.embedding_size = embedding_size
    self.user_embedding = tf.keras.layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )
    self.user_bias = tf.keras.layers.Embedding(num_users, 1)
    self.place_embedding = tf.keras.layers.Embedding(
        num_place,
        embedding_size,
        embeddings_initializer='he_normal',
        embeddings_regularizer=keras.regularizers.l2(1e-6)
    )
    self.place_bias = tf.keras.layers.Embedding(num_place, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    place_vector = self.place_embedding(inputs[:, 1])
    place_bias = self.place_bias(inputs[:, 1])
    dot_user_place = tf.tensordot(user_vector, place_vector, 2)
    x = dot_user_place + user_bias + place_bias
    return tf.nn.sigmoid(x)

model = Recommenders(num_users, num_place, 50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

model.load_weights('recommender.weights.h5')

# Streamlit app title
st.title('Rekomendasi Tempat Wisata di Depok')

# Input user_id
st.header('Rekomendasi Berdasarkan User ID')
user_id_input = st.number_input("Masukkan User ID:", min_value=0, max_value=len(user_ids)-1, step=1)

if st.button('Tampilkan Rekomendasi Berdasarkan User ID'):
    user_id = user_id_input
    place_have_not_visit = ur[ur.user_id == user_id]
    place_have_not_visit = place[place['int_place_id'].isin(place_have_not_visit.int_place_id.values)]['int_place_id']
    place_have_not_visit = list(set(place_have_not_visit).intersection(set(place_to_place_encoded.keys())))

    if len(place_have_not_visit) > 0:
        place_have_not_visit = [[place_to_place_encoded.get(x)] for x in place_have_not_visit]
        user_encoder = user_to_user_encoded.get(user_id)
        user_place_array = np.hstack(([[user_encoder]] * len(place_have_not_visit), place_have_not_visit))
        ratings = model.predict(user_place_array).flatten()

        top_ratings_indices = ratings.argsort()[-5:][::-1]
        recommended_place_ids = [place_encoded_to_place.get(place_have_not_visit[x][0]) for x in top_ratings_indices]
        recommended_place = place[place['int_place_id'].isin(recommended_place_ids)]

        st.write('Top 5 Tempat Rekomendasi:')
        for row in recommended_place.itertuples():
            st.write(f"**Nama Tempat**: {row.name}")
            st.write(f"**Alamat**: {row.formatted_address}")
            st.write(f"**Rating**: {row.rating} | **Jumlah Pemberi Rating**: {row.user_ratings_total}")
            st.write(f"[Link Google Maps](https://www.google.com/maps/place/?q=place_id:{row.place_id})")
            st.write("---")
    else:
        st.write("Tidak ada rekomendasi tempat yang tersedia.")

# Input kategori
st.header('Rekomendasi Berdasarkan Kategori')
categories = place['category'].unique().tolist()
desired_category = st.selectbox("Pilih Kategori:", categories)

if st.button('Tampilkan Rekomendasi Berdasarkan Kategori'):
    filtered_places = place[place['category'] == desired_category]
    sorted_filtered_places = filtered_places.sort_values(by=['user_ratings_total', 'rating'], ascending=[False, False])
    top_5_places = sorted_filtered_places.head(5)

    st.write(f"Top 5 Tempat Rekomendasi di Kategori: {desired_category}")
    for row in top_5_places.itertuples():
        st.write(f"**Nama Tempat**: {row.name}")
        st.write(f"**Alamat**: {row.formatted_address}")
        st.write(f"**Rating**: {row.rating} | **Jumlah Pemberi Rating**: {row.user_ratings_total}")
        st.write(f"[Link Google Maps](https://www.google.com/maps/place/?q=place_id:{row.place_id})")
        st.write("---")

# Input tempat yang disukai
st.header('Rekomendasi Berdasarkan Tempat yang Disukai')
liked_place_name = st.text_input("Masukkan nama tempat yang Anda sukai:")

if st.button('Tampilkan Rekomendasi Berdasarkan Tempat yang Disukai'):
    liked_place_row = place[place['name'].str.contains(liked_place_name, case=False, na=False)]
    if liked_place_row.empty:
        st.write(f"Tempat dengan nama '{liked_place_name}' tidak ditemukan.")
    else:
        liked_place_id = liked_place_row['int_place_id'].values[0]
        similar_users = ur[(ur['int_place_id'] == liked_place_id) & (ur['rating'] >= 4.0)]['user_id'].unique()

        if len(similar_users) == 0:
            st.write("Tidak ada pengguna lain yang memberikan rating tinggi untuk tempat ini.")
        else:
            similar_places = ur[(ur['user_id'].isin(similar_users)) & (ur['int_place_id'] != liked_place_id)]
            if similar_places.empty:
                st.write("Tidak ada rekomendasi tempat lain yang mirip.")
            else:
                recommended_place_ids = similar_places['int_place_id'].unique()
                place_have_not_visit = [[place_to_place_encoded.get(x)] for x in recommended_place_ids]
                user_encoder = user_to_user_encoded.get(user_id_input)
                user_place_array = np.hstack(([[user_encoder]] * len(place_have_not_visit), place_have_not_visit))
                ratings = model.predict(user_place_array).flatten()
                top_ratings_indices = ratings.argsort()[-5:][::-1]
                final_recommendations = [place_encoded_to_place.get(place_have_not_visit[x][0]) for x in top_ratings_indices]
                final_recommendations = [x for x in final_recommendations if x != liked_place_id]

                if len(final_recommendations) > 0:
                    recommended_place = place[place['int_place_id'].isin(final_recommendations)]
                    st.write('Top 5 tempat rekomendasi berdasarkan tempat yang disukai:')
                    for row in recommended_place.itertuples():
                        st.write(f"**Nama Tempat**: {row.name}")
                        st.write(f"**Alamat**: {row.formatted_address}")
                        st.write(f"**Rating**: {row.rating} | **Jumlah Pemberi Rating**: {row.user_ratings_total}")
                        st.write(f"[Link Google Maps](https://www.google.com/maps/place/?q=place_id:{row.place_id})")
                        st.write("---")
                else:
                    st.write("Tidak ada tempat rekomendasi yang tersedia.")
