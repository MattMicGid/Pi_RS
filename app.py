import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load data
place = pd.read_csv('clean_depok.csv')
ur = pd.read_csv('user_ratings.csv')

# Encoding User_ID dan Place_ID ke bentuk numerik
user_ids = ur['user_id'].unique().tolist()
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

place_ids = ur['int_place_id'].unique().tolist()
place_to_place_encoded = {x: i for i, x in enumerate(place_ids)}
place_encoded_to_place = {i: x for i, x in enumerate(place_ids)}

# Menambahkan kolom encoded ke dataframe user ratings
ur['user'] = ur['user_id'].map(user_to_user_encoded)
ur['place'] = ur['int_place_id'].map(place_to_place_encoded)
ur['rating'] = ur['rating'].values.astype(np.float32)

# Mendapatkan jumlah user dan tempat
num_users = len(user_to_user_encoded)
num_place = len(place_encoded_to_place)

# Mendapatkan nilai minimum dan maksimum rating
min_rating = min(ur['rating'])
max_rating = max(ur['rating'])

# Definisi Model Rekomendasi
class Recommenders(tf.keras.Model):
    def __init__(self, num_users, num_place, embedding_size, **kwargs):
        super(Recommenders, self).__init__(**kwargs)
        self.user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.place_embedding = tf.keras.layers.Embedding(
            num_place, embedding_size,
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

# Inisialisasi dan Load Model
model = Recommenders(num_users, num_place, 50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
model.load_weights('recommender.weights.h5')

# Streamlit App
st.title('Rekomendasi Tempat Wisata di Depok')

# Display a scrollable box with a list of places
place_names = sorted(place['name'].tolist())
selected_place = st.selectbox("Daftar Tempat Wisata:", place_names)

# Display the details of the selected place
if selected_place:
    place_details = place[place['name'] == selected_place].iloc[0]
    st.write(f"**Nama Tempat**: {place_details['name']}")
    st.write(f"**Alamat**: {place_details['formatted_address']}")
    st.write(f"**Rating**: {place_details['rating']} | **Jumlah Pemberi Rating**: {place_details['user_ratings_total']}")
    st.write(f"[Link Google Maps](https://www.google.com/maps/place/?q=place_id:{place_details['place_id']})")

# Input Tempat yang Disukai
st.header('Rekomendasi Berdasarkan Tempat yang Disukai')
liked_place_names = st.text_input("Masukkan nama tempat yang Anda sukai (pisahkan dengan koma):")

# Modifikasi Rekomendasi Berdasarkan Tempat yang Disukai menggunakan beberapa tempat
if st.button('Tampilkan Rekomendasi Berdasarkan Tempat yang Disukai'):
    liked_place_list = [name.strip() for name in liked_place_names.split(',')]  # Ambil tempat yang diinput
    
    liked_place_embeddings = []
    
    # Ambil embedding untuk setiap tempat yang diinput
    for liked_place_name in liked_place_list:
        liked_place_row = place[place['name'].str.contains(liked_place_name, case=False, na=False)]
        if liked_place_row.empty:
            st.write(f"Tempat dengan nama '{liked_place_name}' tidak ditemukan.")
        else:
            liked_place_id = liked_place_row['int_place_id'].values[0]
            liked_place_encoded = place_to_place_encoded[liked_place_id]
            liked_place_embedding = model.place_embedding(np.array([liked_place_encoded]))
            liked_place_embeddings.append(liked_place_embedding)
    
    if liked_place_embeddings:
        # Hitung rata-rata embedding dari tempat yang diinput
        avg_embedding = np.mean([embed.numpy() for embed in liked_place_embeddings], axis=0)
        
        # Hitung kemiripan (dot product) antara embedding rata-rata dan semua tempat lainnya
        all_place_embeddings = model.place_embedding.embeddings.numpy()
        similarity_scores = np.dot(all_place_embeddings, avg_embedding.T).flatten()
        
        # Ambil tempat dengan skor tertinggi selain tempat yang disukai
        similar_place_indices = similarity_scores.argsort()[-6:][::-1]  # Top 5
        similar_place_ids = [place_encoded_to_place[i] for i in similar_place_indices]
        recommended_places = place[place['int_place_id'].isin(similar_place_ids)]
        
        if not recommended_places.empty:
            st.write(f"Top 5 tempat rekomendasi berdasarkan tempat yang Anda sukai: {', '.join(liked_place_list)}")
            for row in recommended_places.itertuples():
                st.write(f"**Nama Tempat**: {row.name}")
                st.write(f"**Alamat**: {row.formatted_address}")
                st.write(f"**Rating**: {row.rating} | **Jumlah Pemberi Rating**: {row.user_ratings_total}")
                st.write(f"**Category**: {row.category}")
                st.write(f"[Link Google Maps](https://www.google.com/maps/place/?q=place_id:{row.place_id})")
                st.write("---")
        else:
            st.write("Tidak ada tempat rekomendasi yang tersedia.")

# Rekomendasi Tempat untuk Pengguna Baru
st.header('Rekomendasi Tempat untuk Pengguna Baru')
if st.button('Tampilkan Saran Tempat'):
    top_places = place.sort_values(by='user_ratings_total', ascending=False).head(5)
    
    st.write("Top 5 tempat rekomendasi untuk pengguna baru berdasarkan jumlah pemberi rating:")
    for row in top_places.itertuples():
        st.write(f"**Nama Tempat**: {row.name}")
        st.write(f"**Alamat**: {row.formatted_address}")
        st.write(f"**Rating**: {row.rating} | **Jumlah Pemberi Rating**: {row.user_ratings_total}")
        st.write(f"**Category**: {row.category}")
        st.write(f"[Link Google Maps](https://www.google.com/maps/place/?q=place_id:{row.place_id})")
        st.write("---")
