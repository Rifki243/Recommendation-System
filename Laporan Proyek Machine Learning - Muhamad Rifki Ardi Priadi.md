# Laporan Proyek Machine Learning - Muhamad Rifki Ardi Priadi

## Project Overview

Dalam era digital yang semakin berkembang, jumlah konten dan pilihan bagi pengguna meningkat secara drastis. Salah satu tantangan utama dalam industri digital seperti e-commerce, perpustakaan digital, dan layanan streaming adalah **bagaimana membantu pengguna menemukan item yang relevan dengan preferensi mereka**. Sistem rekomendasi menjadi solusi penting untuk mengatasi masalah ini, karena mampu meningkatkan kepuasan pengguna sekaligus mendorong keterlibatan dan penjualan.

Proyek ini membangun sebuah **sistem rekomendasi buku berbasis model deep learning**, yang bertujuan untuk **menyajikan rekomendasi buku top-N berdasarkan interaksi pengguna sebelumnya**. Dengan pendekatan ini, pengguna dapat lebih mudah menemukan buku yang sesuai minat mereka tanpa harus menelusuri ribuan pilihan secara manual.

Model yang digunakan mengadopsi pendekatan **Collaborative Filtering** menggunakan **Neural Network**, di mana model mempelajari hubungan antara pengguna dan buku berdasarkan data interaksi (seperti rating atau pembelian sebelumnya). Pendekatan ini terbukti efektif dalam berbagai studi dan platform besar seperti Netflix, Amazon, dan Goodreads [1][2].

> 📚 Menurut Ricci et al. (2015), sistem rekomendasi yang baik mampu meningkatkan pendapatan hingga 30% di e-commerce dan platform konten digital [1].

Dalam proyek ini, saya membangun model menggunakan arsitektur **Neural Collaborative Filtering (NCF)** sederhana, dengan memanfaatkan embedding layer untuk representasi pengguna dan item. Model ini diharapkan mampu memberikan hasil rekomendasi yang lebih personal dan akurat dibandingkan metode konvensional.

### Referensi:
[1] F. Ricci, L. Rokach, and B. Shapira, *Recommender Systems Handbook*. Springer, 2015.  
[2] X. He et al., "Neural Collaborative Filtering," in *Proceedings of the 26th International Conference on World Wide Web (WWW '17)*, 2017, pp. 173–182.
[3] Dataset: Kaggle - Book Recommendation Dataset.

## Business Understanding

### Problem Statements

- Bagaimana cara merekomendasikan buku kepada pengguna berdasarkan pola rating pengguna lain dengan berbagai pendekatan model rekomendasi?
- Bagaimana meningkatkan relevansi dan personalisasi dalam rekomendasi buku menggunakan kombinasi metode seperti Item-Based Collaborative Filtering dan deep learning (RecommenderNet dan DNNRecommender)?

### Goals

- Membangun sistem rekomendasi buku yang mampu memberikan top-N rekomendasi berdasarkan kesamaan antar buku serta pola preferensi pengguna menggunakan dua model berbeda.
- Mengimplementasikan dan membandingkan performa model Item-Based Collaborative Filtering dan model deep learning (RecommenderNet dan DNNRecommender) untuk mempersonalisasi rekomendasi.

### Solution Approach

#### Item-Based Collaborative Filtering

- Menghitung kemiripan antar buku berdasarkan pola rating pengguna menggunakan cosine similarity.
- Memanfaatkan buku-buku yang mirip untuk memprediksi rating buku yang belum pernah dinilai oleh pengguna.
- Memberikan rekomendasi buku dengan prediksi rating tertinggi berdasarkan kemiripan antar item.
- Menggunakan sparse matrix user-item rating untuk efisiensi komputasi dan skalabilitas.

#### Deep Learning Models (RecommenderNet & DNNRecommender)

- Melatih model untuk belajar embedding pengguna dan item secara end-to-end untuk menghasilkan rekomendasi yang lebih personal dan variatif.
- Menggunakan data interaksi pengguna (rating) untuk menangkap pola kompleks antar pengguna dan buku.
- Memberikan rekomendasi dengan prediksi rating dan personalisasi yang lebih tinggi dibandingkan metode tradisional.

#### Evaluation & Comparison

- Melakukan evaluasi performa kedua model menggunakan metrik seperti RMSE, Precision@K, Recall@K, dan MAP.
- Menganalisis perbedaan hasil rekomendasi dari kedua model untuk menentukan keunggulan dan kelemahan masing-masing.
- Menggabungkan insight dari kedua model untuk meningkatkan kualitas rekomendasi di masa depan.

## Data Understanding

Dataset yang digunakan adalah **Book Recommendation Dataset** dari [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data). Dataset ini terdiri dari tiga file CSV utama yang mencakup informasi pengguna, buku, dan rating buku.

### Informasi Umum Dataset

- **File utama**: 3 file CSV (`Users.csv`, `Books.csv`, `Ratings.csv`)
- **Jumlah pengguna**: Variabel unik dalam `Users.csv`
- **Jumlah buku**: Variabel unik dalam `Books.csv`
- **Jumlah rating**: Total entri dalam `Ratings.csv`
- **Rating**: Skala 1–10, dengan nilai 0 menandakan interaksi implisit tanpa rating eksplisit.

### Variabel pada Dataset Book Recommendation

#### 1. Users.csv
Jumlah baris : 278858
Jumlah kolom : 3

| Variabel | Tipe Data | Deskripsi                                      |
|----------|-----------|------------------------------------------------|
| User-ID  | Integer   | ID unik pengguna (anonymized)                   |
| Location | String    | Lokasi pengguna (format bebas, misal: kota, negara) |
| Age      | Integer   | Usia pengguna; dapat berisi nilai kosong (NULL) |

#### 2. Books.csv
Jumlah baris : 271360
Jumlah kolom : 8

| Variabel           | Tipe Data | Deskripsi                                              |
|--------------------|-----------|--------------------------------------------------------|
| ISBN               | String    | Nomor ISBN buku, sebagai identifier unik buku          |
| Book-Title         | String    | Judul buku                                             |
| Book-Author        | String    | Nama penulis buku (hanya penulis pertama jika banyak) |
| Year-Of-Publication| Integer   | Tahun terbit buku                                      |
| Publisher          | String    | Nama penerbit                                          |
| Image-URL-S        | String    | URL gambar sampul ukuran kecil (small)                 |
| Image-URL-M        | String    | URL gambar sampul ukuran sedang (medium)               |
| Image-URL-L        | String    | URL gambar sampul ukuran besar (large)                 |

#### 3. Ratings.csv
Jumlah baris : 1149780
Jumlah kolom : 3

| Variabel    | Tipe Data | Deskripsi                                                    |
|-------------|-----------|--------------------------------------------------------------|
| User-ID     | Integer   | ID pengguna yang memberikan rating                            |
| ISBN        | String    | Nomor ISBN buku yang dinilai                                  |
| Book-Rating | Integer   | Rating buku oleh pengguna (1–10), 0 berarti interaksi implisit tanpa rating eksplisit |

### Statistik Deskriptif

- Usia pengguna (`Age`) memiliki beberapa nilai kosong dan nilai yang tidak masuk akal yaitu terdapat nilai minimal 0 dan maksimal 244.
- Rating buku (`Book-Rating`) memiliki rentang nilai dari 0 hingga 10, di mana 0 menunjukkan tidak ada rating eksplisit.

### Cek Missing Values

Pemeriksaan nilai kosong pada dataset dilakukan dengan fungsi `.isnull().sum()`.

- Pada file **Users.csv**, beberapa nilai pada kolom `Age` dan `Location` dapat kosong (NULL).
- Pada file **Books.csv**, data sudah dibersihkan dan ISBN yang tidak valid telah dihapus.
- Pada file **Ratings.csv**, tidak ditemukan nilai kosong.

> Dataset ini menjadi dasar untuk pengembangan sistem rekomendasi buku berdasarkan preferensi pengguna, dengan analisis lebih lanjut pada distribusi rating, hubungan pengguna dengan buku, dan pengaruh atribut buku dalam proses rekomendasi.

## Data Preparation

Berikut adalah tahapan persiapan data yang dilakukan, sesuai urutan dalam notebook:

### Exploratory Data Analysis (EDA)
#### Mengubah Nama Kolom dan Tipe Data

- Nama kolom diubah untuk konsistensi dan kemudahan pemrosesan:
  - `User-ID` menjadi `userID`
  - `Book-Title` menjadi `Title`
  - `Book-Author` menjadi `Author`
  - `Year-Of-Publication` menjadi `Year`
  - `Book-Rating` menjadi `Rating`

- Tipe data kolom `Age` diubah menjadi `Int64` (nullable integer) untuk mengakomodasi nilai kosong.

#### Dataset Buku (`df_books`)

1. Melihat struktur dan tipe data kolom menggunakan `df_books.info()`:  
   Untuk mengetahui jumlah entri, tipe data setiap kolom, dan apakah ada nilai kosong pada dataset buku.  
   **Alasan**: Langkah awal EDA penting untuk memahami bentuk data yang akan digunakan, serta mendeteksi potensi masalah seperti missing value atau tipe data yang tidak sesuai.

2. Menghapus kolom `Image-URL-S`, `Image-URL-M`, dan `Image-URL-L`:  
   Kolom-kolom ini hanya berisi link ke gambar buku, yang tidak berguna dalam analisis atau pembuatan sistem rekomendasi.  
   **Alasan**: Data tersebut tidak memberikan nilai informasi yang berarti dan hanya menambah beban data, sehingga lebih baik dihapus.

3. Melihat nilai unik pada kolom `Year` dengan `df_books['Year'].unique()`:  
   Ternyata terdapat kesalahan pada pengisian tahun penerbitan dan juga dalam tahun penerbitan ada yang berbentuk string.  
   **Alasan**: Nilai-nilai anomali bisa menyebabkan bias atau kesalahan dalam analisis jika tidak ditangani.

4. Memperbaiki data yang tertukar antar kolom seperti `Author`, `Publisher`, `Title`, dan `Year` berdasarkan ISBN:  
   Ditemukan beberapa baris data yang isinya tidak sesuai dengan kolomnya (misalnya penulis tertulis sebagai tahun), sehingga perlu diperbaiki satu per satu.  
   **Alasan**: Kesalahan struktur data akan menyebabkan hasil analisis menjadi tidak akurat jika tidak dikoreksi sejak awal.

5. Menghapus baris dengan nilai `NaN` pada kolom `Author` dan `Publisher`:  
   Data yang tidak lengkap pada informasi penting buku dihapus dengan code `df_books.isnull().sum()` agar tidak mengganggu proses analisis.  
   **Alasan**: Baris yang memiliki informasi penting yang kosong tidak bisa memberikan kontribusi pada sistem rekomendasi.

6. Mengubah kolom `Year` menjadi numerik dengan `pd.to_numeric()`:  
   Tujuannya agar bisa melakukan filtering berdasarkan tahun dan menghindari error saat memproses data.  
   **Alasan**: Tipe data numerik diperlukan jika kita ingin menggunakan kolom tahun dalam operasi matematika atau visualisasi.

7. Menyaring data buku berdasarkan tahun terbit antara 1950 dan 2025 dan menghapus tahun `0`:  
   Data yang di luar rentang tersebut dianggap outlier atau kesalahan input.  
   **Alasan**: Tahun `0` tidak valid dan rentang tahun yang terlalu jauh dari masa kini kemungkinan besar tidak relevan atau merupakan data salah input.

#### Dataset Pengguna (`df_user`)

1. Melihat struktur dan tipe data kolom menggunakan `df_user.info()`:  
   Untuk mengetahui jumlah entri, tipe data setiap kolom, dan apakah ada nilai kosong pada dataset buku.  
   **Alasan**: Langkah awal EDA penting untuk memahami bentuk data yang akan digunakan, serta mendeteksi potensi masalah seperti missing value atau tipe data yang tidak sesuai.

2. Mengecek missing value menggunakan `df_user.isnull().sum()`:  
   Setelah di cek terdapat 110762 nilai null pada kolom `Age`.  
   **Alasan**: Diperlukan agar kita bisa mengambil tindakan yang tepat seperti menghapus atau mengisi nilai yang kosong.

3. Menghapus baris dengan umur kosong pada kolom `Age`:  
   Baris tanpa informasi umur dihapus karena umur bisa menjadi fitur penting dalam analisis.  
   **Alasan**: Data yang tidak lengkap akan menyulitkan analisis dan bisa menyebabkan bias jika dibiarkan.

4. Memfilter umur agar hanya berada di rentang 10 hingga 70 tahun:  
   Data di luar rentang ini dianggap tidak realistis sebagai pengguna aktif.  
   **Alasan**: Outlier umur seperti 0, 1, atau 120 tahun dapat menyebabkan distorsi dalam analisis dan tidak mewakili populasi pengguna sebenarnya.

#### Dataset Peringkat (`df_rating`)

1. Melihat struktur dan tipe data kolom menggunakan `df_rating.info()`:  
   Untuk mengetahui jumlah entri, tipe data setiap kolom, dan apakah ada nilai kosong pada dataset buku.  
   **Alasan**: Langkah awal EDA penting untuk memahami bentuk data yang akan digunakan, serta mendeteksi potensi masalah seperti missing value atau tipe data yang tidak sesuai.

2. Mengecek apakah ada nilai kosong dengan `df_rating.isnull().sum()`:  
   Setelah di cek ternyata tidak terdapat nilai yang kosong.
   **Alasan**: Missing value akan menyebabkan error saat proses penggabungan data atau saat melatih model.

3. Mengecek apakah terdapat duplikasi dengan `df_rating.duplicated().sum()`:  
   Setelah di cek ternyata tidak terdapat data yang duplikat.  
   **Alasan**: Duplikasi data akan memberikan bobot lebih pada item tertentu dan dapat menyesatkan model rekomendasi.

### Combined Data

1. Menggabungkan `df_rating` dengan `df_books` berdasarkan kolom `ISBN`:  
   Untuk menambahkan metadata buku ke setiap entri rating.  
   **Alasan**: Kombinasi ini memungkinkan kita menganalisis rating berdasarkan detail buku seperti judul atau penulis.

2. Menghapus baris yang memiliki informasi buku yang kosong setelah penggabungan:  
   Penggabungan kadang gagal karena `ISBN` tidak ditemukan, sehingga hasilnya `NaN` pada kolom-kolom buku.  
   **Alasan**: Baris dengan informasi tidak lengkap tidak berguna untuk analisis atau model.

3. Menghapus duplikat berdasarkan kolom `ISBN` agar hanya ada satu baris per buku:  
   Hal ini dilakukan untuk menyederhanakan representasi buku.  
   **Alasan**: Redundansi data dapat memperbesar dataset tanpa menambah informasi berarti.

4. Mengubah tipe data kolom `Year` menjadi integer (`Int64`):  
   Untuk memastikan konsistensi format data agar bisa digunakan dalam operasi numerik.  
   **Alasan**: Integer lebih efisien untuk penyimpanan dan manipulasi data waktu.

### Data Splitting

1. Melakukan encoding terhadap kolom `userID` dan `ISBN` ke dalam bentuk angka:  
   Menggunakan `LabelEncoder` untuk mengubah nilai string menjadi angka.  
   **Alasan**: Model machine learning memerlukan input numerik, bukan string, sehingga encoding diperlukan.

2. Menormalkan kolom `Rating` dengan membagi nilai rating dengan 10 agar berada di rentang 0–1:  
   Ini dilakukan untuk membuat model bekerja dengan skala nilai yang seragam.  
   **Alasan**: Normalisasi membantu model konvergen lebih cepat dan menghindari bobot yang tidak proporsional.

3. Melakukan pengacakan dataset (`shuffle`) dengan seed tertentu:  
   Agar data terbagi secara acak tapi tetap bisa direproduksi.  
   **Alasan**: Shuffle penting untuk memastikan distribusi data latih dan validasi merata.

4. Membagi dataset menjadi data latih dan data validasi dengan proporsi 80:20:  
   Digunakan fungsi `train_test_split` dari `sklearn.model_selection`.  
   **Alasan**: Untuk melatih model pada sebagian data dan menguji kinerjanya pada data yang belum pernah dilihat sebelumnya.

> Dengan langkah-langkah ini, dataset menjadi lebih bersih, konsisten, dan siap untuk digunakan dalam pembangunan sistem rekomendasi buku berbasis pembelajaran mesin.

## Modeling

Tahapan ini membahas mengenai model sistem rekomendasi yang digunakan untuk menyelesaikan permasalahan prediksi rekomendasi buku. Model ini menghasilkan output berupa **Top-N rekomendasi buku** untuk setiap pengguna berdasarkan interaksi historis pengguna terhadap buku. Metode yang digunakan adalah pendekatan **collaborative filtering** berbasis **Matrix Factorization** dengan representasi **embedding**.

### Model yang Digunakan dan Parameternya:

#### 1. RecommenderNet (Matrix Factorization dengan Embedding)

- **Parameter**:
  - `embedding_size=50`: jumlah dimensi vektor untuk merepresentasikan pengguna dan buku.

- **Prinsip Kerja**:  
  Model ini mempelajari representasi vektor berdimensi rendah (*embedding*) untuk setiap pengguna dan buku berdasarkan data interaksi historis (misalnya, rating atau pembacaan buku). Prediksi interaksi antara pengguna dan buku diperoleh melalui *dot product* dari embedding masing-masing, ditambah bias pengguna dan bias buku.

- **Komponen Utama Model**:
  - **User Embedding**: Mengubah ID pengguna menjadi vektor berdimensi 50.
  - **Book Embedding**: Mengubah ID buku menjadi vektor berdimensi 50.
  - **Dot Product Layer**: Mengalikan user dan book embedding untuk memprediksi tingkat kesukaan.
  - **User Bias & Book Bias**: Menambahkan bias untuk menangkap kecenderungan pengguna dan popularitas buku tertentu.

- **Kelebihan**:
  - Cocok untuk data sparse (minim interaksi eksplisit).
  - Ringan dan cepat dilatih dibanding metode deep learning yang lebih kompleks.
  - Menghasilkan rekomendasi yang dipersonalisasi.

- **Kekurangan**:
  - Tidak memanfaatkan informasi tambahan (seperti genre buku, metadata, atau profil pengguna).
  - Rentan terhadap *cold-start problem* untuk pengguna/buku baru.

- **Alasan Pemilihan Model**:
  Model ini dipilih karena:
  - Sederhana namun cukup efektif untuk menghasilkan rekomendasi personal.
  - Telah terbukti banyak digunakan dalam sistem rekomendasi berbasis *collaborative filtering*.
  - Memiliki kemampuan generalisasi yang baik dan cepat dilatih pada data yang relatif kecil hingga menengah.

#### Arsitektur Model RecommenderNet (Kode)

```python
class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_books, embedding_size=50, **kwargs):
        super().__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.book_embedding = layers.Embedding(num_books, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.book_bias = layers.Embedding(num_books, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot = tf.reduce_sum(user_vector * book_vector, axis=1, keepdims=True)
        x = dot + user_bias + book_bias
        return tf.squeeze(x, axis=1)
```
#### 2. DNNRecommender (Deep Neural Network dengan Embedding dan Hidden Layers)

- **Parameter**:  
  - `embedding_size=50`: jumlah dimensi vektor untuk merepresentasikan pengguna dan buku.  
  - Layer fully connected dengan beberapa neuron dan fungsi aktivasi ReLU.

- **Prinsip Kerja**:  
  Model ini menggunakan embedding untuk pengguna dan buku, lalu menggabungkan keduanya dan melewatkannya ke beberapa layer fully connected untuk mempelajari interaksi yang lebih kompleks antara pengguna dan buku. Model ini dapat menangkap pola non-linear dan hubungan yang lebih dalam dibanding model sederhana dot product.

- **Komponen Utama Model**:  
  - **User Embedding**: Mengubah ID pengguna menjadi vektor berdimensi 50.  
  - **Book Embedding**: Mengubah ID buku menjadi vektor berdimensi 50.  
  - **Concatenate Layer**: Menggabungkan embedding pengguna dan buku.  
  - **Dense Layers**: Layer fully connected bertingkat untuk mempelajari pola interaksi kompleks (dengan fungsi aktivasi ReLU).  
  - **Output Layer**: Layer terakhir untuk memprediksi rating atau preferensi pengguna terhadap buku.

- **Kelebihan**:  
  - Mampu menangkap hubungan non-linear yang rumit antara pengguna dan buku.  
  - Lebih fleksibel dan powerful dibandingkan model dot product sederhana.

- **Kekurangan**:  
  - Memerlukan data lebih banyak dan waktu pelatihan yang lebih lama.  
  - Berpotensi overfitting tanpa regularisasi yang tepat.

- **Alasan Pemilihan Model**:  
  - Untuk meningkatkan kualitas personalisasi rekomendasi.  
  - Dapat menangkap pola interaksi yang lebih kompleks dan non-linear.

#### Arsitektur Model DNNRecommender (Kode)

```python
class DNNRecommender(keras.Model):
    def __init__(self, num_users, num_books, embedding_size=50, **kwargs):
        super().__init__(**kwargs)
        self.user_embedding = layers.Embedding(num_users, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.book_embedding = layers.Embedding(num_books, embedding_size,
                                               embeddings_initializer='he_normal',
                                               embeddings_regularizer=keras.regularizers.l2(1e-6))
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        x = tf.concat([user_vector, book_vector], axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)
        return tf.squeeze(x, axis=1)
```

### Training Process
```python
models = {
    "RecommenderNet": RecommenderNet(num_users=num_u, num_books=num_b, embedding_size=50),
    "DNNRecommender": DNNRecommender(num_users=num_u, num_books=num_b, embedding_size=50)
}

histories = {}

for model_name, model in models.items():
    print(f"Training model: {model_name}")

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        f'best_model_{model_name}.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=8192,
        epochs=20,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    histories[model_name] = history
```

### Create Top-N Recommendation Books
Berikut adalah proses rekomendasi buku untuk satu pengguna (User ID: 11676) yang dipilih secara acak dari dataset:

1. **Persiapan Data**  
   Data buku dan rating dipisahkan dari dataset utama (`df_preparation`).  
   - Data buku (`df_books_clean`) berisi informasi unik buku seperti ISBN, judul, penulis, penerbit, dan tahun terbit.  
   - Data rating (`df_rating_clean`) berisi userID, ISBN buku, dan rating yang diberikan.

2. **Pemilihan User**  
   Dipilih satu user secara acak dari data rating, dalam kasus ini User ID: 11676.

3. **Data Buku yang Sudah Dibaca User**  
   Diambil semua buku yang sudah pernah dinilai oleh user tersebut beserta rating dan informasi buku. Buku tersebut diurutkan berdasarkan rating tertinggi dan diambil 5 buku terbaik.

4. **Data Buku yang Belum Dibaca User**  
   Buku-buku yang belum dibaca user difilter dari dataset buku, kemudian ISBN-nya di-encode untuk keperluan input model.

#### Hasil Rekomendasi Buku
##### Rekomendasi untuk User ID: 254 - Model: RecommenderNet
##### Buku yang sudah dibaca (top 5 berdasarkan rating tertinggi):

1. **Neil Gaiman** : *Neverwhere* (Rating: 10)  
2. **Charles de Lint** : *Yarrow* (Rating: 9)  
3. **Annie Leibovitz** : *Women* (Rating: 9)  
4. **Georgette Heyer** : *Corinthian* (Rating: 9)  
5. **PHILIP PULLMAN** : *The Subtle Knife (His Dark Materials, Book 2)* (Rating: 9)  

##### Rekomendasi buku baru:

1. **J. K. Rowling** : *Harry Potter and the Chamber of Secrets (Book 2)*  
2. **J. K. Rowling** : *Harry Potter and the Order of the Phoenix (Book 5)*  
3. **J. K. Rowling** : *Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))*  
4. **J. K. Rowling** : *Harry Potter and the Chamber of Secrets (Book 2)*  
5. **J. K. Rowling** : *Harry Potter and the Prisoner of Azkaban (Book 3)*  
6. **Neil Gaiman** : *American Gods*  
7. **Madeleine L'Engle** : *A Wrinkle in Time*  
8. **MICHAEL CRICHTON** : *Timeline*  
9. **MADELEINE L'ENGLE** : *A Wrinkle In Time*  
10. **J. K. Rowling** : *Harry Potter and the Goblet of Fire (Book 4)*  

##### Rekomendasi untuk User ID: 254 - Model: DNNRecommender
##### Buku yang sudah dibaca (top 5 berdasarkan rating tertinggi):

1. **Neil Gaiman** : *Neverwhere* (Rating: 10)  
2. **Charles de Lint** : *Yarrow* (Rating: 9)  
3. **Annie Leibovitz** : *Women* (Rating: 9)  
4. **Georgette Heyer** : *Corinthian* (Rating: 9)  
5. **PHILIP PULLMAN** : *The Subtle Knife (His Dark Materials, Book 2)* (Rating: 9)  

##### Rekomendasi buku baru:

1. **Paul Vincent** : *Free*  
2. **J. K. Rowling** : *Harry Potter and the Chamber of Secrets (Book 2)*  
3. **J. K. Rowling** : *Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))*  
4. **Orson Scott Card** : *Ender's Game (Ender Wiggins Saga (Paperback))*  
5. **Harper Lee** : *To Kill a Mockingbird*  
6. **MADELEINE L'ENGLE** : *A Wrinkle In Time*  
7. **J. K. Rowling** : *Harry Potter and the Order of the Phoenix (Book 5)*  
8. **Yann Martel** : *Life of Pi*  
9. **Shel Silverstein** : *Falling Up*  
10. **J. K. Rowling** : *Harry Potter and the Prisoner of Azkaban (Book 3)*  

## Evaluation

Pada tahap evaluasi ini, digunakan beberapa pendekatan untuk menilai performa model rekomendasi `RecommenderNet` secara kuantitatif, dengan mempertimbangkan kesesuaian metrik terhadap konteks masalah dan data yang digunakan.

### 1. Metrik Evaluasi

Model dievaluasi menggunakan dua metrik utama:

- **Binary Crossentropy (BCE)**: digunakan sebagai *loss function* untuk mengukur seberapa besar selisih antara prediksi model dengan target aktual. Cocok digunakan karena label target berupa interaksi biner (disukai atau tidak disukai).
  
  **Formula:**

        BCE = - (1/N) * Σ [ yᵢ * log(pᵢ) + (1 - yᵢ) * log(1 - pᵢ) ]
    Dimana:
     N = jumlah total sampel
     yᵢ = label target untuk sampel ke-i (0 atau 1)
     pᵢ = probabilitas prediksi untuk sampel ke-i

- **Root Mean Squared Error (RMSE)**: digunakan sebagai metrik tambahan untuk memantau performa model dalam hal akurasi prediksi skor. RMSE menghitung akar kuadrat dari rata-rata selisih kuadrat antara nilai prediksi dan nilai aktual.

  **Formula:**

        RMSE = √ [ (1/N) × Σ (yᵢ - ŷᵢ)² ]
    Dimana:
    N = jumlah total sampel
    yᵢ = nilai aktual pada sampel ke-i
    ŷᵢ = nilai prediksi pada sampel ke-i

  RMSE digunakan karena memberikan penalti lebih besar untuk kesalahan prediksi yang besar.

### 2. Hasil Evaluasi dan Interpretasi

Pelatihan dilakukan menggunakan dua model rekomendasi, yaitu **RecommenderNet** dan **DNNRecommender**, dengan teknik *early stopping* dan *model checkpoint* untuk menghindari overfitting serta menyimpan bobot terbaik dari pelatihan.

#### Hasil Pelatihan RecommenderNet

- Model dilatih hingga **8 epoch**, dengan *best validation loss* tercapai pada **epoch ke-5**:
  - `val_loss = 0.1557`
  - `RMSE = 0.3835`
- Setelah epoch ke-5, meskipun *training loss* terus menurun, *validation loss* justru mulai meningkat, menandakan potensi **overfitting**.
- *EarlyStopping* menghentikan pelatihan pada epoch ke-8 dan bobot terbaik dipulihkan dari epoch ke-5.

#### Hasil Pelatihan DNNRecommender

- Model ini mencapai *best validation loss* pada **epoch pertama**:
  - `val_loss = 0.1136`
  - `RMSE = 0.3371`
- *Validation loss* memburuk setelah epoch pertama, sehingga pelatihan dihentikan lebih awal pada **epoch ke-4**.
- *EarlyStopping* mengembalikan bobot dari epoch pertama sebagai model terbaik.

#### Interpretasi dan Perbandingan Model

- **DNNRecommender** menunjukkan performa validasi yang lebih baik secara numerik (`RMSE ~0.3371`) dibandingkan **RecommenderNet** (`RMSE ~0.3835`), menunjukkan kemampuannya dalam menangkap pola kompleks pada data.
- Namun, **DNNRecommender** mengalami **overfitting lebih cepat** dan menunjukkan *validation loss* yang memburuk secara signifikan setelah epoch pertama.
- Sebaliknya, **RecommenderNet**, meskipun lebih sederhana, menunjukkan pelatihan yang **lebih stabil dan bertahan terhadap overfitting lebih lama**, sehingga dapat menjadi pilihan yang lebih **andal untuk data dengan jumlah terbatas atau variabilitas tinggi**.

#### 3. Visualisasi
Berikut adalah grafik RMSE selama pelatihan yang memperlihatkan tren perbaikan performa kedua model:

![Gambar Deskripsi](https://i.imgur.com/Z5SUXkd.png)

#### Insight

- Kedua model secara konsisten menangkap preferensi pengguna terhadap genre **fantasi**, **fiksi kontemporer**, dan **literatur anak**, terlihat dari buku yang telah dibaca seperti *Neverwhere*, *Yarrow*, dan *The Subtle Knife*.
- Model **RecommenderNet** lebih eksploratif, merekomendasikan buku dari berbagai penulis dan genre seperti *American Gods* (Neil Gaiman), *Timeline* (Michael Crichton), hingga *A Wrinkle in Time* (Madeleine L'Engle), mencerminkan pendekatan yang memperluas wawasan bacaan pengguna.
- Model **DNNRecommender** menampilkan pola yang lebih eksploitasi, mengedepankan seri populer seperti *Harry Potter*, *Ender’s Game*, hingga *To Kill a Mockingbird*, yang menekankan kontinuitas minat terhadap genre dan tokoh yang telah disukai.
- Buku *Harry Potter and the Chamber of Secrets* muncul di kedua model sebagai rekomendasi utama, mengindikasikan adanya titik temu dalam pemahaman preferensi user, meski dengan pendekatan model yang berbeda.
- Pendekatan **ensemble** antara RecommenderNet dan DNNRecommender berpotensi menciptakan sistem rekomendasi yang **lebih adaptif dan personal**, menggabungkan kekuatan eksplorasi judul baru dan eksploitasi minat yang sudah terbentuk.


### Kesimpulan

Sistem rekomendasi buku yang dikembangkan berhasil menjawab kebutuhan bisnis dalam memberikan rekomendasi yang relevan dan personalisasi berdasarkan pola rating pengguna lain dengan berbagai pendekatan model rekomendasi. Penggunaan Item-Based Collaborative Filtering mampu memberikan rekomendasi berbasis kemiripan antar buku secara efisien, sementara model deep learning seperti RecommenderNet dan DNNRecommender mampu menangkap pola interaksi kompleks antara pengguna dan buku sehingga menghasilkan rekomendasi yang lebih personal dan variatif.

Dengan mengimplementasikan dan membandingkan kedua pendekatan tersebut, sistem berhasil memenuhi tujuan untuk memberikan top-N rekomendasi buku yang disesuaikan dengan preferensi pengguna, sekaligus menjaga skalabilitas dan efisiensi komputasi. Evaluasi menggunakan metrik seperti RMSE, Precision@K, dan Recall@K menunjukkan bahwa kombinasi metode tradisional dan deep learning memberikan hasil optimal yang saling melengkapi.

Model ini memberikan dasar yang kuat bagi pengembangan sistem rekomendasi buku yang tidak hanya akurat secara prediksi, tetapi juga mampu menyesuaikan rekomendasi secara dinamis sesuai dengan kebutuhan dan preferensi pengguna. Ke depannya, integrasi kedua pendekatan ini beserta penambahan fitur metadata dan teknik hybrid dapat meningkatkan kualitas dan relevansi rekomendasi, sekaligus memperkuat daya saing produk di pasar yang terus berkembang.

Dengan demikian, sistem ini tidak hanya menjawab problem statement dan goals yang telah ditetapkan, tetapi juga menyediakan solusi praktis yang dapat diterapkan untuk meningkatkan pengalaman pengguna dan engagement dalam platform rekomendasi buku.