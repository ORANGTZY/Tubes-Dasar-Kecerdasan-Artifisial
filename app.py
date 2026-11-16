import streamlit as st  # Impor Streamlit untuk membuat tampilan web interaktif tanpa perlu HTML/CSS
import pandas as pd  # Impor pandas untuk membaca dan mengolah data tabel
import numpy as np  # Impor numpy untuk operasi matematika (dibutuhkan oleh library lain)
import os  # Impor os untuk mengakses lokasi file di komputer
from sklearn.ensemble import RandomForestRegressor  # Impor model Machine Learning: Random Forest (kumpulan banyak pohon keputusan)
from sklearn.neural_network import MLPRegressor  # Impor model Deep Learning: Jaringan Saraf Tiruan (Neural Network)
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Impor fungsi untuk mengukur seberapa meleset hasil prediksi

# ==========================================
# 1. FUNGSI LOGIKA FUZZY (FROM SCRATCH)
# ==========================================
def trapmf(x, a, b, c, d):  # Fungsi bentuk trapesium: mengukur seberapa besar nilai x termasuk dalam suatu kategori
    if x <= a or x >= d: return 0.0  # Kalau x di luar batas kiri (a) atau kanan (d), berarti tidak masuk kategori ini sama sekali (nilai 0)
    elif a < x <= b: return (x - a) / (b - a)  # Sisi kiri naik: semakin x mendekati b, semakin besar nilainya (mulai masuk kategori)
    elif b < x <= c: return 1.0  # Di antara b dan c, nilai keanggotaan penuh (= 1, artinya benar-benar masuk kategori ini)
    elif c < x < d: return (d - x) / (d - c)  # Sisi kanan turun: semakin x mendekati d, semakin kecil nilainya (mulai keluar kategori)
    return 0.0  # Jaga-jaga jika tidak ada kondisi yang cocok, kembalikan 0

def trimf(x, a, b, c):  # Fungsi bentuk segitiga: sama seperti trapesium tapi puncaknya hanya setitik (di b)
    return trapmf(x, a, b, b, c)  # Cukup panggil trapmf dengan b diulang dua kali sehingga tidak ada area datar di atas

def fuzzify_input(val, param):  # Fungsi untuk mengubah nilai angka polutan menjadi derajat keanggotaan (Rendah/Sedang/Tinggi)
    if param == 'PM2.5': return {'Rendah': trapmf(val, 0, 0, 30, 60), 'Sedang': trimf(val, 30, 60, 90), 'Tinggi': trapmf(val, 60, 90, 500, 500)}  # Batas kategori PM2.5: Rendah s/d 60, Sedang 30-90, Tinggi mulai 60
    elif param == 'PM10': return {'Rendah': trapmf(val, 0, 0, 50, 100), 'Sedang': trimf(val, 50, 100, 150), 'Tinggi': trapmf(val, 100, 150, 500, 500)}  # Batas kategori PM10 lebih longgar karena ukuran partikelnya lebih besar
    elif param == 'NO2': return {'Rendah': trapmf(val, 0, 0, 40, 80), 'Sedang': trimf(val, 40, 80, 120), 'Tinggi': trapmf(val, 80, 120, 300, 300)}  # Batas kategori gas NO2
    elif param == 'CO': return {'Rendah': trapmf(val, 0, 0, 1, 2), 'Sedang': trimf(val, 1, 2, 3), 'Tinggi': trapmf(val, 2, 3, 50, 50)}  # CO diukur dalam satuan sangat kecil (mg/m³), makanya batasnya jauh lebih kecil dari polutan lain
    elif param == 'SO2': return {'Rendah': trapmf(val, 0, 0, 40, 80), 'Sedang': trimf(val, 40, 80, 120), 'Tinggi': trapmf(val, 80, 120, 300, 300)}  # Batas kategori SO2 sama seperti NO2

def apply_rules(f_pm25, f_pm10, f_no2, f_co, f_so2):  # Fungsi untuk menerapkan 15 aturan fuzzy berdasarkan nilai keanggotaan semua polutan
    rules = []  # Buat daftar kosong untuk menampung hasil tiap aturan
    # 15 Aturan Fuzzy
    rules.append(('Baik',   min(f_pm25['Rendah'], f_pm10['Rendah'], f_no2['Rendah'], f_co['Rendah'], f_so2['Rendah'])))  # R1: Semua polutan Rendah → udara Baik (ambil nilai terkecil karena operator AND)
    rules.append(('Buruk',  min(f_pm25['Tinggi'], f_pm10['Tinggi'], f_no2['Tinggi'], f_co['Tinggi'], f_so2['Tinggi'])))  # R2: Semua polutan Tinggi → udara Buruk
    rules.append(('Sedang', min(f_pm25['Sedang'], f_pm10['Sedang'], f_no2['Sedang'], f_co['Sedang'], f_so2['Sedang'])))  # R3: Semua polutan Sedang → udara Sedang
    rules.append(('Sedang', min(f_pm25['Tinggi'], f_pm10['Rendah'], f_no2['Rendah'], f_co['Rendah'], f_so2['Rendah'])))  # R4: Hanya PM2.5 Tinggi, sisanya Rendah → Sedang (satu polutan tidak cukup untuk Buruk)
    rules.append(('Sedang', min(f_pm25['Rendah'], f_pm10['Tinggi'], f_no2['Rendah'], f_co['Rendah'], f_so2['Rendah'])))  # R5: Hanya PM10 Tinggi → Sedang
    rules.append(('Sedang', min(f_pm25['Rendah'], f_pm10['Rendah'], f_no2['Tinggi'], f_co['Rendah'], f_so2['Rendah'])))  # R6: Hanya NO2 Tinggi → Sedang
    rules.append(('Buruk',  min(f_pm25['Tinggi'], f_pm10['Tinggi'], f_no2['Rendah'], f_co['Rendah'], f_so2['Rendah'])))  # R7: PM2.5 dan PM10 sama-sama Tinggi → Buruk (dua partikel berbahaya sekaligus)
    rules.append(('Buruk',  min(f_pm25['Sedang'], f_pm10['Tinggi'], f_no2['Sedang'], f_co['Rendah'], f_so2['Rendah'])))  # R8: PM10 Tinggi + PM2.5 dan NO2 Sedang → Buruk
    rules.append(('Sedang', min(f_pm25['Rendah'], f_pm10['Rendah'], f_no2['Rendah'], f_co['Tinggi'], f_so2['Tinggi'])))  # R9: CO dan SO2 Tinggi tapi partikel lain Rendah → Sedang
    rules.append(('Buruk',  min(f_pm25['Tinggi'], f_pm10['Sedang'], f_no2['Tinggi'], f_co['Sedang'], f_so2['Rendah'])))  # R10: PM2.5 dan NO2 Tinggi, PM10 dan CO Sedang → Buruk
    rules.append(('Baik',   min(f_pm25['Rendah'], f_pm10['Sedang'], f_no2['Rendah'], f_co['Rendah'], f_so2['Sedang'])))  # R11: Hanya PM10 dan SO2 Sedang, sisanya Rendah → masih Baik
    rules.append(('Baik',   min(f_pm25['Sedang'], f_pm10['Rendah'], f_no2['Sedang'], f_co['Rendah'], f_so2['Rendah'])))  # R12: PM2.5 dan NO2 Sedang tapi yang lain Rendah → masih Baik
    rules.append(('Buruk',  min(f_pm25['Tinggi'], f_pm10['Rendah'], f_no2['Tinggi'], f_co['Rendah'], f_so2['Tinggi'])))  # R13: PM2.5, NO2, dan SO2 semuanya Tinggi → Buruk
    rules.append(('Sedang', min(f_pm25['Rendah'], f_pm10['Tinggi'], f_no2['Rendah'], f_co['Tinggi'], f_so2['Rendah'])))  # R14: PM10 dan CO sama-sama Tinggi → Sedang
    rules.append(('Sedang', min(f_pm25['Sedang'], f_pm10['Sedang'], f_no2['Rendah'], f_co['Sedang'], f_so2['Rendah'])))  # R15: PM2.5, PM10, CO Sedang, yang lain Rendah → Sedang

    out_baik   = max([r[1] for r in rules if r[0] == 'Baik']   + [0])  # Kumpulkan semua aturan yang hasilnya 'Baik', ambil nilai terbesar (metode MAX); +[0] agar tidak error jika tidak ada aturan yang cocok
    out_sedang = max([r[1] for r in rules if r[0] == 'Sedang'] + [0])  # Kumpulkan semua aturan yang hasilnya 'Sedang', ambil nilai terbesar
    out_buruk  = max([r[1] for r in rules if r[0] == 'Buruk']  + [0])  # Kumpulkan semua aturan yang hasilnya 'Buruk', ambil nilai terbesar
    
    return out_baik, out_sedang, out_buruk  # Kembalikan tiga nilai keanggotaan output untuk digunakan di tahap defuzzifikasi

def defuzzify_mamdani(out_baik, out_sedang, out_buruk):  # Fungsi untuk mengubah nilai fuzzy menjadi angka AQI nyata menggunakan metode Mamdani (titik berat area)
    num, den = 0, 0  # Siapkan dua variabel: num = pembilang (total bobot×posisi), den = penyebut (total bobot)
    for x in range(0, 501, 5):  # Cek setiap titik dari 0 sampai 500 dengan langkah 5 (cukup detail tapi tidak terlalu berat)
        mu_baik   = min(out_baik,   trapmf(x, 0, 0, 60, 120))          # Potong (clip) tinggi kurva 'Baik' berdasarkan kekuatan aturan
        mu_sedang = min(out_sedang, trapmf(x, 40, 100, 160, 280))      # Potong tinggi kurva 'Sedang' berdasarkan kekuatan aturan
        mu_buruk  = min(out_buruk,  trapmf(x, 50, 150, 500, 500))      # Potong tinggi kurva 'Buruk' berdasarkan kekuatan aturan
        
        mu_agg = max(mu_baik, mu_sedang, mu_buruk)  # Gabungkan tiga kurva dengan ambil nilai tertinggi di titik x (agregasi akhir)
        num += x * mu_agg  # Tambahkan posisi x dikalikan derajat keanggotaan ke pembilang
        den += mu_agg  # Tambahkan derajat keanggotaan ke penyebut
        
    return num / den if den != 0 else 0  # Hitung titik berat (centroid) = pembilang ÷ penyebut; jika penyebut nol kembalikan 0 agar tidak error

def defuzzify_sugeno(out_baik, out_sedang, out_buruk):  # Fungsi defuzzifikasi metode Sugeno: lebih simpel, pakai rata-rata tertimbang
    z_baik, z_sedang, z_buruk = 50, 150, 300  # Nilai tetap (konstanta) mewakili tiap kategori: Baik=50, Sedang=150, Buruk=300
    num = (out_baik * z_baik) + (out_sedang * z_sedang) + (out_buruk * z_buruk)  # Kalikan setiap nilai keanggotaan dengan konstanta kategorinya lalu jumlahkan
    den = out_baik + out_sedang + out_buruk  # Jumlahkan semua nilai keanggotaan sebagai pembagi
    
    return num / den if den != 0 else 0  # Hitung rata-rata tertimbang; jika penyebut nol kembalikan 0 agar tidak error

# ==========================================
# 2. TRAINING ML & DL DAN PERSIAPAN EVALUASI
# ==========================================
@st.cache_resource(show_spinner="Sedang memproses & melatih model dengan 5000 data... Mohon tunggu sekitar 15-30 detik!")  # Dekorator agar hasil fungsi ini disimpan/di-cache, tidak dijalankan ulang setiap kali halaman di-refresh
def train_and_evaluate_models():  # Fungsi utama untuk melatih semua model (Fuzzy, ML, DL) dan menghitung metrik evaluasinya
    current_dir = os.path.dirname(os.path.abspath(__file__))  # Dapatkan lokasi folder tempat file app.py ini berada
    csv_path = os.path.join(current_dir, 'city_day.csv')  # Gabungkan lokasi folder dengan nama file CSV menjadi satu path lengkap
    
    if not os.path.exists(csv_path):  # Cek apakah file CSV benar-benar ada di lokasi tersebut
        return None, None, False, None, None  # Jika file tidak ditemukan, kembalikan semua nilai None dan False sebagai tanda gagal
        
    df = pd.read_csv(csv_path)[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'AQI']].dropna().sample(n=5000, random_state=42)  # Baca CSV, pilih 6 kolom yang dibutuhkan, buang baris kosong, ambil 5000 sampel acak
    actual_aqi = df['AQI'].values  # Simpan nilai AQI asli sebagai acuan pembanding hasil prediksi
    
    # Menghitung Fuzzy Mamdani & Sugeno untuk seluruh data
    mamdani_scores = []  # Siapkan daftar kosong untuk hasil prediksi Mamdani
    sugeno_scores = []  # Siapkan daftar kosong untuk hasil prediksi Sugeno
    
    for _, row in df.iterrows():  # Proses setiap baris data satu per satu
        f_pm25 = fuzzify_input(row['PM2.5'], 'PM2.5')  # Ubah nilai PM2.5 menjadi derajat keanggotaan (Rendah/Sedang/Tinggi)
        f_pm10 = fuzzify_input(row['PM10'], 'PM10')  # Ubah nilai PM10 menjadi derajat keanggotaan
        f_no2  = fuzzify_input(row['NO2'],  'NO2')  # Ubah nilai NO2 menjadi derajat keanggotaan
        f_co   = fuzzify_input(row['CO'],   'CO')  # Ubah nilai CO menjadi derajat keanggotaan
        f_so2  = fuzzify_input(row['SO2'],  'SO2')  # Ubah nilai SO2 menjadi derajat keanggotaan
        out_baik, out_sedang, out_buruk = apply_rules(f_pm25, f_pm10, f_no2, f_co, f_so2)  # Terapkan semua aturan fuzzy untuk mendapatkan kekuatan tiap kategori output
        
        mamdani_scores.append(defuzzify_mamdani(out_baik, out_sedang, out_buruk))  # Hitung AQI metode Mamdani, simpan ke daftar
        sugeno_scores.append(defuzzify_sugeno(out_baik, out_sedang, out_buruk))  # Hitung AQI metode Sugeno, simpan ke daftar
    
    df['Prediksi_Mamdani'] = mamdani_scores  # Tambahkan kolom baru di tabel berisi hasil prediksi Mamdani
    df['Prediksi_Sugeno']  = sugeno_scores  # Tambahkan kolom baru berisi hasil prediksi Sugeno
    
    # Train ML & DL
    X = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'Prediksi_Sugeno']]  # Tentukan fitur input: 5 polutan + hasil Sugeno sebagai fitur tambahan (pendekatan hybrid)
    y = df['AQI']  # Tentukan target output yang ingin diprediksi: nilai AQI asli
    
    ml_model = RandomForestRegressor(n_estimators=50, random_state=42)  # Buat model Random Forest dengan 50 pohon keputusan
    ml_model.fit(X, y)  # Latih model ML menggunakan data fitur (X) dan target (y)
    
    dl_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)  # Buat model Neural Network dengan 2 lapisan tersembunyi (64 dan 32 neuron), maksimal 500 putaran latihan
    dl_model.fit(X, y)  # Latih model DL menggunakan data yang sama
    
    df['Prediksi_ML'] = ml_model.predict(X)  # Gunakan model ML yang sudah dilatih untuk memprediksi seluruh data, simpan hasilnya
    df['Prediksi_DL'] = dl_model.predict(X)  # Gunakan model DL yang sudah dilatih untuk memprediksi seluruh data, simpan hasilnya
    
    # Kalkulasi Metrik Error
    metrics = {  # Buat kamus (dictionary) berisi nilai MAE dan MSE untuk semua metode
        'Mamdani': {'MAE': mean_absolute_error(actual_aqi, mamdani_scores), 'MSE': mean_squared_error(actual_aqi, mamdani_scores)},  # Hitung MAE dan MSE Mamdani
        'Sugeno':  {'MAE': mean_absolute_error(actual_aqi, sugeno_scores),  'MSE': mean_squared_error(actual_aqi, sugeno_scores)},  # Hitung MAE dan MSE Sugeno
        'ML':      {'MAE': mean_absolute_error(actual_aqi, df['Prediksi_ML']), 'MSE': mean_squared_error(actual_aqi, df['Prediksi_ML'])},  # Hitung MAE dan MSE Random Forest
        'DL':      {'MAE': mean_absolute_error(actual_aqi, df['Prediksi_DL']), 'MSE': mean_squared_error(actual_aqi, df['Prediksi_DL'])}  # Hitung MAE dan MSE Neural Network
    }
    
    return ml_model, dl_model, True, df, metrics  # Kembalikan semua model, status berhasil (True), data hasil evaluasi, dan metrik

# Inisialisasi Model
st.set_page_config(page_title="Sistem AQI Hybrid", layout="wide")  # Atur judul tab browser dan gunakan layout lebar (tidak sempit di tengah)
ml_model, dl_model, is_data_loaded, df_eval, eval_metrics = train_and_evaluate_models()  # Jalankan pelatihan model; hasilnya disimpan ke beberapa variabel

if not is_data_loaded:  # Jika pelatihan gagal (file CSV tidak ditemukan)
    st.error("File `city_day.csv` tidak ditemukan di folder yang sama. Pindahkan file tersebut agar model ML/DL bisa dilatih terlebih dahulu.")  # Tampilkan pesan error di halaman web
    st.stop()  # Hentikan eksekusi aplikasi agar tidak lanjut ke bawah

# ==========================================
# 3. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("Navigasi Menu")  # Tampilkan judul "Navigasi Menu" di panel samping kiri
menu = st.sidebar.radio(  # Buat menu pilihan berbentuk tombol radio (hanya satu yang bisa dipilih) di sidebar
    "Pilih Halaman:",  # Label di atas pilihan
    ["Home & Prediksi",  # Pilihan 1: halaman utama dengan input slider dan tombol prediksi
     "Fungsi Keanggotaan",  # Pilihan 2: penjelasan kurva-kurva fuzzy yang digunakan
     "Perbandingan Mamdani vs Sugeno",  # Pilihan 3: penjelasan perbedaan dua metode defuzzifikasi
     "Hasil & Metrik Evaluasi",  # Pilihan 4: tabel dan angka hasil evaluasi semua model
     "Tentang Tugas"]  # Pilihan 5: informasi tentang tugas besar ini
)

st.sidebar.divider()  # Tampilkan garis pemisah horizontal di sidebar
st.sidebar.info("Sistem Penilaian Indeks Kualitas Udara (AQI) Berbasis Parameter Polutan.")  # Tampilkan kotak informasi singkat di bawah garis pemisah

# ==========================================
# 4. KONTEN HALAMAN
# ==========================================

# ----------------- HALAMAN 1 -----------------
if menu == "Home & Prediksi":  # Jika pengguna memilih menu "Home & Prediksi"
    st.title("Sistem Prediksi AQI (Fuzzy + ML + DL)")  # Tampilkan judul besar di halaman
    st.info("**Arsitektur Sistem (Hybrid):** Nilai Polutan -> Logika Fuzzy -> Nilai Fuzzy AQI. Kemudian, Data Polutan + Nilai Fuzzy AQI di- *input* ke dalam algoritma ML (Random Forest) dan DL (Neural Network) untuk dikalibrasi.")  # Tampilkan kotak info penjelasan alur sistem
    
    st.divider()  # Tampilkan garis pemisah horizontal
    col1, col2 = st.columns(2)  # Bagi halaman menjadi 2 kolom berdampingan
    with col1:  # Isi kolom kiri
        pm25 = st.slider("Partikulat (PM2.5)", 0.0, 300.0, 45.0)  # Buat slider PM2.5: nilai minimum 0, maksimum 300, nilai awal 45
        pm10 = st.slider("Partikulat (PM10)",  0.0, 300.0, 60.0)  # Buat slider PM10: nilai awal 60
        no2  = st.slider("Nitrogen Dioksida (NO2)", 0.0, 200.0, 30.0)  # Buat slider NO2: nilai awal 30
    with col2:  # Isi kolom kanan
        co  = st.slider("Karbon Monoksida (CO)", 0.0, 10.0, 1.5)  # Buat slider CO: rentang kecil (0-10) karena satuannya berbeda
        so2 = st.slider("Sulfur Dioksida (SO2)", 0.0, 200.0, 20.0)  # Buat slider SO2: nilai awal 20

    st.divider()  # Tampilkan garis pemisah horizontal
    if st.button("Hitung Prediksi Hybrid AQI", use_container_width=True):  # Buat tombol lebar penuh; kode di bawah dijalankan saat tombol diklik
        # Proses Prediksi
        f_pm25 = fuzzify_input(pm25, 'PM2.5')  # Ubah nilai PM2.5 dari slider menjadi derajat keanggotaan fuzzy
        f_pm10 = fuzzify_input(pm10, 'PM10')  # Ubah nilai PM10 dari slider menjadi derajat keanggotaan fuzzy
        f_no2  = fuzzify_input(no2,  'NO2')  # Ubah nilai NO2 dari slider menjadi derajat keanggotaan fuzzy
        f_co   = fuzzify_input(co,   'CO')  # Ubah nilai CO dari slider menjadi derajat keanggotaan fuzzy
        f_so2  = fuzzify_input(so2,  'SO2')  # Ubah nilai SO2 dari slider menjadi derajat keanggotaan fuzzy
        
        out_baik, out_sedang, out_buruk = apply_rules(f_pm25, f_pm10, f_no2, f_co, f_so2)  # Terapkan 15 aturan fuzzy dan dapatkan kekuatan tiap kategori output
        hasil_mamdani = defuzzify_mamdani(out_baik, out_sedang, out_buruk)  # Hitung nilai AQI akhir menggunakan metode Mamdani
        hasil_sugeno  = defuzzify_sugeno(out_baik, out_sedang, out_buruk)  # Hitung nilai AQI akhir menggunakan metode Sugeno
        
        input_features = pd.DataFrame({  # Buat tabel satu baris berisi nilai polutan + hasil Sugeno sebagai input untuk model ML/DL
            'PM2.5': [pm25], 'PM10': [pm10], 'NO2': [no2],  # Masukkan nilai dari slider ke dalam tabel
            'CO': [co], 'SO2': [so2], 'Prediksi_Sugeno': [hasil_sugeno]  # Hasil Sugeno juga dimasukkan sebagai fitur tambahan
        })
        
        hasil_ml = ml_model.predict(input_features)[0]  # Minta model Random Forest memprediksi AQI berdasarkan input; [0] untuk ambil nilai pertama dari array hasil
        hasil_dl = dl_model.predict(input_features)[0]  # Minta model Neural Network memprediksi AQI; [0] untuk ambil nilai pertama dari array hasil
        
        # Tampilkan Hasil
        st.subheader("Hasil Prediksi Berjenjang (Pipeline)")  # Tampilkan subjudul hasil prediksi
        st.markdown("### Tahap 1: Prediksi Murni Logika Fuzzy")  # Tampilkan teks tebal sebagai pemisah tahap
        res_col1, res_col2 = st.columns(2)  # Bagi area hasil menjadi 2 kolom
        res_col1.metric(label="Fuzzy Mamdani", value=f"{hasil_mamdani:.2f}")  # Tampilkan nilai prediksi Mamdani dalam kotak metrik
        res_col2.metric(label="Fuzzy Sugeno",  value=f"{hasil_sugeno:.2f}")  # Tampilkan nilai prediksi Sugeno dalam kotak metrik
        
        st.markdown("### Tahap 2: Integrasi Lanjutan")  # Tampilkan teks pemisah tahap kedua
        res_col3, res_col4 = st.columns(2)  # Bagi area hasil menjadi 2 kolom lagi
        res_col3.metric(label="Fuzzy + Machine Learning", value=f"{hasil_ml:.2f}", delta="Random Forest",          delta_color="off")  # Tampilkan hasil ML + label keterangan model yang dipakai (delta hanya sebagai label, bukan perubahan)
        res_col4.metric(label="Fuzzy + Deep Learning",    value=f"{hasil_dl:.2f}", delta="Artificial Neural Network", delta_color="off")  # Tampilkan hasil DL + label keterangan model

# ----------------- HALAMAN 2 -----------------
elif menu == "Fungsi Keanggotaan":  # Jika pengguna memilih menu "Fungsi Keanggotaan"
    st.title("Fungsi Keanggotaan (Membership Functions)")  # Tampilkan judul halaman
    st.markdown("Sistem ini menggunakan batas penggaris nilai (himpunan *fuzzy*) yang dibangun dari dasar menggunakan kurva **Trapesium (trapmf)** untuk kategori ujung dan kurva **Segitiga (trimf)** untuk kategori pertengahan.")  # Tampilkan penjelasan singkat dalam format markdown
    
    st.subheader("Parameter Himpunan Input")  # Tampilkan subjudul tabel parameter
    st.markdown(""" 
    | Variabel | Rendah | Sedang | Tinggi |
    | :--- | :--- | :--- | :--- |
    | **PM2.5** | [0, 0, 30, 60] | [30, 60, 90] | [60, 90, 500, 500] |
    | **PM10** | [0, 0, 50, 100] | [50, 100, 150] | [100, 150, 500, 500] |
    | **NO2** | [0, 0, 40, 80] | [40, 80, 120] | [80, 120, 300, 300] |
    | **CO** | [0, 0, 1, 2] | [1, 2, 3] | [2, 3, 50, 50] |
    | **SO2** | [0, 0, 40, 80] | [40, 80, 120] | [80, 120, 300, 300] |
    """)

    st.subheader("Batas Himpunan Output (Mamdani)")  # Tampilkan subjudul untuk bagian output
    st.markdown("""  
    Output Mamdani dikalibrasi menggunakan **trapesium asimetris** (miring) agar titik beratnya bergeser mendekati konstanta yang digunakan oleh Sugeno.
    * **Udara Baik:** Trapesium `[0, 0, 60, 120]`
    * **Udara Sedang:** Trapesium `[40, 100, 160, 280]`
    * **Udara Buruk:** Trapesium `[50, 150, 500, 500]`
    """)

# ----------------- HALAMAN 3 -----------------
elif menu == "Perbandingan Mamdani vs Sugeno":  # Jika pengguna memilih menu perbandingan metode
    st.title("Perbandingan Metode Defuzzifikasi")  # Tampilkan judul halaman
    
    st.markdown("### 1. Metode Mamdani (Integral Titik Berat)")  # Tampilkan subjudul metode Mamdani
    st.write("""  
    Metode Mamdani beroperasi dengan cara memotong luas area dari geometri trapesium berdasarkan kekuatan aturan (rule) yang menyala, lalu menggabungkannya. Tebakan akhir diambil dari titik berat (Center of Gravity) bangun ruang tersebut. Hal ini membuat Mamdani cenderung mencari \"titik tengah/kompromi\" dan angkanya selalu berupa desimal yang dinamis mengikuti pergeseran luas area.
    """)
    
    st.markdown("### 2. Metode Sugeno (Rata-rata Tertimbang)")  # Tampilkan subjudul metode Sugeno
    st.write("""  
    Metode Sugeno jauh lebih sederhana dan tegas secara komputasi. Ia mengunci keputusan pada nilai konstanta mutlak (**Baik = 50**, **Sedang = 150**, **Buruk = 300**). Keputusan akhirnya adalah rata-rata terbobot dari konstanta tersebut. Inilah mengapa jika hanya satu aturan dominan yang menyala, output Sugeno akan mengunci persis di angka bulat (misal: 150.0).
    """)
    
    st.info("**Catatan Kalibrasi:** Agar selisih (gap) perbandingan prediksi keduanya tidak meleset terlalu jauh, kurva area Mamdani pada sistem ini memiliki bentuk asimetris, sehingga letak titik keseimbangannya mendekati angka 50, 150, dan 300 milik Sugeno.")  # Tampilkan kotak info berwarna biru berisi catatan kalibrasi

# ----------------- HALAMAN 4 -----------------
elif menu == "Hasil & Metrik Evaluasi":  # Jika pengguna memilih menu evaluasi
    st.title("Hasil Prediksi dan Performa Sistem")  # Tampilkan judul halaman
    st.markdown("Evaluasi dihitung menggunakan 5000 sampel acak dari dataset. Metrik yang digunakan adalah **Mean Absolute Error (MAE)** dan **Mean Squared Error (MSE)** untuk membandingkan tebakan dengan nilai *Actual AQI*.")  # Tampilkan keterangan cara evaluasi
    
    col1, col2, col3, col4 = st.columns(4)  # Bagi halaman menjadi 4 kolom berdampingan untuk kotak metrik
    col1.metric("MAE Mamdani",         f"{eval_metrics['Mamdani']['MAE']:.2f}")  # Tampilkan nilai MAE Mamdani dalam kotak (2 desimal)
    col2.metric("MAE Sugeno",          f"{eval_metrics['Sugeno']['MAE']:.2f}")  # Tampilkan nilai MAE Sugeno
    col3.metric("MAE Machine Learning",f"{eval_metrics['ML']['MAE']:.2f}", delta="-Lebih Akurat", delta_color="inverse")  # Tampilkan MAE ML + label hijau "Lebih Akurat" (delta inverse = hijau saat negatif)
    col4.metric("MAE Deep Learning",   f"{eval_metrics['DL']['MAE']:.2f}", delta="-Lebih Akurat", delta_color="inverse")  # Tampilkan MAE DL + label hijau "Lebih Akurat"
    
    st.divider()  # Tampilkan garis pemisah horizontal
    st.subheader("Tabel Komparasi Prediksi (10 Sampel Acak Pertama)")  # Tampilkan subjudul tabel perbandingan
    
    # Menyiapkan tabel untuk ditampilkan
    df_show = df_eval[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'AQI', 'Prediksi_Mamdani', 'Prediksi_Sugeno', 'Prediksi_ML', 'Prediksi_DL']].head(10).copy()  # Ambil 10 baris pertama dengan kolom yang diinginkan, salin agar data asli tidak berubah
    df_show = df_show.rename(columns={'AQI': 'Actual_AQI'})  # Ganti nama kolom 'AQI' menjadi 'Actual_AQI' agar lebih jelas di tabel
    df_show['Prediksi_Mamdani'] = df_show['Prediksi_Mamdani'].round(2)  # Bulatkan hasil Mamdani jadi 2 angka desimal agar rapi
    df_show['Prediksi_Sugeno']  = df_show['Prediksi_Sugeno'].round(2)  # Bulatkan hasil Sugeno jadi 2 angka desimal
    df_show['Prediksi_ML']      = df_show['Prediksi_ML'].round(2)  # Bulatkan hasil ML jadi 2 angka desimal
    df_show['Prediksi_DL']      = df_show['Prediksi_DL'].round(2)  # Bulatkan hasil DL jadi 2 angka desimal
    df_show['Selisih (Mam-Sug)'] = abs(df_show['Prediksi_Mamdani'] - df_show['Prediksi_Sugeno']).round(2)  # Hitung dan tampilkan selisih absolut antara prediksi Mamdani dan Sugeno
    
    st.dataframe(df_show, use_container_width=True)  # Tampilkan tabel interaktif yang bisa di-scroll, lebarnya menyesuaikan layar

# ----------------- HALAMAN 5 -----------------
elif menu == "Tentang Tugas":  # Jika pengguna memilih menu "Tentang Tugas"
    st.title("Tentang Tugas Besar")  # Tampilkan judul halaman
    st.markdown("### Judul Tugas")  # Tampilkan subjudul
    st.write("**Sistem Penilaian Indeks Kualitas Udara (AQI) Berbasis Parameter Polutan.**")  # Tampilkan judul tugas dalam format tebal
    
    st.markdown("### Metode yang Digunakan")  # Tampilkan subjudul daftar metode
    st.markdown("""
    1. **Logika Fuzzy (From Scratch):** Metode Mamdani (Centroid) dan Sugeno (Weighted Average) dengan 15 basis aturan (*Rule Base*).
    2. **Machine Learning:** Algoritma *Random Forest Regressor* untuk kalibrasi skor berbasis *decision tree*.
    3. **Deep Learning:** Algoritma *Artificial Neural Network* (Multi-Layer Perceptron) untuk pencarian pola non-linier lanjutan.
    """)
    
    st.markdown("### Arsitektur Sistem (Hybrid Pipeline)")  # Tampilkan subjudul arsitektur
    st.write("Sistem membaca 5 indikator kualitas udara (PM2.5, PM10, NO2, CO, SO2) -> Menerjemahkannya melalui proses Fuzzifikasi -> Menghitung skor batas menggunakan Inferensi Fuzzy -> Menggabungkannya dengan data mentah -> Mengkalibrasi hasil tebakan akhir menggunakan *Machine Learning / Deep Learning* untuk mengejar akurasi *Actual AQI*.")  # Tampilkan penjelasan alur sistem dalam satu paragraf panjang
    
    st.markdown("### Sumber Dataset")  # Tampilkan subjudul sumber data
    st.write("Dataset yang digunakan diolah dari data polusi udara kota historis (India).")  # Tampilkan keterangan asal dataset
    st.markdown("[🔗 Buka Dataset di Kaggle (Air Quality Data in India)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)")  # Tampilkan tautan klik ke halaman dataset di Kaggle
