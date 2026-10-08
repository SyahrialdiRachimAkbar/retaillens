# Mulai dari sini, bang

RetailLens adalah satu project lengkap: analisis transaksi, segmentasi pelanggan,
prediksi pembelian ulang, dashboard interaktif, notebook, dan laporan hasil.

## Buka dashboard

Di PowerShell, masuk ke folder project lalu jalankan:

```powershell
cd 'C:\Users\Hype AMD\repos\retaillens'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka alamat localhost yang muncul. Kalau file data belum dibuat, jalankan dulu:

```powershell
.\.venv\Scripts\python.exe -m retaillens.pipeline
```

## Urutan belajar

1. Baca `reports/findings.md` untuk memahami hasil dan batasannya.
2. Buka `notebooks/retail_analysis.ipynb` untuk mengikuti analisis.
3. Coba filter negara, lihat kelompok pelanggan, dan unduh ranking prediksi.
4. Baca halaman **Model lab**: bandingkan model dengan baseline.
5. Pelajari `retaillens/pipeline.py`, lalu jalankan `python -m pytest` di environment project.

## Cara menjelaskan saat interview

“Saya menggunakan data transaksi publik untuk memahami kelompok pelanggan dan
memprediksi pembelian ulang dalam 30 hari. Saya membuat pipeline yang bisa dijalankan
ulang, mengevaluasi model berdasarkan urutan waktu untuk mencegah kebocoran data,
dan menampilkan hasilnya lewat dashboard. Rekomendasi kampanyenya berupa usulan
eksperimen, karena data ini belum membuktikan dampak intervensi.”

Angka kinerja yang benar ada di laporan hasil. Jangan mengartikan prediksi pembelian
sebagai bukti bahwa kampanye pasti meningkatkan penjualan.
