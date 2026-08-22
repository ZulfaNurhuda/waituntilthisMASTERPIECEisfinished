# Frontend

Form input tunggal → tampilan output. 

## Alur

1. Pengguna pilih komoditas 
2. Pengguna input riwayat suhu/upload CSV (`timestamp,suhu_celsius`) atau isi manual beberapa baris.
3. Klik "Prediksi" → kirim POST ke backend.
4. Tampilkan hasil: jam/hari tersisa, status (`layak` / `waspada` / `segera_tindak_lanjuti`), confidence, rentang interval.

## API yang dipanggil

```
POST http://backend:8000/(placeholder)
Body: {
  
}
```


## Setup

```bash

```