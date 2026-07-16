# 🌱 SmartGreenify 

**IoT Sensör Ağı ve ML Tabanlı Akıllı Sera Sistemi**



---

## Genel Bakış

SmartGreenify, **Raspberry Pi 5** üzerinde çalışan yapay zeka destekli otonom sera sistemidir.

###  Temel Özellikler
-  **AutoML** ile en uygun sulama tahmini (Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting ve isteğe bağlı XGBoost)
-  **%40 su tasarrufu** (geleneksel yönteme göre)
-  **Real-time dashboard** (WebSocket + Chart.js)
-  **PWA desteği** - her cihazda çalışır
-  **Ntfy.sh bildirimleri**
-  **Kapsamlı loglama** (6 ayrı log dosyası)
-  **Analytics & Raporlar** (PDF/Excel/Plotly)

### 📈 Test Sonuçları (15 gün)
-  **%99.8** başarı oranı
-  **357.2 saat** kesintisiz çalışma
-  **4,318/4,320** başarılı kayıt
-  **R² = 0.847** ML model performansı

---

## 🛠 Donanım

| Bileşen | Model | Pin |
|---------|-------|-----|
| **Ana işlemci** | Raspberry Pi 5 (4GB) | - |
| **Sıcaklık/Nem/Basınç** | BME280 | SPI (GPIO 8-11) |
| **ADC** | ADS1115 | I²C (GPIO 2-3) |
| **Toprak Nemi** | Kapasitif Sensör | ADS1115 A0 |
| **Işık** | LDR Modülü | GPIO 26 |
| **Pompa** | 5V Su Pompası | Röle GPIO 27 |

---

##  Kurulum

### 1. Sistem Hazırlığı
```bash
# Raspberry Pi OS güncellemesi
sudo apt update && sudo apt upgrade -y

# Python paketleri
sudo apt install python3-pip python3-lgpio python3-spidev python3-smbus2 -y
```

### 2. Python Kütüphaneleri
```bash
# Zorunlu
pip3 install flask requests

# WebSocket (opsiyonel ama önerilen)
pip3 install flask-socketio

# AutoML özellikleri
pip3 install numpy scikit-learn xgboost

# Analytics & Raporlar
pip3 install plotly pandas reportlab openpyxl
```

### 3. SPI/I²C Aktifleştirme
```bash
sudo raspi-config
# Interface Options → SPI → Enable
# Interface Options → I2C → Enable
# Reboot
```

### 4. Projeyi Çalıştır
```bash
python3 smartgreenify.py
```

Tarayıcıda: `http://localhost:5050`

---

##  Konfigürasyon

`Config` sınıfındaki ayarlar:

```python
UPDATE_INTERVAL = 1          # Sensör okuma (saniye)
MAX_HISTORY = 60             # Bellek grafiği (60 saniye)
FLASK_PORT = 5050            # Web arayüzü portu
AUTO_IRRIGATION_ENABLED = True  # Otomatik sulama
NTFY_ENABLED = True          # Bildirimleri aç/kapat
NTFY_TOPIC = "sg_bahce_2025" # Ntfy topic ismi
```

### Bitki Profilleri
```python
plant_profiles = {
    "Roka": {"min_moisture": 45, "max_temp": 26, "icon": "🌱"},
    "Domates": {"min_moisture": 50, "max_temp": 30, "icon": "🍅"},
    "Fesleğen": {"min_moisture": 40, "max_temp": 28, "icon": "🌿"},
    # ...
}
```

---

## 📊 Özellikler

### Web Arayüzü
- **Ana Sayfa:** `/` - Dashboard, grafikler, kontroller
- **Analytics:** `/analytics/dashboard` - Plotly dashboard
- **Heatmap:** `/analytics/heatmap` - Sulama zaman dağılımı
- **Korelasyon:** `/analytics/correlation` - Sensör korelasyonları

### API Endpoints
```bash
GET  /data                    # Sensör verileri (JSON)
POST /manual_irrigation       # Manuel sulama
POST /select_plant            # Bitki değiştir
POST /add_schedule            # Zamanlayıcı ekle
POST /delete_schedule         # Zamanlayıcı sil
GET  /analytics/export_pdf    # PDF rapor indir
GET  /analytics/export_excel  # Excel rapor indir
```

### AutoML Makine Öğrenmesi
- **Algoritmalar:** Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting ve (kuruluysa) XGBoost
- **Seçim:** Son %20 zaman sıralı kayıt üzerinde en düşük MAE'ye sahip model otomatik seçilir; gelecek veriler eğitim setine karışmaz.
- **Eğitim:** Her 7 günde otomatik
- **Girdi:** Toprak nemi, sıcaklık, hava nemi, ışık, saat
- **Çıktı:** Optimal sulama saati (0-23)
- **Model dosyası:** `ml_model.pkl`
- **Donanım güvenliği:** AutoML yalnızca `data_log.csv` dosyasını okur. Sensör, GPIO, SPI/I²C ve röle pin yapılandırmasını değiştirmez.

### Loglama
```
logs/
├── system.log          # Genel sistem olayları
├── sensor_readings.log # Her sensör okuması
├── irrigation.log      # Sulama işlemleri
├── ml_training.log     # ML eğitim detayları
├── performance.log     # Uptime, performans
└── errors.log          # Sadece hatalar
```

### Bildirimler (Ntfy.sh)
1. Ntfy uygulamasını yükle (Android/iOS)
2. Topic ekle: 
3. Sistem otomatik bildirim gönderir:
   -  Sulama başladı/bitti
   -  Düşük toprak nemi
   -  ML modeli güncellendi

---

##  Kullanım

### Manuel Sulama
```python
# Web arayüzünden
Süre seç (30s, 1dk, 2dk, 5dk) → Başlat

# veya API ile
curl -X POST http://localhost:5050/manual_irrigation \
  -H "Content-Type: application/json" \
  -d '{"action":"start","duration":60}'
```

### Zamanlayıcı Ekle
```python
# Web arayüzünden
Saat:7, Dakika:0 → Ekle

# veya API ile
curl -X POST http://localhost:5050/add_schedule \
  -H "Content-Type: application/json" \
  -d '{"hour":7,"minute":0}'
```

---

##  Sorun Giderme

### Sensör Okumuyor
```bash
# SPI/I2C kontrol
ls /dev/spi*   # /dev/spidev0.0 görünmeli
ls /dev/i2c*   # /dev/i2c-1 görünmeli

# I2C adres tara
sudo i2cdetect -y 1  # 0x48 (ADS1115) görünmeli
```

### Pompa Çalışmıyor
```bash
# GPIO test
python3 -c "import lgpio; h=lgpio.gpiochip_open(0); lgpio.gpio_claim_output(h,27); lgpio.gpio_write(h,27,0)"
```



---

## Veri Dosyaları

```
data_log.csv          # Sensör kayıtları
schedule.json         # Zamanlayıcılar
statistics.json       # İstatistikler
ml_model.pkl          # ML modeli
static/
  ├── smartgreenify_report.pdf   # PDF rapor
  └── smartgreenify_report.xlsx  # Excel rapor
```

---

##  Güvenlik

- **Atomic file write:** Veri kaybı önleme
- **Thread-safe:** Eş zamanlı erişim koruması
- **Error handling:** Try-catch blokları
- **Auto recovery:** Kesinti sonrası otomatik devam

---



##  İletişim & Destek

kdorukdemirtas@hotmail.com
---


