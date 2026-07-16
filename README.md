# 🌱 SmartGreenify

**Raspberry Pi üzerinde çalışan, gerçek sensör verileriyle sulamayı izleyen ve yerel AutoML ile öneri üreten akıllı sera uygulaması.**

SmartGreenify; BME280, ADS1115, kapasitif toprak nem sensörü, LDR ve röle ile çalışır. Canlı gösterge paneli, PWA kurulumu, bildirimler, analitik raporlar ve makine öğrenmesi önerilerini tek bir Python uygulamasında bir araya getirir.

> Donanım güvenliği: Bu proje mevcut SPI/I²C yapılandırmasını, LDR için **GPIO 26** ve röle için **GPIO 27** atamasını korur.

## Öne çıkanlar

- Canlı sensör paneli ve Chart.js grafikleri
- Dayanıklı WebSocket bağlantısı: otomatik yeniden bağlanma ve HTTP yenileme yedeği
- Kurulabilir PWA ve güvenli uygulama-kabuğu önbelleği
- Yerel AutoML: Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting ve isteğe bağlı XGBoost
- PDF / Excel / Plotly analitik raporları
- Zamanlı, manuel ve ML destekli sulama akışları
- Ntfy bildirimleri ve dönen sistem, sensör, sulama, ML ve performans günlükleri

## Donanım bağlantıları

| Bileşen | Bağlantı | Uygulamadaki karşılığı |
|---|---:|---|
| BME280 sıcaklık / nem / basınç | SPI `0.0` (GPIO 8–11) | `SPI_BUS=0`, `SPI_DEVICE=0` |
| ADS1115 ADC | I²C `0x48` (GPIO 2–3) | `I2C_BUS=1` |
| Kapasitif toprak nem sensörü | ADS1115 A0 | ADC kanal 0 |
| LDR | GPIO 26 | `LDR_PIN=26` |
| Pompa rölesi | GPIO 27 | `RELAY_PIN=27` |

## Hızlı başlangıç

### 1. Raspberry Pi hazırlığı

```bash
sudo apt update
sudo apt install -y python3-pip python3-lgpio python3-spidev python3-smbus2
```

`raspi-config` içinden **SPI** ve **I²C** arayüzlerini etkinleştirin, ardından cihazı yeniden başlatın.

### 2. Uygulamayı kurun ve başlatın

```bash
git clone https://github.com/kdorukdemirtas-star/SmartGreenify.git
cd SmartGreenify
python3 -m pip install -r requirements.txt
python3 SmartGreenify.py
```

Panel: `http://<raspberry-pi-ip>:5050`

XGBoost kurulamazsa uygulama çalışmaya devam eder; AutoML diğer scikit-learn modellerini karşılaştırır.

## Günlük kullanım

- **Panel:** canlı değerleri, bitki profilini, pompaları ve programları yönetin.
- **Manuel sulama:** süreyi seçip başlatın; pompa durumunu panelden takip edin.
- **Zamanlama:** saat ve dakikayı ekleyin; gereksiz programları silin.
- **Raporlar:** paneldeki Analytics alanından Plotly, PDF veya Excel çıktısı alın.

## AutoML nasıl çalışır?

1. Sistem `data_log.csv` içindeki sensör kayıtlarını kullanır.
2. En az 30 kayıt olduğunda zaman sırasını bozmadan son %20’yi doğrulama için ayırır.
3. Aday modelleri MAE’ye göre karşılaştırır ve en iyi modeli seçer.
4. Seçilen model tüm mevcut veriyle yeniden eğitilir ve `ml_model.pkl` dosyasına kaydedilir.

Girdiler: toprak nemi, sıcaklık, hava nemi, ışık durumu, saat ve pompa durumu. Çıktı, sulama için önerilen saattir. Model yalnızca günlük dosyasını okur; GPIO, röle, SPI veya I²C ayarlarına erişmez.

Eğitim başarılıysa yedi günde bir yenilenir. Veri henüz yeterli değilse sistem, kaynak tüketimini sınırlamak için en fazla saatte bir tekrar dener.

## Yapılandırma

Tüm uygulama ayarları `SmartGreenify.py` içindeki `Config` sınıfındadır.

| Ayar | Varsayılan | Açıklama |
|---|---:|---|
| `UPDATE_INTERVAL` | `2` sn | Sensör okuma aralığı |
| `FLASK_PORT` | `5050` | Web paneli portu |
| `AUTO_IRRIGATION_ENABLED` | `True` | Otomatik sulamayı açar/kapatır |
| `AUTO_IRRIGATION_DURATION` | `60` sn | Otomatik sulama süresi |
| `AUTO_IRRIGATION_MIN_INTERVAL` | `3600` sn | İki otomatik sulama arasındaki alt sınır |
| `ML_RETRAIN_INTERVAL` | `7 gün` | Başarılı eğitimler arasındaki süre |

Oturum anahtarını sabit kodlamak yerine başlatma öncesinde ortam değişkeniyle belirleyebilirsiniz:

```bash
export SMARTGREENIFY_SECRET_KEY='uzun-ve-rastgele-bir-deger'
python3 SmartGreenify.py
```

## API özeti

| Yöntem | Uç nokta | Amaç |
|---|---|---|
| `GET` | `/data` | Güncel sensör ve panel verisi |
| `POST` | `/manual_irrigation` | Manuel sulama başlat / durdur |
| `POST` | `/select_plant` | Bitki profili seç |
| `POST` | `/add_schedule` | Sulama programı ekle |
| `POST` | `/delete_schedule` | Sulama programı sil |
| `GET` | `/analytics/dashboard` | Analitik görünüm |
| `GET` | `/analytics/export_pdf` | PDF raporu üret |
| `GET` | `/analytics/export_excel` | Excel raporu üret |

## Dosyalar ve günlükler

```text
data_log.csv              # Sensör verileri
schedule.json             # Sulama programları
statistics.json           # Uygulama istatistikleri
ml_model.pkl              # Seçilen AutoML modeli
logs/system.log           # Sistem olayları
logs/sensor_readings.log  # Sensör okumaları
logs/irrigation.log       # Sulama olayları
logs/ml_training.log      # Model eğitimi
logs/performance.log      # Çalışma ve CSV metrikleri
logs/errors.log           # Hatalar
```

## Sorun giderme

**Sensör verisi gelmiyor**

```bash
ls /dev/spi*   # /dev/spidev0.0 beklenir
ls /dev/i2c*   # /dev/i2c-1 beklenir
sudo i2cdetect -y 1  # 0x48 beklenir
```

**Pompa çalışmıyor**

Önce güç kaynağını, röleyi ve GPIO 27 bağlantısını fiziksel olarak kontrol edin. Pompayı test ederken su tesisatını gözetimsiz bırakmayın.

**AutoML görünmüyor**

`numpy` ve `scikit-learn` paketlerinin kurulu olduğundan ve günlükte en az 30 geçerli kayıt bulunduğundan emin olun. XGBoost isteğe bağlıdır.

## Güvenlik notları

- Uygulamayı yalnızca güvendiğiniz yerel ağlarda erişilebilir yapın.
- Varsayılan pin atamalarını fiziksel tesisata uymadan değiştirmeyin.
- Pompa ve röle için uygun güç kaynağı, sigorta ve suya karşı yalıtım kullanın.
- Üretimde paneli doğrudan internete açmak yerine VPN veya ters vekil üzerinden kimlik doğrulama ekleyin.

## Lisans

[MIT](LICENSE)
