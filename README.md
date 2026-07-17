# 🌱 SmartGreenify

**A Raspberry Pi smart-garden dashboard that combines real sensor monitoring, safe irrigation controls, local AutoML recommendations, and reporting.**

SmartGreenify reads the BME280, ADS1115 soil-moisture sensor, and LDR; it can drive a pump relay, show a live web dashboard, create reports, and select an irrigation model from several local ML candidates.

## Highlights

- Live dashboard with WebSocket reconnection and HTTP fallback
- Six sensor and garden-health summary cards
- Soil, temperature, humidity, pressure, and multi-series environment charts
- Installable PWA with a safe app-shell cache strategy
- Local AutoML selection: Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting, and optional XGBoost
- Manual, scheduled, and model-assisted irrigation workflows
- Plotly analytics, PDF/Excel exports, Ntfy notifications, and rotating logs

## Hardware map

| Component | Connection | App setting |
|---|---:|---|
| BME280 | SPI `0.0` (GPIO 8–11) | `SPI_BUS=0`, `SPI_DEVICE=0` |
| ADS1115 ADC | I²C `0x48` (GPIO 2–3) | `I2C_BUS=1` |
| Capacitive soil sensor | ADS1115 A0 | ADC channel 0 |
| LDR | GPIO 26 | `LDR_PIN=26` |
| Pump relay | GPIO 27 | `RELAY_PIN=27` |

> Hardware safety: the project preserves the existing sensor buses and GPIO assignments. Do not change pin assignments unless the physical wiring changes too.

## Quick start

### 1. Prepare Raspberry Pi OS

```bash
sudo apt update
sudo apt install -y python3-pip python3-lgpio python3-spidev python3-smbus2
```

Enable **SPI** and **I²C** using `raspi-config`, then reboot.

### 2. Install and run

```bash
git clone https://github.com/kdorukdemirtas-star/SmartGreenify.git
cd SmartGreenify
python3 -m pip install -r requirements.txt
python3 SmartGreenify.py
```

Open `http://<raspberry-pi-ip>:5050`.

XGBoost is optional: if it is not installed, AutoML continues with the available scikit-learn models.

## Dashboard and visualizations

The dashboard displays current sensor readings and the following visualizations:

- **Soil moisture trend** for short-term irrigation context
- **Temperature, humidity, and pressure trends** for environmental monitoring
- **Environmental overview** combining soil moisture, air humidity, and temperature in one comparison chart
- **Garden-health cards** for daily irrigation count, total irrigations, and current plant-health score
- **Analytics pages** for Plotly dashboard, irrigation heatmap, and sensor correlation matrix

## How Codex and GPT-5.6 were used

Codex and GPT-5.6 were used as development collaborators to improve the software around the existing hardware design. They helped review the project structure, strengthen the AutoML workflow, refine the responsive dashboard, add visualization ideas, improve PWA and WebSocket resilience, and rewrite the documentation in English.

The hardware decisions remained under project control: sensor buses, GPIO assignments, and irrigation behavior were intentionally preserved. Codex and GPT-5.6 supported implementation and documentation work; they do not operate the physical garden or replace human supervision of the pump and electrical setup.

## How AutoML works

1. The application reads valid records from `data_log.csv`.
2. With at least 30 records, it keeps the newest 20% as a chronological validation set.
3. It compares candidates by mean absolute error (MAE).
4. It retrains the selected model on the full available data and stores it in `ml_model.pkl`.

Inputs are soil moisture, temperature, humidity, daylight, hour, and pump activity. The model only reads logged data; it does not access GPIO, SPI, I²C, or relay configuration. Successful models retrain weekly; when data is insufficient, retraining is retried at most once per hour to conserve CPU.

## Configuration

Settings live in the `Config` class in `SmartGreenify.py`.

| Setting | Default | Meaning |
|---|---:|---|
| `UPDATE_INTERVAL` | `2` seconds | Sensor polling interval |
| `FLASK_PORT` | `5050` | Dashboard port |
| `AUTO_IRRIGATION_ENABLED` | `True` | Enables automatic irrigation |
| `AUTO_IRRIGATION_DURATION` | `60` seconds | Automatic irrigation duration |
| `AUTO_IRRIGATION_MIN_INTERVAL` | `3600` seconds | Minimum delay between automatic runs |
| `ML_RETRAIN_INTERVAL` | `7 days` | Interval after a successful model training |

Set a persistent session secret before starting the app:

```bash
export SMARTGREENIFY_SECRET_KEY='a-long-random-secret'
python3 SmartGreenify.py
```

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/data` | Current dashboard data |
| `POST` | `/manual_irrigation` | Start or stop manual irrigation |
| `POST` | `/select_plant` | Select a plant profile |
| `POST` | `/add_schedule` | Add a schedule |
| `POST` | `/delete_schedule` | Delete a schedule |
| `GET` | `/analytics/dashboard` | Analytics dashboard |
| `GET` | `/analytics/export_pdf` | Create PDF report |
| `GET` | `/analytics/export_excel` | Create Excel report |

## Runtime files

```text
data_log.csv              # Sensor records
schedule.json             # Schedules
statistics.json           # App statistics
ml_model.pkl              # Selected AutoML model
logs/system.log           # System events
logs/sensor_readings.log  # Sensor reads
logs/irrigation.log       # Irrigation events
logs/ml_training.log      # Model training
logs/performance.log      # Runtime and CSV metrics
logs/errors.log           # Errors only
```

## Troubleshooting

**No sensor values**

```bash
ls /dev/spi*   # expect /dev/spidev0.0
ls /dev/i2c*   # expect /dev/i2c-1
sudo i2cdetect -y 1  # expect 0x48
```

**Pump does not run**

Check the power supply, relay, and GPIO 27 wiring before testing. Never leave pump testing unattended.

**AutoML is unavailable**

Ensure `numpy` and `scikit-learn` are installed and at least 30 valid log records exist. XGBoost remains optional.

## Security notes

- Keep the dashboard on a trusted local network.
- Do not expose the pump controls directly to the public internet.
- Use a VPN or an authenticated reverse proxy for remote access.
- Use appropriate electrical protection and water-safe enclosure practices.

## License

[MIT](LICENSE)
