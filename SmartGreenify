#!/usr/bin/env python3
# SmartGreenify v4.0 - GERÇEK SENSÖRLER + Analytics + PWA + Ntfy + Comprehensive Logging
# BME280 (SPI) + ADS1115 (I2C) + LDR (Digital) + Röle

import threading, time, csv, math, json, os, atexit, signal, sys, random
from datetime import datetime, timedelta
from statistics import mean
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

# Gerçek sensör kütüphaneleri
try:
    import lgpio
    import spidev
    from smbus2 import SMBus
    SENSORS_AVAILABLE = True
    print("✅ Sensör kütüphaneleri yüklü")
except ImportError as e:
    SENSORS_AVAILABLE = False
    print(f"❌ Sensör kütüphaneleri yok: {e}")
    print("   Kurulum: sudo apt install python3-lgpio python3-spidev python3-smbus2")
    sys.exit(1)

# Ntfy için requests
try:
    import requests
    REQUESTS_AVAILABLE = True
    print("✅ Requests yüklü")
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ Requests yok - pip install requests (Ntfy kapalı)")

# ML kütüphaneleri
try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
    print("✅ ML kütüphaneleri yüklü")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML kapalı - pip install numpy scikit-learn")

# Plotly
try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    import pandas as pd
    PLOTLY_AVAILABLE = True
    print("✅ Plotly yüklü")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly kapalı - pip install plotly pandas")

# PDF/Excel
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    PDF_AVAILABLE = True
    print("✅ PDF yüklü")
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️ PDF kapalı - pip install reportlab")

try:
    from openpyxl import Workbook
    from openpyxl.chart import LineChart, BarChart, Reference
    from openpyxl.styles import Font, PatternFill, Alignment
    EXCEL_AVAILABLE = True
    print("✅ Excel yüklü")
except ImportError:
    EXCEL_AVAILABLE = False
    print("⚠️ Excel kapalı - pip install openpyxl")

try:
    from flask_socketio import SocketIO
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

# Config
class Config:
    UPDATE_INTERVAL = 2  # 2 saniye sensör okuma
    MAX_HISTORY = 60
    DEFAULT_PLANT = "Roka"
    FLASK_PORT = 5050
    CSV_FILE = "data_log.csv"
    SCHEDULE_FILE = "schedule.json"
    STATS_FILE = "statistics.json"
    STATIC_FOLDER = "static"
    
    # Log dosyaları
    LOG_FOLDER = "logs"
    SYSTEM_LOG = "logs/system.log"
    SENSOR_LOG = "logs/sensor_readings.log"
    IRRIGATION_LOG = "logs/irrigation.log"
    ML_LOG = "logs/ml_training.log"
    PERFORMANCE_LOG = "logs/performance.log"
    ERROR_LOG = "logs/errors.log"
    
    # Ntfy ayarları
    NTFY_ENABLED = True
    NTFY_TOPIC = f"sg_bahce_{random.randint(1000,9999)}"
    
    # Hardware pinler
    SPI_BUS = 0
    SPI_DEVICE = 0
    I2C_BUS = 1
    ADS_ADDR = 0x48
    RELAY_PIN = 27
    LDR_PIN = 26
    
    # Toprak nem eşikleri
    SOIL_DRY = 15000
    SOIL_WET = 8000
    
    # Otomatik sulama
    AUTO_IRRIGATION_ENABLED = True
    AUTO_IRRIGATION_DURATION = 60
    AUTO_IRRIGATION_MIN_INTERVAL = 3600

# Log klasörünü oluştur
if not os.path.exists(Config.LOG_FOLDER):
    os.makedirs(Config.LOG_FOLDER)

if not os.path.exists(Config.STATIC_FOLDER):
    os.makedirs(Config.STATIC_FOLDER)

# Loglama sistemi konfigürasyonu
def setup_logging():
    """Kapsamlı loglama sistemini ayarla"""
    
    # Ana logger
    main_logger = logging.getLogger('smartgreenify')
    main_logger.setLevel(logging.DEBUG)
    
    # System log (genel sistem olayları)
    system_handler = RotatingFileHandler(
        Config.SYSTEM_LOG, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    system_handler.setLevel(logging.INFO)
    system_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    system_handler.setFormatter(system_formatter)
    
    # Error log (sadece hatalar)
    error_handler = RotatingFileHandler(
        Config.ERROR_LOG, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(system_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(system_formatter)
    
    main_logger.addHandler(system_handler)
    main_logger.addHandler(error_handler)
    main_logger.addHandler(console_handler)
    
    return main_logger

logger = setup_logging()

# Özelleştirilmiş loggerlar
class SensorLogger:
    """Sensör okumalarını loglar"""
    def __init__(self):
        self.logger = logging.getLogger('smartgreenify.sensor')
        handler = RotatingFileHandler(
            Config.SENSOR_LOG, maxBytes=20*1024*1024, backupCount=10, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        self.total_reads = 0
        self.failed_reads = 0
    
    def log_reading(self, soil, temp, hum, pressure, light, pump_on):
        """Her sensör okumasını logla"""
        self.total_reads += 1
        self.logger.info(
            f"SOIL={soil:.2f}% | TEMP={temp:.2f}°C | HUM={hum:.2f}% | "
            f"PRESS={pressure:.2f}hPa | LIGHT={'DAY' if light else 'NIGHT'} | "
            f"PUMP={'ON' if pump_on else 'OFF'}"
        )
    
    def log_error(self, sensor_name, error):
        """Sensör okuma hatasını logla"""
        self.failed_reads += 1
        self.logger.error(f"❌ {sensor_name} OKUMA HATASI: {error}")
        logger.error(f"Sensör hatası: {sensor_name} - {error}")
    
    def get_success_rate(self):
        """Başarı oranını hesapla"""
        if self.total_reads == 0:
            return 100.0
        return ((self.total_reads - self.failed_reads) / self.total_reads) * 100

class IrrigationLogger:
    """Sulama olaylarını loglar"""
    def __init__(self):
        self.logger = logging.getLogger('smartgreenify.irrigation')
        handler = RotatingFileHandler(
            Config.IRRIGATION_LOG, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.total_irrigations = 0
        self.ml_irrigations = 0
        self.manual_irrigations = 0
    
    def log_start(self, reason, soil_before, duration):
        """Sulama başlangıcını logla"""
        self.total_irrigations += 1
        if "ML" in reason or "Otomatik" in reason:
            self.ml_irrigations += 1
        else:
            self.manual_irrigations += 1
        
        self.logger.info(
            f"🚿 SULAMA BAŞLADI | Sebep: {reason} | "
            f"Toprak Nemi: {soil_before:.1f}% | Süre: {duration}s"
        )
    
    def log_end(self, soil_after, water_used):
        """Sulama bitişini logla"""
        self.logger.info(
            f"✅ SULAMA BİTTİ | Son Nem: {soil_after:.1f}% | "
            f"Su Kullanımı: {water_used:.3f}L"
        )
    
    def log_moisture_change(self, before, after, change_percent):
        """Nem değişimini logla"""
        self.logger.info(
            f"📊 NEM DEĞİŞİMİ: {before:.1f}% → {after:.1f}% "
            f"(Δ{change_percent:+.1f}%)"
        )

class MLLogger:
    """Makine öğrenmesi süreçlerini loglar"""
    def __init__(self):
        self.logger = logging.getLogger('smartgreenify.ml')
        handler = RotatingFileHandler(
            Config.ML_LOG, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.training_count = 0
    
    def log_training_start(self, data_points):
        """Eğitim başlangıcını logla"""
        self.training_count += 1
        self.logger.info(
            f"🤖 EĞİTİM BAŞLADI | Eğitim #{self.training_count} | "
            f"Veri Noktası: {data_points}"
        )
    
    def log_training_complete(self, r2_score, mae, rmse, duration):
        """Eğitim tamamlandığında logla"""
        self.logger.info(
            f"✅ EĞİTİM TAMAMLANDI | R²={r2_score:.3f} | "
            f"MAE={mae:.3f} | RMSE={rmse:.3f} | Süre={duration:.3f}s"
        )
    
    def log_prediction(self, predicted_hour, confidence):
        """Tahmin yapıldığında logla"""
        self.logger.info(
            f"💡 TAHMİN: Optimal Sulama Saati = {predicted_hour}:00 | "
            f"Güven: {confidence:.2f}"
        )
    
    def log_error(self, error):
        """ML hatasını logla"""
        self.logger.error(f"❌ ML HATASI: {error}")

class PerformanceLogger:
    """Sistem performansını loglar"""
    def __init__(self):
        self.logger = logging.getLogger('smartgreenify.performance')
        handler = RotatingFileHandler(
            Config.PERFORMANCE_LOG, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.start_time = time.time()
        self.csv_writes = 0
        self.csv_failures = 0
        self.websocket_disconnects = 0
    
    def log_uptime(self):
        """Sistem çalışma süresini logla"""
        uptime_seconds = time.time() - self.start_time
        uptime_hours = uptime_seconds / 3600
        self.logger.info(f"⏱️ UPTIME: {uptime_hours:.2f} saat ({uptime_seconds:.0f} saniye)")
    
    def log_csv_write(self, success=True):
        """CSV yazma işlemini logla"""
        if success:
            self.csv_writes += 1
            self.logger.debug(f"📝 CSV Yazıldı (Toplam: {self.csv_writes})")
        else:
            self.csv_failures += 1
            self.logger.error(f"❌ CSV Yazma Hatası (Toplam Hata: {self.csv_failures})")
    
    def log_websocket_event(self, event_type):
        """WebSocket olayını logla"""
        if event_type == "disconnect":
            self.websocket_disconnects += 1
            self.logger.warning(f"🔌 WebSocket Bağlantı Koptu (Toplam: {self.websocket_disconnects})")
        elif event_type == "reconnect":
            self.logger.info("🔌 WebSocket Yeniden Bağlandı")
    
    def log_summary(self):
        """Performans özetini logla"""
        uptime_hours = (time.time() - self.start_time) / 3600
        self.logger.info(
            f"📊 PERFORMANS ÖZETİ | "
            f"Çalışma: {uptime_hours:.2f}h | "
            f"CSV: {self.csv_writes} yazma, {self.csv_failures} hata | "
            f"WebSocket: {self.websocket_disconnects} kopma"
        )

# Logger örnekleri
sensor_logger = SensorLogger()
irrigation_logger = IrrigationLogger()
ml_logger = MLLogger()
performance_logger = PerformanceLogger()

plant_profiles = {
    "Roka": {"min_moisture":45,"max_temp":26,"icon":"🌱"},
    "Domates":{"min_moisture":50,"max_temp":30,"icon":"🍅"},
    "Fesleğen":{"min_moisture":40,"max_temp":28,"icon":"🌿"},
    "Marul":{"min_moisture":55,"max_temp":22,"icon":"🥬"},
    "Nane":{"min_moisture":60,"max_temp":25,"icon":"🍃"},
    "Genel":{"min_moisture":45,"max_temp":28,"icon":"🪴"}
}

def send_ntfy(title, message, priority="default", tags=""):
    """Ntfy.sh bildirim gönder"""
    if not REQUESTS_AVAILABLE or not Config.NTFY_ENABLED:
        return False
    
    try:
        url = f"https://ntfy.sh/{Config.NTFY_TOPIC}"
        headers = {
            "Title": title,
            "Priority": priority,
            "Tags": tags,
        }
        
        response = requests.post(url, data=message, headers=headers, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ Ntfy OK: {title}")
            return True
        else:
            logger.error(f"❌ Ntfy {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ntfy exception: {e}")
        return False

@contextmanager
def safe_file_write(filepath, mode='w'):
    temp = filepath + '.tmp'
    try:
        with open(temp, mode, newline='', encoding='utf-8') as f:
            yield f
        os.replace(temp, filepath)
        performance_logger.log_csv_write(success=True)
    except Exception as e:
        performance_logger.log_csv_write(success=False)
        if os.path.exists(temp):
            try: os.remove(temp)
            except: pass
        raise

def safe_json_load(filepath, default=None):
    if not os.path.exists(filepath):
        return default or {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return default or {}

def safe_json_save(filepath, data):
    try:
        with safe_file_write(filepath) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False

# ==================== GERÇEK SENSÖRLER ====================

class RealSensor:
    """BME280 (SPI) + ADS1115 (I2C) + LDR (Digital)"""
    
    def __init__(self):
        self.lock = threading.RLock()
        
        # BME280 SPI başlat
        self.spi = spidev.SpiDev()
        self.spi.open(Config.SPI_BUS, Config.SPI_DEVICE)
        self.spi.max_speed_hz = 1000000
        self.spi.mode = 0b00
        
        # I2C başlat (ADS1115)
        self.i2c_bus = SMBus(Config.I2C_BUS)
        
        # GPIO başlat (LDR + Röle)
        self.gpio_chip = lgpio.gpiochip_open(0)
        
        # LDR'yi input olarak ayarla
        try:
            lgpio.gpio_free(self.gpio_chip, Config.LDR_PIN)
        except:
            pass
        lgpio.gpio_claim_input(self.gpio_chip, Config.LDR_PIN)
        
        # BME280 konfigüre et
        self._write_bme_reg(0xF2, 0x01)
        self._write_bme_reg(0xF4, 0x27)
        self._write_bme_reg(0xF5, 0xA0)
        
        # Kalibrasyon verilerini oku
        self.cal = self._read_calibration()
        self.t_fine = 0
        
        logger.info("✅ Gerçek sensörler başlatıldı")
        logger.info(f"   - BME280: SPI {Config.SPI_BUS}.{Config.SPI_DEVICE}")
        logger.info(f"   - ADS1115: I2C 0x{Config.ADS_ADDR:02X}")
        logger.info(f"   - LDR: GPIO {Config.LDR_PIN}")
    
    def _read_bme_reg(self, reg, length):
        with self.lock:
            reg |= 0x80
            result = self.spi.xfer2([reg] + [0x00] * length)
            return result[1:]
    
    def _write_bme_reg(self, reg, value):
        with self.lock:
            reg &= 0x7F
            self.spi.xfer2([reg, value])
    
    def _read_calibration(self):
        calib = {}
        data = self._read_bme_reg(0x88, 24)
        
        def to_signed(val, bits=16):
            if val >= (1 << (bits - 1)):
                return val - (1 << bits)
            return val
        
        calib['T1'] = data[1] << 8 | data[0]
        calib['T2'] = to_signed(data[3] << 8 | data[2])
        calib['T3'] = to_signed(data[5] << 8 | data[4])
        calib['P1'] = data[7] << 8 | data[6]
        calib['P2'] = to_signed(data[9] << 8 | data[8])
        calib['P3'] = to_signed(data[11] << 8 | data[10])
        calib['P4'] = to_signed(data[13] << 8 | data[12])
        calib['P5'] = to_signed(data[15] << 8 | data[14])
        calib['P6'] = to_signed(data[17] << 8 | data[16])
        calib['P7'] = to_signed(data[19] << 8 | data[18])
        calib['P8'] = to_signed(data[21] << 8 | data[20])
        calib['P9'] = to_signed(data[23] << 8 | data[22])
        calib['H1'] = self._read_bme_reg(0xA1, 1)[0]
        
        hdata = self._read_bme_reg(0xE1, 7)
        calib['H2'] = to_signed((hdata[1] << 8) | hdata[0])
        calib['H3'] = hdata[2]
        calib['H4'] = to_signed((hdata[3] << 4) | (hdata[4] & 0x0F), 12)
        calib['H5'] = to_signed((hdata[5] << 4) | (hdata[4] >> 4), 12)
        calib['H6'] = to_signed(hdata[6], 8)
        
        return calib
    
    def _compensate_temperature(self, adc_T):
        var1 = (((adc_T / 16384.0) - (self.cal['T1'] / 1024.0)) * self.cal['T2'])
        var2 = ((((adc_T / 131072.0) - (self.cal['T1'] / 8192.0)) ** 2) * self.cal['T3'])
        self.t_fine = var1 + var2
        return self.t_fine / 5120.0
    
    def _compensate_pressure(self, adc_P):
        var1 = (self.t_fine / 2.0) - 64000.0
        var2 = var1 * var1 * self.cal['P6'] / 32768.0
        var2 = var2 + var1 * self.cal['P5'] * 2.0
        var2 = (var2 / 4.0) + (self.cal['P4'] * 65536.0)
        var1 = (self.cal['P3'] * var1 * var1 / 524288.0 + self.cal['P2'] * var1) / 524288.0
        var1 = (1.0 + var1 / 32768.0) * self.cal['P1']
        
        if var1 == 0:
            return 0
        
        p = 1048576.0 - adc_P
        p = (p - (var2 / 4096.0)) * 6250.0 / var1
        var1 = self.cal['P9'] * p * p / 2147483648.0
        var2 = p * self.cal['P8'] / 32768.0
        p = p + (var1 + var2 + self.cal['P7']) / 16.0
        
        return p / 100.0
    
    def _compensate_humidity(self, adc_H):
        h = self.t_fine - 76800.0
        h = (adc_H - (self.cal['H4'] * 64.0 + self.cal['H5'] / 16384.0 * h)) * \
            (self.cal['H2'] / 65536.0 * (1.0 + self.cal['H6'] / 67108864.0 * h * \
            (1.0 + self.cal['H3'] / 67108864.0 * h)))
        h = h * (1.0 - self.cal['H1'] * h / 524288.0)
        return max(0, min(100, h))
    
    def read_bme(self):
        try:
            with self.lock:
                data = self._read_bme_reg(0xF7, 8)
                
                adc_P = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4)
                adc_T = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4)
                adc_H = (data[6] << 8) | data[7]
                
                temperature = self._compensate_temperature(adc_T)
                pressure = self._compensate_pressure(adc_P)
                humidity = self._compensate_humidity(adc_H)
                
                return round(temperature, 2), round(humidity, 2), round(pressure, 2)
        except Exception as e:
            sensor_logger.log_error("BME280", e)
            return 20.0, 50.0, 1013.0
    
    def read_soil(self):
        try:
            with self.lock:
                config = 0xC383
                self.i2c_bus.write_i2c_block_data(
                    Config.ADS_ADDR, 0x01, 
                    [config >> 8, config & 0xFF]
                )
                time.sleep(0.01)
                
                data = self.i2c_bus.read_i2c_block_data(Config.ADS_ADDR, 0x00, 2)
                value = (data[0] << 8) | data[1]
                
                if value > 32767:
                    value -= 65536
                
                if value >= Config.SOIL_DRY:
                    percent = 0.0
                elif value <= Config.SOIL_WET:
                    percent = 100.0
                else:
                    percent = 100.0 - ((value - Config.SOIL_WET) / (Config.SOIL_DRY - Config.SOIL_WET) * 100.0)
                
                return round(max(0, min(100, percent)), 2)
        except Exception as e:
            sensor_logger.log_error("ADS1115/Soil", e)
            return 50.0
    
    def read_light(self):
        """LDR dijital okuma: True=Gündüz, False=Gece"""
        try:
            value = lgpio.gpio_read(self.gpio_chip, Config.LDR_PIN)
            return value == 0
        except Exception as e:
            sensor_logger.log_error("LDR", e)
            return True
    
    def cleanup(self):
        try:
            self.spi.close()
            self.i2c_bus.close()
            lgpio.gpiochip_close(self.gpio_chip)
            logger.info("✅ Sensörler temizlendi")
        except:
            pass

class Actuator:
    def __init__(self, gpio_chip):
        self.gpio_chip = gpio_chip
        self.irrigation_on = False
        self.lock = threading.RLock()
        self.last_on_time = None
        self.total_irrigation_time = 0
        
        self.set_irrigation(False)
        logger.info(f"✅ Röle hazır (GPIO {Config.RELAY_PIN})")
    
    def set_irrigation(self, on):
        with self.lock:
            try:
                if on and not self.irrigation_on:
                    lgpio.gpio_claim_output(self.gpio_chip, Config.RELAY_PIN)
                    lgpio.gpio_write(self.gpio_chip, Config.RELAY_PIN, 0)
                    self.last_on_time = time.time()
                    logger.info("💦 Pompa AÇIK")
                elif not on and self.irrigation_on:
                    lgpio.gpio_claim_input(self.gpio_chip, Config.RELAY_PIN)
                    if self.last_on_time:
                        self.total_irrigation_time += time.time() - self.last_on_time
                        self.last_on_time = None
                    logger.info("⏹️ Pompa KAPALI")
                
                self.irrigation_on = bool(on)
            except Exception as e:
                logger.error(f"Röle hatası: {e}")
    
    def get_total_time(self):
        with self.lock:
            total = self.total_irrigation_time
            if self.irrigation_on and self.last_on_time:
                total += time.time() - self.last_on_time
            return total
    
    def cleanup(self):
        self.set_irrigation(False)

class IrrigationSchedule:
    def __init__(self):
        self.schedules = safe_json_load(Config.SCHEDULE_FILE, {"time_based": []})
        self.lock = threading.RLock()
    
    def save_schedule(self):
        with self.lock:
            safe_json_save(Config.SCHEDULE_FILE, self.schedules)
    
    def add_time_schedule(self, hour, minute):
        with self.lock:
            self.schedules["time_based"].append({"hour": hour, "minute": minute, "enabled": True})
            self.save_schedule()
    
    def delete_time_schedule(self, index):
        with self.lock:
            try:
                if 0 <= index < len(self.schedules["time_based"]):
                    self.schedules["time_based"].pop(index)
                    self.save_schedule()
                    return True
                return False
            except:
                return False

class Statistics:
    def __init__(self):
        self.stats = safe_json_load(Config.STATS_FILE, {"total_irrigations": 0, "daily_stats": {}})
        self.lock = threading.RLock()
    
    def save_stats(self):
        with self.lock:
            safe_json_save(Config.STATS_FILE, self.stats)
    
    def record_irrigation(self, duration):
        with self.lock:
            self.stats["total_irrigations"] += 1
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.stats["daily_stats"]:
                self.stats["daily_stats"][today] = {"count": 0}
            self.stats["daily_stats"][today]["count"] += 1
            self.save_stats()
    
    def get_daily_summary(self):
        today = datetime.now().strftime("%Y-%m-%d")
        return self.stats["daily_stats"].get(today, {"count": 0})
    
    def get_plant_health_score(self, soil, temp, light, profile):
        score = 100
        if soil < profile["min_moisture"]:
            score -= (profile["min_moisture"] - soil) * 2
        if temp > profile["max_temp"]:
            score -= (temp - profile["max_temp"]) * 3
        return max(0, min(100, int(score)))

class MLOptimizer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.last_train_time = 0
        self.training_history = []
        self.lock = threading.RLock()
        
        if not ML_AVAILABLE:
            logger.warning("ML özellikleri devre dışı")
            return
        
        self.load_model()
    
    def load_model(self):
        if not ML_AVAILABLE:
            return False
        
        model_file = "ml_model.pkl"
        if os.path.exists(model_file):
            try:
                import pickle
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.scaler = data.get('scaler')
                    self.last_train_time = data.get('train_time', 0)
                    self.training_history = data.get('history', [])
                logger.info("✅ ML modeli yüklendi")
                return True
            except Exception as e:
                logger.error(f"ML yükleme hatası: {e}")
        return False
    
    def save_model(self):
        if not ML_AVAILABLE or not self.model:
            return False
        
        try:
            import pickle
            data = {
                'model': self.model,
                'scaler': self.scaler,
                'train_time': self.last_train_time,
                'history': self.training_history[-10:]
            }
            with open("ml_model.pkl", 'wb') as f:
                pickle.dump(data, f)
            logger.info("✅ ML modeli kaydedildi")
            return True
        except Exception as e:
            logger.error(f"ML kaydetme hatası: {e}")
            return False
    
    def train_model(self, csv_file):
        if not ML_AVAILABLE or not os.path.exists(csv_file):
            return False
        
        with self.lock:
            try:
                start_time = time.time()
                
                data = []
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            data.append({
                                'timestamp': row['timestamp'],
                                'soil': float(row['soil']),
                                'temp': float(row['temp']),
                                'hum': float(row['hum']),
                                'light': 1 if row['light'] == 'Gündüz' else 0,
                                'pump': row['pump'] == 'True'
                            })
                        except:
                            continue
                
                if len(data) < 20:
                    ml_logger.logger.info(f"Yetersiz veri: {len(data)}/20")
                    return False
                
                ml_logger.log_training_start(len(data))
                
                X, y = [], []
                for i in range(1, len(data)):
                    prev = data[i-1]
                    curr = data[i]
                    
                    X.append([
                        prev['soil'], prev['temp'], prev['hum'], prev['light'],
                        datetime.fromisoformat(prev['timestamp']).hour,
                        1 if prev['pump'] else 0
                    ])
                    y.append(curr['soil'] - prev['soil'])
                
                X = np.array(X)
                y = np.array(y)
                
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                
                self.model = LinearRegression()
                self.model.fit(X_scaled, y)
                
                # Performans metrikleri
                score = self.model.score(X_scaled, y)
                predictions = self.model.predict(X_scaled)
                mae = np.mean(np.abs(y - predictions))
                rmse = np.sqrt(np.mean((y - predictions)**2))
                
                duration = time.time() - start_time
                self.last_train_time = time.time()
                
                self.training_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'score': score,
                    'mae': mae,
                    'rmse': rmse,
                    'data_points': len(X),
                    'duration': duration
                })
                
                self.save_model()
                ml_logger.log_training_complete(score, mae, rmse, duration)
                
                return True
            except Exception as e:
                ml_logger.log_error(str(e))
                return False
    
    def predict_optimal_irrigation_time(self, soil, temp, hum, light_daytime):
        if not ML_AVAILABLE or not self.model:
            return None
        
        with self.lock:
            try:
                best_hour = None
                best_score = float('inf')
                
                for hour in range(24):
                    features = np.array([[soil, temp, hum, 1 if light_daytime else 0, hour, 0]])
                    features_scaled = self.scaler.transform(features)
                    predicted_change = self.model.predict(features_scaled)[0]
                    
                    score = abs(predicted_change) + (0.5 if 6 <= hour <= 18 else 0)
                    
                    if score < best_score:
                        best_score = score
                        best_hour = hour
                
                if best_hour is not None:
                    ml_logger.log_prediction(best_hour, 1.0 - (best_score / 10.0))
                
                return best_hour
            except Exception as e:
                ml_logger.log_error(f"Tahmin hatası: {e}")
                return None
    
    def get_model_stats(self):
        with self.lock:
            if not self.training_history:
                return None
            
            last = self.training_history[-1]
            return {
                'score': round(last['score'], 3),
                'mae': round(last.get('mae', 0), 3),
                'rmse': round(last.get('rmse', 0), 3),
                'data_points': last['data_points'],
                'total_trainings': len(self.training_history)
            }

class SmartController:
    def __init__(self, sensor, actuator):
        self.sensor = sensor
        self.actuator = actuator
        self.selected_plant = Config.DEFAULT_PLANT
        self.lock = threading.RLock()
        self.schedule = IrrigationSchedule()
        self.statistics = Statistics()
        self.manual_irrigation_end = None
        self.last_auto_irrigation = 0
        self.ml_suggested_hour = None
    
    def set_plant(self, name):
        with self.lock:
            if name in plant_profiles:
                self.selected_plant = name
                logger.info(f"✅ Bitki değiştirildi: {name}")
                return True
            return False
    
    def start_manual_irrigation(self, duration, soil_before):
        with self.lock:
            self.manual_irrigation_end = time.time() + duration
            self.actuator.set_irrigation(True)
            irrigation_logger.log_start("Manuel", soil_before, duration)
            logger.info(f"💦 Manuel sulama başladı: {duration}s")
    
    def start_auto_irrigation(self, reason, soil_before):
        with self.lock:
            if time.time() - self.last_auto_irrigation < Config.AUTO_IRRIGATION_MIN_INTERVAL:
                remaining = int((Config.AUTO_IRRIGATION_MIN_INTERVAL - (time.time() - self.last_auto_irrigation)) / 60)
                logger.info(f"⏳ Çok erken - {remaining} dakika bekle")
                return False
            
            self.manual_irrigation_end = time.time() + Config.AUTO_IRRIGATION_DURATION
            self.actuator.set_irrigation(True)
            self.last_auto_irrigation = time.time()
            irrigation_logger.log_start(reason, soil_before, Config.AUTO_IRRIGATION_DURATION)
            logger.info(f"🤖 Otomatik sulama başladı: {reason}")
            return True
    
    def cleanup(self):
        with self.lock:
            self.manual_irrigation_end = None
            self.actuator.cleanup()

# Flask App
app = Flask(__name__, static_folder=Config.STATIC_FOLDER)
app.config['SECRET_KEY'] = 'sg2025'

if SOCKETIO_AVAILABLE:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
else:
    socketio = None

sensor = RealSensor()
actuator = Actuator(sensor.gpio_chip)
controller = SmartController(sensor, actuator)
ml_optimizer = MLOptimizer()

time_history = []
soil_history = []
temp_history = []
hum_history = []
pressure_history = []
light_history = []
pump_history = []
data_lock = threading.RLock()

def initialize_csv():
    if os.path.exists(Config.CSV_FILE):
        logger.info(f"✅ Mevcut CSV bulundu: {Config.CSV_FILE}")
        return
    
    try:
        with safe_file_write(Config.CSV_FILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp","plant","soil","temp","hum","pressure","light","pump"])
        logger.info(f"✅ Yeni CSV oluşturuldu")
    except Exception as e:
        logger.error(f"❌ CSV oluşturma hatası: {e}")

initialize_csv()
shutdown_event = threading.Event()

def background_loop():
    last_ml_train = 0
    ml_train_interval = 604800  # 7 gün
    last_ml_check_hour = -1
    last_csv_save = 0
    csv_save_interval = 300  # 5 dakika
    last_low_soil_alert = 0
    low_soil_alert_interval = 3600  # 1 saat
    last_performance_log = 0
    performance_log_interval = 3600  # 1 saat
    
    soil_before_irrigation = None
    
    logger.info(f"🔄 Background loop başladı - Sensör: {Config.UPDATE_INTERVAL}s, CSV: 5dk")
    
    while not shutdown_event.is_set():
        try:
            soil = sensor.read_soil()
            temp, hum, pressure = sensor.read_bme()
            light_daytime = sensor.read_light()
            current_hour = datetime.now().hour
            current_time = time.time()
            
            # Sensör okumalarını logla
            sensor_logger.log_reading(soil, temp, hum, pressure, light_daytime, actuator.irrigation_on)

            with data_lock:
                label = datetime.now().strftime("%H:%M")
                time_history.append(label)
                soil_history.append(round(soil, 2))
                temp_history.append(round(temp, 2))
                hum_history.append(round(hum, 2))
                pressure_history.append(round(pressure, 2))
                light_history.append(light_daytime)
                pump_history.append(bool(actuator.irrigation_on))
                
                if len(time_history) > Config.MAX_HISTORY:
                    for lst in [time_history, soil_history, temp_history, hum_history, 
                               pressure_history, light_history, pump_history]:
                        lst.pop(0)
            
            # ML bazlı otomatik sulama
            if Config.AUTO_IRRIGATION_ENABLED and ML_AVAILABLE and ml_optimizer.model:
                optimal_hour = ml_optimizer.predict_optimal_irrigation_time(soil, temp, hum, light_daytime)
                
                if optimal_hour is not None:
                    controller.ml_suggested_hour = optimal_hour
                    
                    if current_hour == optimal_hour and last_ml_check_hour != current_hour:
                        profile = plant_profiles[controller.selected_plant]
                        
                        if soil < profile["min_moisture"] and not actuator.irrigation_on:
                            if controller.start_auto_irrigation(f"ML Tavsiye (Saat: {optimal_hour}:00)", soil):
                                soil_before_irrigation = soil
                                send_ntfy(
                                    "Otomatik Sulama Basladi",
                                    f"ML tavsiyesi: {optimal_hour}:00 - Toprak: {soil:.1f}% (Min: {profile['min_moisture']}%)",
                                    "default",
                                    "droplet,robot"
                                )
                        
                        last_ml_check_hour = current_hour
            
            # Düşük toprak nemi acil uyarısı
            profile = plant_profiles[controller.selected_plant]
            if soil < profile["min_moisture"] - 5 and (current_time - last_low_soil_alert) > low_soil_alert_interval:
                send_ntfy(
                    "ACIL: Cok Dusuk Nem!",
                    f"{controller.selected_plant} toprak nemi cok dusuk: {soil:.1f}% (Min: {profile['min_moisture']}%)",
                    "urgent",
                    "warning,droplet,sos"
                )
                last_low_soil_alert = current_time
            
            # Manuel sulama kontrolü
            if controller.manual_irrigation_end:
                if current_time >= controller.manual_irrigation_end:
                    controller.manual_irrigation_end = None
                    actuator.set_irrigation(False)
                    
                    # Sulama sonrası nem değişimini logla
                    if soil_before_irrigation is not None:
                        soil_after = soil
                        change = soil_after - soil_before_irrigation
                        irrigation_logger.log_moisture_change(soil_before_irrigation, soil_after, change)
                        
                        # Su kullanımı hesapla (yaklaşık)
                        water_used = (Config.AUTO_IRRIGATION_DURATION / 3600) * 0.1  # L cinsinden
                        irrigation_logger.log_end(soil_after, water_used)
                        soil_before_irrigation = None
                    
                    controller.statistics.record_irrigation(actuator.get_total_time())
                    send_ntfy(
                        "Sulama Tamamlandi",
                        f"Sulama suresinin sonuna ulasti",
                        "default",
                        "droplet,white_check_mark"
                    )
            
            # CSV'ye kaydet
            if (current_time - last_csv_save) >= csv_save_interval:
                try:
                    with open(Config.CSV_FILE, "a", newline="", encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            controller.selected_plant,
                            soil, temp, hum, pressure,
                            "Gündüz" if light_daytime else "Gece",
                            actuator.irrigation_on
                        ])
                    logger.debug(f"📝 CSV: {soil:.1f}% nem")
                    last_csv_save = current_time
                except Exception as e:
                    logger.error(f"❌ CSV hatası: {e}")
            
            # ML eğitimi
            if ML_AVAILABLE and (current_time - last_ml_train) > ml_train_interval:
                logger.info("🤖 ML eğitiliyor...")
                if ml_optimizer.train_model(Config.CSV_FILE):
                    last_ml_train = current_time
                    send_ntfy("ML Model Egitildi", "Yeni tahminler hazir!", "default", "brain")
            
            # Performans logları
            if (current_time - last_performance_log) >= performance_log_interval:
                performance_logger.log_uptime()
                performance_logger.log_summary()
                logger.info(f"📊 Sensör başarı oranı: {sensor_logger.get_success_rate():.2f}%")
                logger.info(f"💧 Toplam sulama: {irrigation_logger.total_irrigations} (ML: {irrigation_logger.ml_irrigations}, Manuel: {irrigation_logger.manual_irrigations})")
                last_performance_log = current_time
            
            # WebSocket broadcast
            if socketio:
                socketio.emit('sensor_update', {
                    'soil': soil, 'temp': temp, 'hum': hum, 'pressure': pressure,
                    'light_daytime': light_daytime,
                    'pump_on': actuator.irrigation_on,
                    'ml_suggested_hour': controller.ml_suggested_hour,
                    'history': {
                        'times': list(time_history[-20:]),
                        'soil': list(soil_history[-20:]),
                        'temp': list(temp_history[-20:]),
                        'hum': list(hum_history[-20:]),
                        'pressure': list(pressure_history[-20:])
                    }
                })
        
        except Exception as e:
            logger.error(f"❌ Loop error: {e}", exc_info=True)
        
        shutdown_event.wait(Config.UPDATE_INTERVAL)
    
    # Kapatma öncesi son performans özeti
    performance_logger.log_summary()
    logger.info("🛑 Background loop sonlandı")

bg_thread = threading.Thread(target=background_loop, daemon=True)
bg_thread.start()

def cleanup_resources():
    logger.info("🧹 Temizlik başlıyor...")
    shutdown_event.set()
    try:
        controller.cleanup()
        controller.statistics.save_stats()
        sensor.cleanup()
        
        # Son log özeti
        logger.info("="*70)
        logger.info("📊 SİSTEM KAPANIŞ ÖZETİ")
        logger.info("="*70)
        logger.info(f"Toplam sensör okuması: {sensor_logger.total_reads}")
        logger.info(f"Başarı oranı: {sensor_logger.get_success_rate():.2f}%")
        logger.info(f"Başarısız okuma: {sensor_logger.failed_reads}")
        logger.info(f"Toplam sulama: {irrigation_logger.total_irrigations}")
        logger.info(f"  - ML bazlı: {irrigation_logger.ml_irrigations}")
        logger.info(f"  - Manuel: {irrigation_logger.manual_irrigations}")
        logger.info(f"CSV yazma: {performance_logger.csv_writes} başarılı, {performance_logger.csv_failures} hata")
        logger.info(f"WebSocket kopma: {performance_logger.websocket_disconnects}")
        
        uptime_hours = (time.time() - performance_logger.start_time) / 3600
        logger.info(f"Toplam çalışma süresi: {uptime_hours:.2f} saat")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Temizlik hatası: {e}")
    
    if bg_thread.is_alive():
        bg_thread.join(timeout=5)
    
    logger.info("✅ Sistem temizlendi ve kapatıldı")

atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, lambda s,f: (cleanup_resources(), sys.exit(0)))

# PWA - Manifest & Service Worker
MANIFEST_JSON = {
    "name": "SmartGreenify",
    "short_name": "SmartGreenify",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#FAF8F3",
    "theme_color": "#27AE60",
    "icons": [{"src": "/static/icon.png", "sizes": "192x192", "type": "image/png"}]
}

SERVICE_WORKER = "const CACHE='sg-v1';self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(['/']))));self.addEventListener('fetch',e=>e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request))));"

# HTML Template (SmartGreen -> SmartGreenify)
INDEX_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="theme-color" content="#27AE60">
<title>SmartGreenify v4.0</title>
<link rel="manifest" href="/manifest.json">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
{% if socketio_available %}<script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>{% endif %}
<style>
*{margin:0;padding:0;box-sizing:border-box}:root{--bg:#FAF8F3;--card:#FFF;--text:#2C3E50;--text2:#7F8C8D;--green:#27AE60;--blue:#3498DB;--orange:#F39C12;--red:#E74C3C;--border:rgba(149,165,166,0.15)}[data-theme="dark"]{--bg:#1a1a1a;--card:#2d2d2d;--text:#ECEFF1;--text2:#B0BEC5;--border:rgba(255,255,255,0.1)}body{font-family:Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.6;transition:all 0.3s}.container{max-width:1400px;margin:0 auto;padding:20px}.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:30px;flex-wrap:wrap;gap:15px}h1{font-size:clamp(24px,5vw,36px);font-weight:700;background:linear-gradient(120deg,var(--green),#52C88A,var(--blue));-webkit-background-clip:text;-webkit-text-fill-color:transparent}.theme-toggle{background:var(--card);border:2px solid var(--border);border-radius:50px;padding:8px 16px;cursor:pointer;font-size:20px}.card{background:var(--card);border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.08);border:1px solid var(--border)}.card:hover{box-shadow:0 4px 16px rgba(0,0,0,0.12);transform:translateY(-2px);transition:all 0.3s}.card-title{font-size:20px;font-weight:600;display:flex;align-items:center;gap:10px;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px}.stat-card{background:var(--card);border-radius:12px;padding:20px;text-align:center;border:1px solid var(--border);position:relative;overflow:hidden}.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,var(--green),var(--blue))}.stat-value{font-size:32px;font-weight:700;color:var(--green);margin-bottom:8px}.stat-label{font-size:13px;color:var(--text2);font-weight:500}.btn{background:linear-gradient(135deg,var(--green),#52C88A);color:#FFF;border:none;padding:12px 24px;border-radius:12px;cursor:pointer;font-size:14px;font-weight:500;margin:5px;text-decoration:none;display:inline-block;text-align:center}.btn:hover{transform:translateY(-2px);box-shadow:0 4px 12px rgba(39,174,96,0.3)}.btn-danger{background:linear-gradient(135deg,var(--red),#C0392B)}.status-badge{display:inline-flex;align-items:center;gap:8px;padding:8px 16px;border-radius:20px;font-size:13px;font-weight:500}.status-on{background:rgba(39,174,96,0.2);color:var(--green)}.status-off{background:rgba(149,165,166,0.1);color:var(--text2)}.chart-container{position:relative;height:300px;margin-top:20px}.live-dot{width:8px;height:8px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}input,select{padding:10px;border:1px solid var(--border);border-radius:8px;background:var(--card);color:var(--text)}.schedule-item{display:flex;justify-content:space-between;align-items:center;padding:12px;background:var(--bg);border-radius:8px;margin-bottom:8px}@media(max-width:768px){.container{padding:15px}.card{padding:16px}.chart-container{height:250px}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<div><h1>🌱 SmartGreenify v4.0</h1><div style="display:flex;align-items:center;gap:8px;margin-top:8px"><span class="live-dot"></span><span style="font-size:13px;color:var(--text2)">Gerçek Sensörler + ML Logları</span></div></div>
<button class="theme-toggle" onclick="toggleTheme()"><span id="theme-icon">🌙</span></button>
</div>
<div class="card">
<div class="card-title">🪴 Bitki</div>
<form method="POST" action="/select_plant" style="display:flex;gap:10px">
<select name="plant" style="flex:1">{% for p in plants %}<option value="{{p}}"{% if p==current %}selected{% endif %}>{{plant_profiles[p].icon}} {{p}}</option>{% endfor %}</select>
<button class="btn" type="submit">Değiştir</button>
</form>
</div>
<div class="card">
<div class="card-title">📊 Sensörler</div>
<div class="grid">
<div class="stat-card"><div class="stat-value" id="soil_value">--</div><div class="stat-label">Toprak Nemi (%)</div></div>
<div class="stat-card"><div class="stat-value" id="temp_value">--</div><div class="stat-label">Sıcaklık (°C)</div></div>
<div class="stat-card"><div class="stat-value" id="hum_value">--</div><div class="stat-label">Hava Nemi (%)</div></div>
<div class="stat-card"><div class="stat-value" id="pressure_value">--</div><div class="stat-label">Basınç (hPa)</div></div>
<div class="stat-card"><div class="stat-value" id="light_value">--</div><div class="stat-label">Işık</div></div>
<div class="stat-card"><span class="status-badge status-off" id="pump_badge">⚫ Kapalı</span><div class="stat-label" style="margin-top:10px">Pompa</div></div>
</div>
</div>
<div class="card">
<div class="card-title">📈 Toprak Nemi</div>
<div class="chart-container"><canvas id="soilChart"></canvas></div>
</div>
<div class="grid">
<div class="card"><div class="card-title">🌡️ Sıcaklık</div><div class="chart-container"><canvas id="tempChart"></canvas></div></div>
<div class="card"><div class="card-title">💨 Nem</div><div class="chart-container"><canvas id="humChart"></canvas></div></div>
</div>
<div class="card">
<div class="card-title">🌍 Basınç</div>
<div class="chart-container"><canvas id="pressureChart"></canvas></div>
</div>
<div class="card">
<div class="card-title">💦 Pompa</div>
<div style="display:flex;gap:10px;margin-bottom:15px">
<select id="manual_duration" style="width:150px"><option value="30">30s</option><option value="60" selected>1dk</option><option value="120">2dk</option><option value="300">5dk</option></select>
<button class="btn" onclick="startWater()">▶️ Başlat</button>
<button class="btn btn-danger" onclick="stopWater()">⏹️ Durdur</button>
</div>
</div>
<div class="card">
<div class="card-title">⏰ Program</div>
<div style="display:flex;gap:10px;margin-bottom:15px">
<input type="number" id="schedule_hour" min="0" max="23" value="7" style="width:80px" placeholder="Saat">
<input type="number" id="schedule_minute" min="0" max="59" value="0" style="width:80px" placeholder="Dk">
<button class="btn" onclick="addSchedule()">➕ Ekle</button>
</div>
<div id="schedule_list"></div>
</div>
<div class="card">
<div class="card-title">📊 İstatistikler</div>
<div class="grid">
<div class="stat-card"><div class="stat-value" id="stat_today">0</div><div class="stat-label">Bugün</div></div>
<div class="stat-card"><div class="stat-value" id="stat_total">0</div><div class="stat-label">Toplam</div></div>
<div class="stat-card"><div class="stat-value" id="health_score">--</div><div class="stat-label">Sağlık</div></div>
</div>
</div>
<div class="card">
<div class="card-title">🤖 Makine Öğrenmesi</div>
<div id="ml_status" style="padding:15px;background:var(--bg);border-radius:8px;margin-bottom:10px"></div>
<div id="ml_prediction" style="font-size:14px;color:var(--text2)"></div>
</div>
<div class="card" style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white">
<div class="card-title" style="color:white">📊 Analytics & Raporlar</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:15px">
<a href="/analytics/dashboard" target="_blank" class="btn" style="background:rgba(255,255,255,0.2);backdrop-filter:blur(10px)">📈 Dashboard</a>
<a href="/analytics/heatmap" target="_blank" class="btn" style="background:rgba(255,255,255,0.2);backdrop-filter:blur(10px)">🔥 Heatmap</a>
<a href="/analytics/correlation" target="_blank" class="btn" style="background:rgba(255,255,255,0.2);backdrop-filter:blur(10px)">🔗 Korelasyon</a>
<button class="btn" onclick="exportPDF()" style="background:rgba(255,255,255,0.2);backdrop-filter:blur(10px)">📄 PDF Rapor</button>
<button class="btn" onclick="exportExcel()" style="background:rgba(255,255,255,0.2);backdrop-filter:blur(10px)">📊 Excel Rapor</button>
</div>
<div id="export_status" style="margin-top:15px;padding:10px;background:rgba(255,255,255,0.1);border-radius:8px;font-size:13px;min-height:40px"></div>
</div>
</div>
<script>
const charts={};function toggleTheme(){const h=document.documentElement;const c=h.getAttribute('data-theme');const n=c==='dark'?'light':'dark';h.setAttribute('data-theme',n);localStorage.setItem('theme',n);document.getElementById('theme-icon').textContent=n==='dark'?'☀️':'🌙'}const s=localStorage.getItem('theme')||'light';document.documentElement.setAttribute('data-theme',s);if(s==='dark')document.getElementById('theme-icon').textContent='☀️';function createChart(id,label,color){const ctx=document.getElementById(id).getContext('2d');charts[id]=new Chart(ctx,{type:'line',data:{labels:[],datasets:[{label:label,data:[],borderColor:color,backgroundColor:color+'20',tension:0.4,fill:true,borderWidth:2}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{beginAtZero:false,grid:{color:'rgba(149,165,166,0.1)'}},x:{grid:{display:false},ticks:{maxTicksLimit:6}}}}})}function updateChart(id,labels,data){if(!charts[id])return;charts[id].data.labels=labels.slice(-20);charts[id].data.datasets[0].data=data.slice(-20);charts[id].update('none')}createChart('soilChart','Toprak','#27AE60');createChart('tempChart','Sıcaklık','#F39C12');createChart('humChart','Nem','#3498DB');createChart('pressureChart','Basınç','#9B59B6');{% if socketio_available %}const socket=io();socket.on('connect',()=>console.log('WebSocket'));socket.on('sensor_update',d=>updateUI(d));{% endif %}function startWater(){const dur=document.getElementById('manual_duration').value;fetch('/manual_irrigation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'start',duration:parseInt(dur)})}).then(r=>r.json()).then(d=>alert(d.message))}function stopWater(){fetch('/manual_irrigation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'stop'})}).then(r=>r.json()).then(d=>alert(d.message))}function addSchedule(){const h=document.getElementById('schedule_hour').value;const m=document.getElementById('schedule_minute').value;fetch('/add_schedule',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({hour:parseInt(h),minute:parseInt(m)})}).then(r=>r.json()).then(d=>{alert(d.message);update()})}function deleteSchedule(idx){if(confirm('Silmek istediğinize emin misiniz?')){fetch('/delete_schedule',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:idx})}).then(r=>r.json()).then(d=>{alert(d.message);update()})}}function exportPDF(){const status=document.getElementById('export_status');status.textContent='⏳ PDF oluşturuluyor...';fetch('/analytics/export_pdf').then(r=>r.json()).then(d=>{if(d.success){status.innerHTML=`✅ ${d.message}<br><a href="${d.download_url}" download style="color:white;text-decoration:underline;font-weight:bold">📥 İndir</a>`}else{status.textContent='❌ Hata: '+d.error}}).catch(e=>{status.textContent='❌ Hata: '+e})}function exportExcel(){const status=document.getElementById('export_status');status.textContent='⏳ Excel oluşturuluyor...';fetch('/analytics/export_excel').then(r=>r.json()).then(d=>{if(d.success){status.innerHTML=`✅ ${d.message}<br><a href="${d.download_url}" download style="color:white;text-decoration:underline;font-weight:bold">📥 İndir</a>`}else{status.textContent='❌ Hata: '+d.error}}).catch(e=>{status.textContent='❌ Hata: '+e})}async function update(){try{const r=await fetch('/data');const d=await r.json();updateUI(d)}catch(e){console.error(e)}}function updateUI(d){if(!d)return;document.getElementById('soil_value').textContent=(d.soil||0).toFixed(1);document.getElementById('temp_value').textContent=(d.temp||0).toFixed(1);document.getElementById('hum_value').textContent=(d.hum||0).toFixed(1);document.getElementById('pressure_value').textContent=(d.pressure||0).toFixed(0);document.getElementById('light_value').textContent=d.light_daytime?'☀️ Gündüz':'🌙 Gece';const badge=document.getElementById('pump_badge');if(d.pump_on){badge.className='status-badge status-on';badge.textContent='⚫ Açık'}else{badge.className='status-badge status-off';badge.textContent='⚫ Kapalı'}if(d.stats){document.getElementById('stat_today').textContent=d.stats.daily.count;document.getElementById('stat_total').textContent=d.stats.total_irrigations}if(d.health_score!==undefined){document.getElementById('health_score').textContent=d.health_score}if(d.ml_enabled){let mlHtml='<strong>✅ ML Aktif</strong><br>';if(d.ml_stats){mlHtml+=`<small>Skor: ${d.ml_stats.score} | MAE: ${d.ml_stats.mae} | RMSE: ${d.ml_stats.rmse} | Veri: ${d.ml_stats.data_points} | Eğitim: ${d.ml_stats.total_trainings}</small>`}else{mlHtml+='<small>Model henüz eğitilmedi (20+ veri gerekli)</small>'}document.getElementById('ml_status').innerHTML=mlHtml;if(d.ml_prediction){document.getElementById('ml_prediction').innerHTML=`<strong>💡 Optimal Sulama:</strong> ${d.ml_prediction}:00`}else{document.getElementById('ml_prediction').innerHTML='Tahmin için daha fazla veri bekleniyor...'}if(d.ml_suggested_hour!==null){document.getElementById('ml_prediction').innerHTML+=`<br><strong>🤖 Sistem Önerisi:</strong> ${d.ml_suggested_hour}:00 (Otomatik: ${d.auto_irrigation_enabled?'Açık':'Kapalı'})`}}else{document.getElementById('ml_status').innerHTML='<strong>⚠️ ML Kapalı</strong><br><small>scikit-learn kurun: pip install scikit-learn numpy</small>';document.getElementById('ml_prediction').innerHTML=''}if(d.schedules){let html='';d.schedules.forEach((s,i)=>{html+=`<div class="schedule-item"><div><strong>${s.hour.toString().padStart(2,'0')}:${s.minute.toString().padStart(2,'0')}</strong></div><button class="btn btn-danger" style="padding:6px 12px;font-size:12px" onclick="deleteSchedule(${i})">🗑️</button></div>`});document.getElementById('schedule_list').innerHTML=html||'<p style="color:var(--text2)">Program yok</p>'}if(d.history){updateChart('soilChart',d.history.times,d.history.soil);updateChart('tempChart',d.history.times,d.history.temp);updateChart('humChart',d.history.times,d.history.hum);updateChart('pressureChart',d.history.times,d.history.pressure)}}if('serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js')}update();{% if not socketio_available %}setInterval(update,5000);{% endif %}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, 
        plants=list(plant_profiles.keys()),
        current=controller.selected_plant,
        plant_profiles=plant_profiles,
        socketio_available=SOCKETIO_AVAILABLE)

@app.route("/manifest.json")
def manifest():
    return jsonify(MANIFEST_JSON)

@app.route("/sw.js")
def service_worker():
    return SERVICE_WORKER, 200, {'Content-Type': 'application/javascript'}

@app.route("/select_plant", methods=["POST"])
def select_plant():
    try:
        plant_name = request.form.get("plant", Config.DEFAULT_PLANT)
        if controller.set_plant(plant_name):
            send_ntfy("Bitki Degisti", f"Yeni bitki: {plant_name} {plant_profiles[plant_name]['icon']}", "default", "seedling")
            return redirect(url_for('index'))
        else:
            return f"Hata: Bilinmeyen bitki '{plant_name}'", 400
    except Exception as e:
        logger.error(f"Bitki değiştirme hatası: {e}")
        return f"Hata: {e}", 500

@app.route("/data")
def data():
    try:
        with data_lock:
            profile = plant_profiles[controller.selected_plant]
            health_score = controller.statistics.get_plant_health_score(
                soil_history[-1] if soil_history else 50,
                temp_history[-1] if temp_history else 20,
                light_history[-1] if light_history else True,
                profile
            )
            
            ml_prediction = None
            ml_stats = None
            if ML_AVAILABLE and ml_optimizer.model:
                ml_prediction = ml_optimizer.predict_optimal_irrigation_time(
                    soil_history[-1] if soil_history else 50,
                    temp_history[-1] if temp_history else 20,
                    hum_history[-1] if hum_history else 50,
                    light_history[-1] if light_history else True
                )
                ml_stats = ml_optimizer.get_model_stats()
            
            return jsonify({
                "soil": soil_history[-1] if soil_history else 0,
                "temp": temp_history[-1] if temp_history else 0,
                "hum": hum_history[-1] if hum_history else 0,
                "pressure": pressure_history[-1] if pressure_history else 0,
                "light_daytime": light_history[-1] if light_history else True,
                "pump_on": actuator.irrigation_on,
                "health_score": health_score,
                "stats": {
                    "daily": controller.statistics.get_daily_summary(),
                    "total_irrigations": controller.statistics.stats["total_irrigations"]
                },
                "schedules": controller.schedule.schedules.get("time_based", []),
                "ml_enabled": ML_AVAILABLE,
                "ml_prediction": ml_prediction,
                "ml_stats": ml_stats,
                "ml_suggested_hour": controller.ml_suggested_hour,
                "auto_irrigation_enabled": Config.AUTO_IRRIGATION_ENABLED,
                "history": {
                    "times": list(time_history),
                    "soil": list(soil_history),
                    "temp": list(temp_history),
                    "hum": list(hum_history),
                    "pressure": list(pressure_history)
                }
            })
    except Exception as e:
        logger.error(f"Data error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/manual_irrigation", methods=["POST"])
def manual_irrigation():
    try:
        data = request.get_json()
        if data.get('action') == 'start':
            duration = data.get('duration', 60)
            soil_before = soil_history[-1] if soil_history else 50
            controller.start_manual_irrigation(duration, soil_before)
            send_ntfy("Sulama Baslatildi", f"Manuel sulama basladi - Sure: {duration}s", "default", "droplet")
            return jsonify({"success": True, "message": "✅ Sulama başlatıldı"})
        else:
            controller.manual_irrigation_end = None
            actuator.set_irrigation(False)
            send_ntfy("Sulama Durduruldu", "Manuel durdurma", "default", "droplet")
            return jsonify({"success": True, "message": "⏹️ Sulama durduruldu"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/add_schedule", methods=["POST"])
def add_schedule():
    try:
        data = request.get_json()
        controller.schedule.add_time_schedule(data.get('hour', 7), data.get('minute', 0))
        return jsonify({"success": True, "message": "✅ Program eklendi"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/delete_schedule", methods=["POST"])
def delete_schedule():
    try:
        data = request.get_json()
        if controller.schedule.delete_time_schedule(data.get('index', -1)):
            return jsonify({"success": True, "message": "✅ Program silindi"})
        return jsonify({"success": False, "message": "❌ Silinemedi"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==================== ANALYTICS ROUTES ====================

@app.route("/analytics/dashboard")
def analytics_dashboard():
    if not PLOTLY_AVAILABLE:
        return "<h1 style='color:red;'>⚠️ Plotly/Pandas yok</h1><p>Kurulum: <code>pip install plotly pandas</code></p>", 500
    
    try:
        if not os.path.exists(Config.CSV_FILE):
            return "<h1 style='color:red;'>❌ CSV dosyası yok</h1>", 404
        
        df = pd.read_csv(Config.CSV_FILE)
        
        if len(df) < 2:
            return "<h1 style='color:orange;'>⚠️ Yetersiz veri</h1>", 400
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.tail(100)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('🌱 Toprak Nemi', '🌡️ Sıcaklık & Nem', '🌍 Basınç', '💧 Sulama Geçmişi'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['soil'], name='Toprak Nemi',
                      line=dict(color='#27AE60', width=2),
                      fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.2)'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['temp'], name='Sıcaklık',
                      line=dict(color='#F39C12', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['hum'], name='Nem',
                      line=dict(color='#3498DB', width=2)),
            row=1, col=2, secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['pressure'], name='Basınç',
                      line=dict(color='#9B59B6', width=2),
                      fill='tozeroy', fillcolor='rgba(155, 89, 182, 0.2)'),
            row=2, col=1
        )
        
        df['pump_numeric'] = df['pump'].map({'True': 1, True: 1, 'False': 0, False: 0})
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['pump_numeric'], name='Pompa',
                   marker_color='#3498DB'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template='plotly_white',
            title_text="📊 SmartGreenify Analytics Dashboard",
            title_font_size=24
        )
        
        fig.update_yaxes(title_text="Nem (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sıcaklık (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Nem (%)", secondary_y=True, row=1, col=2)
        fig.update_yaxes(title_text="hPa", row=2, col=1)
        fig.update_yaxes(title_text="Durum", row=2, col=2)
        
        html = fig.to_html(full_html=True, include_plotlyjs='cdn')
        return html
        
    except Exception as e:
        logger.error(f"Dashboard hatası: {e}", exc_info=True)
        return f"<h1 style='color:red;'>❌ Hata: {str(e)}</h1>", 500

@app.route("/analytics/heatmap")
def analytics_heatmap():
    if not PLOTLY_AVAILABLE:
        return "<h1 style='color:red;'>⚠️ Plotly/Pandas yok</h1>", 500
    
    try:
        if not os.path.exists(Config.CSV_FILE):
            return "<h1 style='color:red;'>❌ CSV dosyası yok</h1>", 404
        
        df = pd.read_csv(Config.CSV_FILE)
        
        if len(df) < 2:
            return "<h1 style='color:orange;'>⚠️ Yetersiz veri</h1>", 400
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        df['pump_numeric'] = df['pump'].map({'True': 1, True: 1, 'False': 0, False: 0})
        
        heatmap_data = df[df['pump_numeric'] == 1].groupby(['date', 'hour']).size().reset_index(name='count')
        
        if len(heatmap_data) == 0:
            return "<h1 style='color:orange;'>⚠️ Sulama verisi yok</h1>", 400
        
        pivot = heatmap_data.pivot(index='date', columns='hour', values='count').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{h}:00" for h in pivot.columns],
            y=[str(d) for d in pivot.index],
            colorscale='Viridis',
            hoverongaps=False,
            colorbar=dict(title="Sulama Sayısı")
        ))
        
        fig.update_layout(
            title='🔥 Sulama Heatmap - Saatlik Dağılım',
            xaxis_title='Saat',
            yaxis_title='Tarih',
            height=600,
            template='plotly_white'
        )
        
        return fig.to_html(full_html=True, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Heatmap hatası: {e}", exc_info=True)
        return f"<h1 style='color:red;'>❌ Hata: {str(e)}</h1>", 500

@app.route("/analytics/correlation")
def analytics_correlation():
    if not PLOTLY_AVAILABLE:
        return "<h1 style='color:red;'>⚠️ Plotly/Pandas yok</h1>", 500
    
    try:
        if not os.path.exists(Config.CSV_FILE):
            return "<h1 style='color:red;'>❌ CSV dosyası yok</h1>", 404
        
        df = pd.read_csv(Config.CSV_FILE)
        
        if len(df) < 2:
            return "<h1 style='color:orange;'>⚠️ Yetersiz veri</h1>", 400
        
        df['pump_numeric'] = df['pump'].map({'True': 1, True: 1, 'False': 0, False: 0})
        df['light_numeric'] = df['light'].map({'Gündüz': 1, 'Gece': 0})
        
        numeric_cols = ['soil', 'temp', 'hum', 'pressure', 'light_numeric', 'pump_numeric']
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=['Toprak', 'Sıcaklık', 'Nem', 'Basınç', 'Işık', 'Pompa'],
            y=['Toprak', 'Sıcaklık', 'Nem', 'Basınç', 'Işık', 'Pompa'],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 14},
            colorbar=dict(title="Korelasyon")
        ))
        
        fig.update_layout(
            title='🔗 Sensör Korelasyon Matrisi',
            height=600,
            template='plotly_white'
        )
        
        return fig.to_html(full_html=True, include_plotlyjs='cdn')
        
    except Exception as e:
        logger.error(f"Korelasyon hatası: {e}", exc_info=True)
        return f"<h1 style='color:red;'>❌ Hata: {str(e)}</h1>", 500

@app.route("/analytics/export_pdf")
def export_pdf():
    if not PDF_AVAILABLE:
        return jsonify({"error": "ReportLab yok - pip install reportlab"}), 500
    
    try:
        if not os.path.exists(Config.CSV_FILE):
            return jsonify({"error": "CSV dosyası yok"}), 404
        
        pdf_file = os.path.join(Config.STATIC_FOLDER, "smartgreenify_report.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        try:
            pdfmetrics.registerFont(TTFont('Turkish', 'DejaVuSans.ttf'))
            font_name = 'Turkish'
        except:
            font_name = 'Helvetica'
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=24,
            textColor=colors.HexColor('#27AE60'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=11
        )
        
        story.append(Paragraph("SmartGreenify Analytics Raporu", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        story.append(Paragraph(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        df = pd.read_csv(Config.CSV_FILE)
        
        if len(df) == 0:
            story.append(Paragraph("Veri yok", normal_style))
        else:
            stats_data = [
                ['Metrik', 'Deger'],
                ['Toplam Kayit', str(len(df))],
                ['Ort. Toprak Nemi', f"{df['soil'].mean():.1f}%"],
                ['Ort. Sicaklik', f"{df['temp'].mean():.1f}C"],
                ['Ort. Nem', f"{df['hum'].mean():.1f}%"],
                ['Toplam Sulama', str(df[df['pump'].astype(str).isin(['True', 'true', '1'])].shape[0])],
                ['Min Toprak', f"{df['soil'].min():.1f}%"],
                ['Max Toprak', f"{df['soil'].max():.1f}%"],
            ]
            
            stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(stats_table)
            story.append(Spacer(1, 0.5*inch))
            
            story.append(Paragraph("Son 10 Kayit", heading_style))
            recent_data = [['Tarih', 'Toprak', 'Sicaklik', 'Pompa']]
            for _, row in df.tail(10).iterrows():
                recent_data.append([
                    row['timestamp'][:16],
                    f"{row['soil']:.1f}%",
                    f"{row['temp']:.1f}C",
                    'Acik' if str(row['pump']) in ['True', 'true', '1'] else 'Kapali'
                ])
            
            recent_table = Table(recent_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
            recent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(recent_table)
        
        doc.build(story)
        
        send_ntfy("PDF Rapor Hazirlandi", f"Rapor basariyla olusturuldu! Toplam {len(df)} kayit.", "default", "page_facing_up")
        
        return jsonify({
            "success": True,
            "message": "PDF olusturuldu",
            "download_url": f"/static/smartgreenify_report.pdf"
        })
        
    except Exception as e:
        logger.error(f"PDF olusturma hatasi: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/analytics/export_excel")
def export_excel():
    """Excel Rapor"""
    if not EXCEL_AVAILABLE:
        return jsonify({"error": "OpenPyXL yok - pip install openpyxl"}), 500
    
    try:
        excel_file = os.path.join(Config.STATIC_FOLDER, "smartgreenify_report.xlsx")
        wb = Workbook()
        
        # Sayfa 1: Özet
        ws1 = wb.active
        ws1.title = "Ozet"
        ws1['A1'] = "SmartGreenify Analytics"
        ws1['A1'].font = Font(size=18, bold=True, color="27AE60")
        ws1['A1'].alignment = Alignment(horizontal='center')
        ws1.merge_cells('A1:D1')
        
        ws1['A3'] = "Rapor Tarihi:"
        ws1['B3'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        df = pd.read_csv(Config.CSV_FILE)
        ws1['A5'] = "Metrik"
        ws1['B5'] = "Deger"
        ws1['A5'].font = Font(bold=True)
        ws1['B5'].font = Font(bold=True)
        ws1['A5'].fill = PatternFill(start_color="27AE60", fill_type="solid")
        ws1['B5'].fill = PatternFill(start_color="27AE60", fill_type="solid")
        
        stats = [
            ["Toplam Kayit", len(df)],
            ["Ort. Toprak Nemi", f"{df['soil'].mean():.1f}%"],
            ["Ort. Sicaklik", f"{df['temp'].mean():.1f}C"],
            ["Ort. Nem", f"{df['hum'].mean():.1f}%"],
            ["Min Toprak", f"{df['soil'].min():.1f}%"],
            ["Max Toprak", f"{df['soil'].max():.1f}%"],
            ["Toplam Sulama", str(df[df['pump'].astype(str).isin(['True', 'true', '1'])].shape[0])],
        ]
        
        for idx, (metric, value) in enumerate(stats, start=6):
            ws1[f'A{idx}'] = metric
            ws1[f'B{idx}'] = value
        
        # Sayfa 2: Ham Veri
        ws2 = wb.create_sheet("Ham Veri")
        headers = ['Tarih', 'Bitki', 'Toprak', 'Sicaklik', 'Nem', 'Basinc', 'Isik', 'Pompa']
        ws2.append(headers)
        
        for cell in ws2[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="3498DB", fill_type="solid")
        
        for _, row in df.tail(100).iterrows():
            ws2.append([
                row['timestamp'],
                row['plant'],
                row['soil'],
                row['temp'],
                row['hum'],
                row['pressure'],
                'Gündüz' if row['light'] == 'Gündüz' else 'Gece',
                row['pump']
            ])
        
        # Sayfa 3: Grafik Verileri
        ws3 = wb.create_sheet("Istatistikler")
        ws3['A1'] = "Sensör İstatistikleri"
        ws3['A1'].font = Font(size=14, bold=True)
        
        ws3['A3'] = "Sensör"
        ws3['B3'] = "Ortalama"
        ws3['C3'] = "Minimum"
        ws3['D3'] = "Maksimum"
        ws3['E3'] = "Std. Sapma"
        
        for cell in ws3[3]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="F39C12", fill_type="solid")
        
        sensor_stats = [
            ["Toprak Nemi (%)", df['soil'].mean(), df['soil'].min(), df['soil'].max(), df['soil'].std()],
            ["Sıcaklık (°C)", df['temp'].mean(), df['temp'].min(), df['temp'].max(), df['temp'].std()],
            ["Nem (%)", df['hum'].mean(), df['hum'].min(), df['hum'].max(), df['hum'].std()],
            ["Basınç (hPa)", df['pressure'].mean(), df['pressure'].min(), df['pressure'].max(), df['pressure'].std()],
        ]
        
        for idx, stats_row in enumerate(sensor_stats, start=4):
            ws3[f'A{idx}'] = stats_row[0]
            ws3[f'B{idx}'] = round(stats_row[1], 2)
            ws3[f'C{idx}'] = round(stats_row[2], 2)
            ws3[f'D{idx}'] = round(stats_row[3], 2)
            ws3[f'E{idx}'] = round(stats_row[4], 2)
        
        # Grafik ekle
        chart = LineChart()
        chart.title = "Toprak Nemi Trendi"
        chart.style = 10
        chart.y_axis.title = 'Nem (%)'
        chart.x_axis.title = 'Kayıt #'
        
        data = Reference(ws2, min_col=3, min_row=1, max_row=min(100, len(df)+1))
        chart.add_data(data, titles_from_data=True)
        ws3.add_chart(chart, "G3")
        
        wb.save(excel_file)
        
        send_ntfy("Excel Rapor Hazirlandi", f"Rapor basariyla olusturuldu! {len(df)} kayit, 3 sayfa.", "default", "bar_chart")
        
        return jsonify({
            "success": True,
            "message": "Excel olusturuldu",
            "download_url": f"/static/smartgreenify_report.xlsx"
        })
        
    except Exception as e:
        logger.error(f"Excel olusturma hatasi: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌱 SMARTGREENIFY v4.0 - GERÇEK SENSÖRLER + ML LOGGING")
    print("="*70)
    print(f"✅ Sensörler: BME280 (SPI) + ADS1115 (I2C) + LDR + Röle")
    print(f"✅ Grafikler: Real-time Chart.js")
    print(f"✅ WebSocket: {SOCKETIO_AVAILABLE}")
    print(f"✅ Dark Mode: Aktif")
    print(f"✅ PWA: Kurulabilir")
    print(f"✅ ML Optimizer: {ML_AVAILABLE}")
    print(f"✅ Analytics Dashboard: {PLOTLY_AVAILABLE}")
    print(f"✅ PDF Raporlar: {PDF_AVAILABLE}")
    print(f"✅ Excel Raporlar: {EXCEL_AVAILABLE}")
    print(f"✅ Ntfy Bildirimler: {REQUESTS_AVAILABLE and Config.NTFY_ENABLED}")
    print(f"✅ Logging: Sistem/Sensör/Sulama/ML/Performans")
    print(f"✅ Veri Güncelleme: {Config.UPDATE_INTERVAL}s")
    print(f"✅ Port: {Config.FLASK_PORT}")
    print("="*70)
    print(f"\n🌐 Tarayıcıda aç: http://localhost:{Config.FLASK_PORT}")
    print("\n📊 Analytics Sayfaları:")
    print(f"   • Dashboard: http://localhost:{Config.FLASK_PORT}/analytics/dashboard")
    print(f"   • Heatmap: http://localhost:{Config.FLASK_PORT}/analytics/heatmap")
    print(f"   • Korelasyon: http://localhost:{Config.FLASK_PORT}/analytics/correlation")
    
    print("\n📁 Log Dosyaları:")
    print(f"   • Sistem: {Config.SYSTEM_LOG}")
    print(f"   • Sensör: {Config.SENSOR_LOG}")
    print(f"   • Sulama: {Config.IRRIGATION_LOG}")
    print(f"   • ML: {Config.ML_LOG}")
    print(f"   • Performans: {Config.PERFORMANCE_LOG}")
    print(f"   • Hatalar: {Config.ERROR_LOG}")
    
    if REQUESTS_AVAILABLE and Config.NTFY_ENABLED:
        print(f"\n📱 Ntfy Topic: {Config.NTFY_TOPIC}")
        print(f"   1. Ntfy uygulamasını aç")
        print(f"   2. Topic ekle: {Config.NTFY_TOPIC}")
        print(f"   3. Bildirimleri bekle!")
    
    print("\n🛑 CTRL+C ile kapatma\n")
    
    # Başlangıç bildirimi
    if REQUESTS_AVAILABLE and Config.NTFY_ENABLED:
        logger.info("📱 Test bildirimi gönderiliyor...")
        success = send_ntfy(
            "SmartGreenify Basladi",
            f"Sistem aktif! Gercek sensorler + ML logging. Port: {Config.FLASK_PORT}",
            "default",
            "seedling,robot"
        )
        if success:
            print("✅ Test bildirimi gönderildi!")
        else:
            print("⚠️ Test bildirimi gönderilemedi")
    
    try:
        if socketio:
            socketio.run(app, host="0.0.0.0", port=Config.FLASK_PORT, debug=False, allow_unsafe_werkzeug=True)
        else:
            app.run(host="0.0.0.0", port=Config.FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        logger.info("Kullanıcı durdurdu")
    finally:
        cleanup_resources()
