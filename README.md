# 🟠 Copper Prediction with Machine Learning — Claude

LME bakır nakit fiyatı için 12 ay ileri tahmin. Her ay üç döneme bölünüyor (1–10, 11–20, 21–ay sonu),
yani toplam **36 dönemlik tek bir tahmin** üretiliyor.

🌐 **Canlı panel:** https://copper-prediction-with-machine-learning-claude.streamlit.app/
(sayfalar: Tahmin · Etkenler · Haberler)

## Ne yapıyor?

1. **Veri:** LME günlük bakır fiyatı ve stokları (2008+), 55 piyasa ve makro seri (metaller, enerji, kurlar,
   faizler, ABD/Avrupa/Asya/Çin ekonomisi, risk endeksleri) ve bakır haberleri (GDELT, 2017+) kullanılıyor.
2. **Haber duyarlılığı:** GDELT'in haber tonu ve haber ilgisi ölçülüyor. Bakır haber başlıkları ayrıca
   **FinBERT** (finans metinleri için eğitilmiş BERT modeli) ile olumlu/olumsuz puanlanıyor.
3. **Modeller:**
   - ElasticNet: bakır + makro.
   - LightGBM: bakır + makro + haber.
   - ElasticNet: bakır + haber.
   Yayımlanan tahmin, bu üç modelin ortalaması.
4. **Optimizasyon:** Hiperparametreler her yıl, arada boşluk bırakan zaman serisi çapraz doğrulamasıyla
   yeniden seçiliyor. ElasticNet için alpha ve L1 oranı ayarlanıyor; LightGBM için 7 parametrede rastgele
   arama yapılıyor.
5. **Dürüst test:** 2015'ten bu yana her 10 günlük dönem için, yalnızca o gün bilinebilecek veriyle
   yeniden eğitim ve tahmin yapılıyor (426 başlangıç × 36 ufuk). Testler, gelecekteki verinin geçmiş
   tahmine sızmadığını doğruluyor.
6. **Etkenler sayfası:** Hangi verinin bugünkü tahmini yukarı/aşağı ittiğini ve geçmişte hangi verilerin
   hep aynı yönde ilişki gösterdiğini gösteriyor.

## Kullanım (VM)

```bash
cd ~/Copper_Prediction_with_Machine_Learning_Claude
./run.sh                 # veri kopyası + haberler + model + tahmin
./run.sh run             # sadece model ve tahmin (~7 dk)
./run.sh test            # sızıntı testleri
./restart_dashboard.sh   # panel: http://localhost:8501
```

## Yapı

```text
copperlab/
  config.py     ayarlar, 55 serilik kaynak listesi (yayın gecikmeleriyle), Türkçe adlar
  dekad.py      10 günlük dönem takvimi
  data.py       bakır dönem fiyatları, makro veriler, tahmin başlangıcı koruması
  news.py       GDELT haber tonu/ilgisi + başlıklar, FinBERT puanlama
  features.py   girdiler ve hedefler
  experts.py    ElasticNet / LightGBM / haberli ElasticNet, yıllık hiperparametre ayarı
  metrics.py    hata ölçüleri, Diebold–Mariano testi
  pipeline.py   uçtan uca çalıştırma, etki analizi, arşiv
  report.py     outputs/REPORT.md
app/            Streamlit: Tahmin ve Etkenler sayfaları
tests/          sızıntı ve tutarlılık testleri
```

## Çıktılar (`outputs/`)

| Dosya | İçerik |
|---|---|
| `forecast_latest.csv` | 36 dönemlik tahmin ve her dönemin geçmiş hata oranı |
| `input_effects.csv` | Her girdinin bugünkü katkısı (USD/t) ve geçmişteki yön tutarlılığı |
| `metrics_by_horizon.csv` | Model × ufuk başına geriye dönük ölçüler |
| `news_sentiment.csv` | 10 günlük haber tonu, ilgi ve FinBERT duyarlılığı |
| `news_value_test.csv` | Haber eklemenin tahmini iyileştirip iyileştirmediği |
| `tuning.csv` | Her yıl seçilen hiperparametreler |
| `REPORT.md`, `run_manifest.json`, `archive/` | Özet rapor, çalıştırma kaydı, yayımlanmış tahminler |

> Araştırma amaçlıdır, yatırım tavsiyesi değildir.
