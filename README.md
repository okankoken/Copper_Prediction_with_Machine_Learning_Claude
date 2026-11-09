# 🟠 Copper Prediction with Machine Learning — Claude

LME bakır nakit fiyatının aylık ortalaması için 1–12 ay ileri, **olasılıklı** tahmin sistemi.

Bu proje, `Copper_Prediction_with_Machine_Learning` projesinin amacını (H1–H12 bakır tahmini, izlenebilir
üretim, gösterge paneli) farklı bir felsefeyle yeniden kurar. Orijinal projeye hiçbir şey yazmaz.
Ham verilerin yalnızca okunur bir kopyasını alır.

## Bu versiyonu farklı yapan ne?

| | Orijinal proje | Bu proje |
|---|---|---|
| Geçmiş | Ana tablo 2015'ten başlıyor, H1 için 21, H12 için 10 backtest gözlemi | Ham veriden 2008'e kadar yeniden kurulan panel; ufuk başına **130–141** backtest gözlemi |
| Kıyas | Naive yöntem referans olarak tutuluyor | Her uzman **ay sonu fiyatı** kuralını yenmek zorunda; yenemeyen ağırlık alamaz |
| Belirsizlik | Güven etiketi (LOW/NORMAL) | Kalibre edilmiş P5–P95 aralıklar ve yukarı/aşağı olasılığı; kapsama oranı test ediliyor |
| Kanıt | Hata ölçütleri | Diebold–Mariano anlamlılık testi ve her ufuk için açık "geçiyor / geçmiyor" kararı |
| Üretim | Ayrı üretim kodu | Üretim tahmini = backtest'in son adımı (aynı fonksiyon) |
| Sızıntı | Belgelenmiş kurallar | Kurallar **testlerle** doğrulanıyor (gelecek veri bozulsa da geçmiş tahmin değişmiyor) |

## Ana bulgu (başlangıç Ekim 2026)

- Ayın son işlem günü fiyatının korunacağını söyleyen basit kural, rastgele yürüyüşü her ufukta yeniyor
  (H1'de hata %21 daha düşük). Sebebi aylık ortalama almanın yarattığı gecikme etkisi.
- Bakır sinyalleri, 54 makro seri ve LightGBM üzerine kurulu ML uzmanları H1'de rastgele yürüyüşü
  geçiyor ama ay sonu kuralını geçemiyor. H3 ve sonrasında rastgele yürüyüşten de kötüler.
  Sistem bu yüzden şu an onlara ağırlık vermiyor. Her ay yeniden değerlendiriliyorlar.
- Yayımlanan kombinasyonun rastgele yürüyüşe göre becerisi H1'de **+%20,7**, H12'de **+%2,3**.
  Fark H1–H4, H6 ve H8–H12'de istatistiksel olarak anlamlı (p < 0,10).
- %80 aralıklar H1–H6'da %79–81, H7–H12'de %74–78 kapsıyor.

Ayrıntılı tablo her çalıştırmada `outputs/REPORT.md` dosyasına yazılır.

## Hızlı başlangıç (VM)

```bash
cd ~/Copper_Prediction_with_Machine_Learning_Claude
./run.sh                 # veri kopyası + backtest + tahmin + rapor (~20 sn)
./run.sh test            # sızıntı ve tutarlılık testleri
./run.sh dashboard       # http://localhost:8501
```

Doğrudan çağrılar da kullanılabilir: `python -m copperlab snapshot|run|all [--allow-stale] [--run-date YYYY-MM-DD]`.

Tahmin koruması: bakır verisi son tamamlanmış aya kadar güncel değilse çalıştırma durur. Yalnızca
backtest için `--allow-stale` kullanılabilir.

## Yapı

```text
copperlab/
  config.py     yollar, ufuklar, 54 serilik kaynak listesi ve her birinin yayın gecikmesi
  snapshot.py   orijinal projeden ham veriyi kopyalar (SHA256 manifest)
  data.py       aylık panel, süren ayı dışlama, tahmin başlangıcı koruması
  features.py   bakırın kendi sinyalleri + makro değişimler + hedefler
  experts.py    walk-forward uzmanlar (rastgele yürüyüş, ay sonu fiyatı, Ridge, faktör Ridge, LightGBM)
  combine.py    ağırlık kazanma kuralı + ampirik tahmin aralıkları
  metrics.py    MAPE/RMSE/sapma/yön, Diebold–Mariano, kapsama
  pipeline.py   uçtan uca çalıştırma, arşiv, manifest
  report.py     Türkçe rapor
app/dashboard.py   Streamlit paneli (6 sekme)
tests/             9 sızıntı/tutarlılık testi
docs/METHOD.md     yöntem notu
```

## Çıktılar (`outputs/`)

| Dosya | İçerik |
|---|---|
| `forecast_latest.csv` | H1–H12 tahmin, P5–P95, yukarı olasılığı, uzman tahminleri, ağırlıklar, karar |
| `metrics_by_horizon.csv` | Uzman × ufuk başına tüm backtest ölçütleri |
| `interval_calibration.csv` | %50/%80/%90 aralık kapsama oranları |
| `backtest_*.csv`, `expert_weights_history.csv` | Tüm geriye dönük tahminler ve ağırlık geçmişi |
| `drivers_latest.csv` | Makro faktör uzmanının katkı ayrıştırması |
| `run_manifest.json` | Yapılandırma, veri SHA256, kod commit'i |
| `archive/` | Yayımlanan her tahminin değişmez kopyası |

Bağımlılıklar VM'deki `miniconda3` ortamında hazır (pandas, scikit-learn, lightgbm, statsmodels,
streamlit, plotly). Ek kurulum gerekmez.

> Bu çalışma araştırma amaçlıdır, yatırım tavsiyesi değildir.
