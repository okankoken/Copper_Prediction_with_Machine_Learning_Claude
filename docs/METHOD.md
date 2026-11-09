## Yöntem

**Hedef.** LME bakır nakit fiyatının aylık ortalaması (USD/t), 1–12 ay sonrası. Modeller fiyatı değil,
log getiriyi tahmin eder: `y_h(t) = log(P[t+h] / P[t])`.

**Tek kural: her tahmin, o gün bilinebilecek bilgiyle yapılır.**

- Her serinin açık bir yayın gecikmesi var: piyasa verileri 0, risk endeksleri 1–2, resmi makro 2 ay.
- Süren ay hiç kullanılmaz; tahmin başlangıcı son tamamlanmış aydır.
- h ay ufkunda model yalnızca o gün gerçekleşmiş hedeflerle eğitilir (`s + h ≤ t`).
- Hiperparametreler yalnızca o günkü eğitim verisinde, arada h−1 aylık boşluk bırakan zaman serisi
  çapraz doğrulamasıyla seçilir.
- Üretim tahmini, geriye dönük testin son adımıdır (aynı kod).

**Girdiler.**

| Grup | İçerik |
|---|---|
| Bakırın kendisi | 1/3/6/12 aylık getiri, 12 aylık ortalamadan sapma, oynaklık, ay sonu / ay ortalaması, 3 ay vadeli – nakit farkı, LME stokları |
| Makro & piyasa | 55 seri (metaller, enerji, kur, faiz, ABD/Avrupa/Asya/Çin makro, risk endeksleri) → 5 PCA faktörü |
| Haber | GDELT bakır piyasası haber tonu ve haber ilgisi; haber başlıklarının FinBERT duyarlılığı (2017+) |

**Modeller.**

| Model | Ne yapar | Ayar |
|---|---|---|
| Fiyat aynı kalır | Kıyas çizgisi | – |
| LME vadeli eğrisi | Ay sonu fiyatı + nakit → 3 ay vadeli fark, 3 aydan sonra düz | – |
| ElasticNet | Bakır + makro faktörler, doğrusal | alpha ve L1 oranı her ay, 4 katlı zaman serisi CV |
| LightGBM | Bakır + makro faktörler + haber | 7 hiperparametre, 15 adaylı rastgele arama, yılda bir, 3 katlı CV |
| Haberli ElasticNet | Bakır + haber duyarlılığı (2017+ verisiyle) | ElasticNet ile aynı |

**ML modeli** = son üç modelin eşit ağırlıklı ortalaması. Panelde ayrıca gösterilir.

**Ana tahmin** = kazanılan ağırlıkla birleşim. Bir model, son değerlendirme penceresinde vadeli eğriden
daha az hata yaptıysa ağırlık alır. h aylık tahminler birbiriyle örtüştüğü için en az 12 bağımsız sonuç
(12·h tahmin) birikmeden ağırlık verilmez.

**Değerlendirme.** Ocak 2015 – Ekim 2026 arasında her ay için yeniden yapılan tahminler (ufuk başına
130–141; haberli model için 2020 sonrası). Ölçüler: MAPE, yön isabeti, "fiyat aynı kalır" tahminine göre
hatadaki azalma ve Diebold–Mariano testi.

**Sınırlar.** Veriler güncel sürümdür. Yayın gecikmeleri varsayımdır. Haber verisi 2017'de başlar.
Yatırım tavsiyesi değildir.
