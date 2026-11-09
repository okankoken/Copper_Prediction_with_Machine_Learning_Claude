# Bakır tahmin raporu — başlangıç 2026-10

Oluşturulma: 2026-11-09T19:42:42+00:00 UTC · kod `uncommitted` · süre 15.6 sn

Başlangıç fiyatı (2026-10 LME nakit ortalaması): **14.532 USD/t**

## Tahmin

| Ufuk | Ay | Tahmin | %80 aralık | Yukarı olasılığı | Değerlendirme |
|---|---|---:|---|---:|---|
| H1 | 2026-11 | 14.449 | 13.813 – 15.062 | %40 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H2 | 2026-12 | 14.449 | 13.232 – 15.581 | %46 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H3 | 2027-01 | 14.449 | 12.971 – 16.595 | %48 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H4 | 2027-02 | 14.449 | 12.786 – 16.859 | %49 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H5 | 2027-03 | 14.449 | 12.590 – 17.115 | %48 | Rastgele yürüyüşten biraz iyi, fark istatistiksel olarak kanıtlanmadı |
| H6 | 2027-04 | 14.449 | 12.432 – 17.323 | %47 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H7 | 2027-05 | 14.449 | 12.201 – 17.369 | %48 | Rastgele yürüyüşten biraz iyi, fark istatistiksel olarak kanıtlanmadı |
| H8 | 2027-06 | 14.449 | 12.033 – 17.540 | %48 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H9 | 2027-07 | 14.449 | 11.877 – 17.733 | %50 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H10 | 2027-08 | 14.449 | 11.818 – 18.178 | %48 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H11 | 2027-09 | 14.449 | 11.683 – 18.624 | %48 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H12 | 2027-10 | 14.449 | 11.361 – 18.803 | %46 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |

## Model güvenilirliği (walk-forward backtest)

Beceri = rastgele yürüyüşe göre ortalama mutlak hatadaki azalma. p < 0,10 ise fark istatistiksel olarak anlamlı kabul edilir (Diebold-Mariano).

| Ufuk | Gözlem | Kombinasyon MAPE | RW MAPE | Beceri | DM p | En iyi uzman |
|---|---:|---:|---:|---:|---:|---|
| H1 | 141 | %2,75 | %3,54 | %20,7 | 0,000 | Ay sonu fiyatı sabit kalır |
| H2 | 140 | %5,32 | %5,81 | %7,0 | 0,029 | Ay sonu fiyatı sabit kalır |
| H3 | 139 | %7,00 | %7,32 | %3,8 | 0,089 | Ay sonu fiyatı sabit kalır |
| H4 | 138 | %8,18 | %8,52 | %4,6 | 0,056 | Ay sonu fiyatı sabit kalır |
| H5 | 137 | %9,27 | %9,51 | %3,1 | 0,139 | Ay sonu fiyatı sabit kalır |
| H6 | 136 | %10,39 | %10,91 | %5,5 | 0,010 | Ay sonu fiyatı sabit kalır |
| H7 | 135 | %11,75 | %12,04 | %2,3 | 0,185 | Ay sonu fiyatı sabit kalır |
| H8 | 134 | %12,46 | %12,76 | %2,8 | 0,089 | Ay sonu fiyatı sabit kalır |
| H9 | 133 | %13,13 | %13,60 | %3,9 | 0,011 | Ay sonu fiyatı sabit kalır |
| H10 | 132 | %14,03 | %14,52 | %3,4 | 0,015 | Ay sonu fiyatı sabit kalır |
| H11 | 131 | %14,99 | %15,51 | %3,2 | 0,007 | Ay sonu fiyatı sabit kalır |
| H12 | 130 | %15,90 | %16,28 | %2,3 | 0,036 | Ay sonu fiyatı sabit kalır |

## Aralıkların kalibrasyonu

İyi kalibre edilmiş bir %80 aralık gerçekleşmelerin yaklaşık %80'ini kapsamalı.

| Ufuk | Gözlem | %50 kapsama | %80 kapsama | %90 kapsama |
|---|---:|---:|---:|---:|
| H1 | 117 | %46 | %79 | %86 |
| H2 | 115 | %43 | %80 | %90 |
| H3 | 113 | %50 | %81 | %88 |
| H4 | 111 | %50 | %81 | %90 |
| H5 | 109 | %50 | %80 | %88 |
| H6 | 107 | %52 | %81 | %85 |
| H7 | 105 | %56 | %77 | %85 |
| H8 | 103 | %54 | %75 | %86 |
| H9 | 101 | %61 | %74 | %87 |
| H10 | 99 | %64 | %74 | %86 |
| H11 | 97 | %66 | %75 | %86 |
| H12 | 95 | %64 | %78 | %82 |

## Tahmini etkileyen başlıca girdiler (makro faktör Ridge uzmanı)

Katkı = katsayı × standartlaştırılmış güncel değer, log-getiri cinsinden; USD karşılığı yaklaşık değerdir. Nedensellik değil, modelin koşullu ilişkisidir.

**H1:** 3 ay vadeli - nakit farkı (contango) (-116 USD/t); Ay sonu fiyatı / ay ortalaması (-106 USD/t); 6 aylık oynaklık (-42 USD/t); Bakır 3 aylık getiri (-40 USD/t); Makro faktör PC2: +euro_area_ppi(c12), +us_2y(c12), +us_cpi(c12) (-35 USD/t)

**H3:** 3 ay vadeli - nakit farkı (contango) (-297 USD/t); 6 aylık oynaklık (-215 USD/t); Makro faktör PC2: +euro_area_ppi(c12), +us_2y(c12), +us_cpi(c12) (-145 USD/t); Ay sonu fiyatı / ay ortalaması (-88 USD/t); Bakır 3 aylık getiri (-65 USD/t)

**H6:** 6 aylık oynaklık (-680 USD/t); 3 ay vadeli - nakit farkı (contango) (-516 USD/t); LME stok 12 aylık değişim (-475 USD/t); Makro faktör PC2: +euro_area_ppi(c12), +us_2y(c12), +us_cpi(c12) (-430 USD/t); Bakır 12 aylık getiri (-398 USD/t)

**H12:** 3 ay vadeli - nakit farkı (contango) (-1.446 USD/t); Makro faktör PC2: +euro_area_ppi(c12), +us_2y(c12), +us_cpi(c12) (-999 USD/t); Bakır 3 aylık getiri (-916 USD/t); 6 aylık oynaklık (-855 USD/t); Makro faktör PC3: +bund_10y(c12), +platinum(c3), -us_housing_starts(c12) (+465 USD/t)

## Orijinal projenin tahmini ile karşılaştırma

Orijinal proje tahmin başlangıcı: 2026-10-31. Orijinal dosya yalnızca okunur kopya olarak alındı.

| Ufuk | Bu proje | Orijinal | Fark | Orijinal bu projenin %80 aralığında mı? |
|---|---:|---:|---:|---|
| H1 | 14.449 | 14.623 | 174 | Evet |
| H2 | 14.449 | 14.725 | 276 | Evet |
| H3 | 14.449 | 16.552 | 2.103 | Evet |
| H4 | 14.449 | 15.486 | 1.037 | Evet |
| H5 | 14.449 | 14.225 | -224 | Evet |
| H6 | 14.449 | 14.641 | 192 | Evet |
| H7 | 14.449 | 15.812 | 1.363 | Evet |
| H8 | 14.449 | 15.476 | 1.027 | Evet |
| H9 | 14.449 | 15.687 | 1.238 | Evet |
| H10 | 14.449 | 17.749 | 3.300 | Evet |
| H11 | 14.449 | 19.669 | 5.220 | Hayır |
| H12 | 14.449 | 17.968 | 3.519 | Evet |

## Sınırlar

- Veriler güncel sürümdür (revizyonlar sonrası); geçmişte yayımlanmış ilk değerler değildir.
- Yayın gecikmeleri varsayımdır: piyasa verileri 0, risk endeksleri 1–2, resmi makro veriler 2 ay.
- Çakışan çok aylık getiriler nedeniyle uzun ufuklarda bağımsız gözlem sayısı görünenden azdır.
- Bu çalışma yatırım tavsiyesi değildir.
