# Bakır tahmin raporu — başlangıç 2026-10

Oluşturulma: 2026-11-09T20:11:33+00:00 UTC · kod `69b4412` · süre 15.5 sn

Başlangıç fiyatı (2026-10 LME nakit ortalaması): **14.532 USD/t**

## Tahmin

| Ufuk | Ay | Tahmin | %80 aralık | Yukarı olasılığı | Değerlendirme |
|---|---|---:|---|---:|---|
| H1 | 2026-11 | 14.406 | 13.763 – 15.036 | %35 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H2 | 2026-12 | 14.363 | 13.122 – 15.489 | %44 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H3 | 2027-01 | 14.320 | 12.848 – 16.414 | %45 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H4 | 2027-02 | 14.320 | 12.727 – 16.692 | %45 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H5 | 2027-03 | 14.320 | 12.455 – 16.965 | %44 | Rastgele yürüyüşten biraz iyi, fark istatistiksel olarak kanıtlanmadı |
| H6 | 2027-04 | 14.320 | 12.317 – 17.163 | %45 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H7 | 2027-05 | 14.320 | 12.108 – 17.175 | %44 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H8 | 2027-06 | 14.320 | 11.962 – 17.412 | %46 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H9 | 2027-07 | 14.320 | 11.735 – 17.572 | %44 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H10 | 2027-08 | 14.320 | 11.795 – 18.146 | %48 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H11 | 2027-09 | 14.320 | 11.633 – 18.487 | %47 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |
| H12 | 2027-10 | 14.320 | 11.348 – 18.732 | %46 | Rastgele yürüyüşü istatistiksel olarak anlamlı şekilde geçiyor |

## ML senaryosu (yayımlanan tahmine dahil değil)

Üç ML uzmanının eşit ağırlıklı ortalaması. Geçmişte vadeli eğri kuralından daha kötü sonuç verdiği için ana tahmine katılmaz; bilgi amaçlı gösterilir.

| Ufuk | ML senaryosu | Backtest MAPE | Vadeli eğriye göre |
|---|---:|---:|---:|
| H1 | 14.379 | %2,87 | %-6,6 |
| H2 | 14.365 | %5,81 | %-10,5 |
| H3 | 14.079 | %7,83 | %-14,5 |
| H4 | 13.516 | %9,90 | %-20,2 |
| H5 | 13.294 | %11,36 | %-22,3 |
| H6 | 13.246 | %12,91 | %-23,9 |
| H7 | 13.145 | %13,45 | %-14,8 |
| H8 | 12.938 | %14,23 | %-14,6 |
| H9 | 12.722 | %15,23 | %-17,8 |
| H10 | 12.732 | %16,07 | %-16,6 |
| H11 | 12.566 | %17,00 | %-15,6 |
| H12 | 12.462 | %18,02 | %-15,5 |

## Model güvenilirliği (walk-forward backtest)

Beceri = rastgele yürüyüşe göre ortalama mutlak hatadaki azalma. p < 0,10 ise fark istatistiksel olarak anlamlı kabul edilir (Diebold-Mariano).

| Ufuk | Gözlem | Kombinasyon MAPE | RW MAPE | Beceri | DM p | En iyi uzman |
|---|---:|---:|---:|---:|---:|---|
| H1 | 141 | %2,73 | %3,54 | %21,3 | 0,000 | LME vadeli eğrisi (ay sonu + contango) |
| H2 | 140 | %5,30 | %5,81 | %7,4 | 0,026 | LME vadeli eğrisi (ay sonu + contango) |
| H3 | 139 | %6,97 | %7,32 | %4,5 | 0,069 | LME vadeli eğrisi (ay sonu + contango) |
| H4 | 138 | %8,19 | %8,52 | %4,6 | 0,069 | LME vadeli eğrisi (ay sonu + contango) |
| H5 | 137 | %9,25 | %9,51 | %3,4 | 0,126 | LME vadeli eğrisi (ay sonu + contango) |
| H6 | 136 | %10,33 | %10,91 | %6,1 | 0,005 | LME vadeli eğrisi (ay sonu + contango) |
| H7 | 135 | %11,54 | %12,04 | %4,4 | 0,031 | LME vadeli eğrisi (ay sonu + contango) |
| H8 | 134 | %12,35 | %12,76 | %3,9 | 0,045 | LME vadeli eğrisi (ay sonu + contango) |
| H9 | 133 | %12,98 | %13,60 | %5,1 | 0,002 | LME vadeli eğrisi (ay sonu + contango) |
| H10 | 132 | %13,91 | %14,52 | %4,3 | 0,004 | LME vadeli eğrisi (ay sonu + contango) |
| H11 | 131 | %14,89 | %15,51 | %4,1 | 0,002 | LME vadeli eğrisi (ay sonu + contango) |
| H12 | 130 | %15,77 | %16,28 | %3,3 | 0,007 | LME vadeli eğrisi (ay sonu + contango) |

## Aralıkların kalibrasyonu

İyi kalibre edilmiş bir %80 aralık gerçekleşmelerin yaklaşık %80'ini kapsamalı.

| Ufuk | Gözlem | %50 kapsama | %80 kapsama | %90 kapsama |
|---|---:|---:|---:|---:|
| H1 | 117 | %46 | %79 | %85 |
| H2 | 115 | %43 | %83 | %91 |
| H3 | 113 | %51 | %82 | %88 |
| H4 | 111 | %49 | %80 | %90 |
| H5 | 109 | %51 | %80 | %87 |
| H6 | 107 | %53 | %81 | %85 |
| H7 | 105 | %55 | %77 | %85 |
| H8 | 103 | %53 | %76 | %88 |
| H9 | 101 | %61 | %76 | %89 |
| H10 | 99 | %65 | %76 | %90 |
| H11 | 97 | %65 | %77 | %87 |
| H12 | 95 | %67 | %79 | %84 |

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
| H1 | 14.406 | 14.623 | 217 | Evet |
| H2 | 14.363 | 14.725 | 363 | Evet |
| H3 | 14.320 | 16.552 | 2.232 | Hayır |
| H4 | 14.320 | 15.486 | 1.166 | Evet |
| H5 | 14.320 | 14.225 | -95 | Evet |
| H6 | 14.320 | 14.641 | 322 | Evet |
| H7 | 14.320 | 15.812 | 1.492 | Evet |
| H8 | 14.320 | 15.476 | 1.157 | Evet |
| H9 | 14.320 | 15.687 | 1.367 | Evet |
| H10 | 14.320 | 17.749 | 3.429 | Evet |
| H11 | 14.320 | 19.669 | 5.349 | Hayır |
| H12 | 14.320 | 17.968 | 3.649 | Evet |

## Sınırlar

- Veriler güncel sürümdür (revizyonlar sonrası); geçmişte yayımlanmış ilk değerler değildir.
- Yayın gecikmeleri varsayımdır: piyasa verileri 0, risk endeksleri 1–2, resmi makro veriler 2 ay.
- Çakışan çok aylık getiriler nedeniyle uzun ufuklarda bağımsız gözlem sayısı görünenden azdır.
- Bu çalışma yatırım tavsiyesi değildir.
