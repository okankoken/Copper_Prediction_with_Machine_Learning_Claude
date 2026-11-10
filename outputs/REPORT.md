# Bakır tahmini — son veri: 21-31 Eki 2026

Oluşturulma 2026-11-10T19:26:48+00:00 UTC · kod `115b5f5` · 536.5 sn

Son 10 günlük dönem ortalaması: **14.485 USD/t**

## Tahmin (10 günlük dönem ortalamaları)

| Dönem | Tahmin USD/t | Değişim | Geçmiş hata (MAPE) | Yön isabeti |
|---|---:|---:|---:|---:|
| 1-10 Kas 2026 | 14.452 | -0,2% | %1,7 | %68 |
| 11-20 Kas 2026 | 14.517 | 0,2% | %3,2 | %60 |
| 21-30 Kas 2026 | 14.515 | 0,2% | %4,2 | %53 |
| 1-10 Ara 2026 | 14.553 | 0,5% | %4,9 | %52 |
| 11-20 Ara 2026 | 14.555 | 0,5% | %5,6 | %52 |
| 21-31 Ara 2026 | 14.721 | 1,6% | %6,5 | %49 |
| 1-10 Oca 2027 | 14.744 | 1,8% | %6,9 | %52 |
| 11-20 Oca 2027 | 14.751 | 1,8% | %7,5 | %48 |
| 21-31 Oca 2027 | 14.771 | 2,0% | %8,1 | %52 |
| 1-10 Şub 2027 | 14.807 | 2,2% | %8,5 | %53 |
| 11-20 Şub 2027 | 14.781 | 2,0% | %8,9 | %55 |
| 21-28 Şub 2027 | 14.822 | 2,3% | %9,2 | %55 |
| 1-10 Mar 2027 | 14.867 | 2,6% | %9,6 | %55 |
| 11-20 Mar 2027 | 14.600 | 0,8% | %9,8 | %52 |
| 21-31 Mar 2027 | 14.635 | 1,0% | %10,4 | %54 |
| 1-10 Nis 2027 | 14.656 | 1,2% | %10,8 | %54 |
| 11-20 Nis 2027 | 14.647 | 1,1% | %11,2 | %53 |
| 21-30 Nis 2027 | 14.650 | 1,1% | %11,5 | %55 |
| 1-10 May 2027 | 14.859 | 2,6% | %12,0 | %55 |
| 11-20 May 2027 | 14.649 | 1,1% | %12,6 | %53 |
| 21-31 May 2027 | 14.574 | 0,6% | %13,1 | %51 |
| 1-10 Haz 2027 | 14.661 | 1,2% | %13,4 | %50 |
| 11-20 Haz 2027 | 14.809 | 2,2% | %13,7 | %49 |
| 21-30 Haz 2027 | 14.910 | 2,9% | %14,0 | %48 |
| 1-10 Tem 2027 | 14.882 | 2,7% | %14,1 | %50 |
| 11-20 Tem 2027 | 14.694 | 1,4% | %14,1 | %50 |
| 21-31 Tem 2027 | 14.700 | 1,5% | %14,2 | %50 |
| 1-10 Ağu 2027 | 14.784 | 2,1% | %16,4 | %44 |
| 11-20 Ağu 2027 | 14.977 | 3,4% | %16,5 | %44 |
| 21-31 Ağu 2027 | 15.060 | 4,0% | %16,5 | %46 |
| 1-10 Eyl 2027 | 15.108 | 4,3% | %16,3 | %50 |
| 11-20 Eyl 2027 | 15.169 | 4,7% | %16,2 | %52 |
| 21-30 Eyl 2027 | 14.974 | 3,4% | %16,1 | %56 |
| 1-10 Eki 2027 | 14.788 | 2,1% | %16,4 | %58 |
| 11-20 Eki 2027 | 14.784 | 2,1% | %16,7 | %58 |
| 21-31 Eki 2027 | 14.725 | 1,7% | %16,9 | %57 |

## Geriye dönük karne

'Fiyat aynı kalır' tahminine göre hatadaki azalma (%), pozitif = daha iyi.

| Model | 1 ay | 3 ay | 6 ay | 12 ay |
|---|---:|---:|---:|---:|
| ML tahmini (3 modelin ortalaması) | 0,4 | -5,9 | -1,1 | -1,4 |
| ElasticNet (bakır + makro) | 2,1 | -6,1 | -1,5 | -20,0 |
| LightGBM (bakır + makro + haber) | -3,6 | -15,5 | -5,8 | 12,2 |
| ElasticNet (bakır + haber) | 5,0 | 10,2 | 9,7 | -20,3 |
| LME vadeli eğrisi | 6,3 | 1,7 | 1,9 | 1,7 |

## Haber duyarlılığı katkısı

Haberli ElasticNet'in habersiz ElasticNet'e göre hata azalması (%), aynı dönemde.

| Ufuk | Gözlem | Katkı | p |
|---|---:|---:|---:|
| 1 ay | 157 | 0,3 | 0,454 |
| 3 ay | 145 | 21,0 | 0,102 |
| 6 ay | 127 | 7,5 | 0,129 |
| 12 ay | 91 | -22,7 | 0,633 |

## Tahmini en çok etkileyen girdiler (3 ay)

**Yukarı itenler:** Japonya 10 yıllık faiz (12 aylık değişim) (52 $); Çin sanayi üretimi (12 aylık değişim) (48 $); Çin sabit sermaye yatırımı (12 aylık değişim) (37 $); USD/JPY (3 aylık değişim) (10 $); ABD para arzı (M2) (3 aylık değişim) (5 $); Peru dizel fiyatı (3 aylık değişim) (4 $); Euro bölgesi ÜFE (3 aylık değişim) (3 $); Kore 10 yıllık faiz (12 aylık değişim) (3 $)

**Aşağı çekenler:** Fed faizi (12 aylık değişim) (-36 $); Madencilik / Şili / Peru haber duyarlılığı (-26 $); Çin gayrimenkul yatırımı (12 aylık değişim) (-16 $); Çin ÜFE (12 aylık değişim) (-14 $); Peru dizel fiyatı (12 aylık değişim) (-13 $); Euro bölgesi bakır ÜFE (3 aylık değişim) (-8 $); LME çinko stoku (3 aylık değişim) (-7 $); Altın (12 aylık değişim) (-7 $)

## Sınırlar

- Veriler güncel sürümdür; geçmişte ilk yayımlanan değerler değildir.
- Haber verisi 2017'den başlar.
- Yatırım tavsiyesi değildir.
