# Bakır tahmini — son veri: 1-10 Kas 2026

Oluşturulma 2026-11-14T22:00:58+00:00 UTC · kod `f3e1536` · 334.5 sn

Son 10 günlük dönem ortalaması: **14.111 USD/t**

## Tahmin (10 günlük dönem ortalamaları)

| Dönem | Tahmin USD/t | Değişim | Geçmiş hata (MAPE) | Yön isabeti |
|---|---:|---:|---:|---:|
| 11-20 Kas 2026 | 13.996 | -0,8% | %1,7 | %67 |
| 21-30 Kas 2026 | 14.031 | -0,6% | %3,2 | %59 |
| 1-10 Ara 2026 | 14.120 | 0,1% | %4,2 | %53 |
| 11-20 Ara 2026 | 14.126 | 0,1% | %4,9 | %51 |
| 21-31 Ara 2026 | 14.108 | -0,0% | %5,7 | %52 |
| 1-10 Oca 2027 | 14.127 | 0,1% | %6,5 | %50 |
| 11-20 Oca 2027 | 14.370 | 1,8% | %6,8 | %53 |
| 21-31 Oca 2027 | 14.469 | 2,5% | %7,2 | %53 |
| 1-10 Şub 2027 | 14.428 | 2,2% | %7,7 | %56 |
| 11-20 Şub 2027 | 14.365 | 1,8% | %8,2 | %55 |
| 21-28 Şub 2027 | 14.341 | 1,6% | %8,7 | %56 |
| 1-10 Mar 2027 | 14.355 | 1,7% | %9,1 | %54 |
| 11-20 Mar 2027 | 14.325 | 1,5% | %9,5 | %55 |
| 21-31 Mar 2027 | 14.437 | 2,3% | %10,3 | %54 |
| 1-10 Nis 2027 | 14.381 | 1,9% | %10,9 | %53 |
| 11-20 Nis 2027 | 14.416 | 2,2% | %11,3 | %52 |
| 21-30 Nis 2027 | 14.406 | 2,1% | %11,7 | %49 |
| 1-10 May 2027 | 14.332 | 1,6% | %12,2 | %50 |
| 11-20 May 2027 | 14.364 | 1,8% | %12,8 | %49 |
| 21-31 May 2027 | 14.410 | 2,1% | %13,3 | %46 |
| 1-10 Haz 2027 | 14.327 | 1,5% | %13,6 | %46 |
| 11-20 Haz 2027 | 14.295 | 1,3% | %13,8 | %46 |
| 21-30 Haz 2027 | 14.479 | 2,6% | %13,9 | %46 |
| 1-10 Tem 2027 | 14.547 | 3,1% | %14,0 | %48 |
| 11-20 Tem 2027 | 14.503 | 2,8% | %14,0 | %51 |
| 21-31 Tem 2027 | 14.430 | 2,3% | %14,0 | %49 |
| 1-10 Ağu 2027 | 14.341 | 1,6% | %14,2 | %52 |
| 11-20 Ağu 2027 | 14.286 | 1,2% | %16,7 | %47 |
| 21-31 Ağu 2027 | 14.174 | 0,4% | %16,8 | %46 |
| 1-10 Eyl 2027 | 14.177 | 0,5% | %17,0 | %47 |
| 11-20 Eyl 2027 | 14.155 | 0,3% | %17,0 | %48 |
| 21-30 Eyl 2027 | 13.912 | -1,4% | %17,2 | %49 |
| 1-10 Eki 2027 | 13.834 | -2,0% | %17,3 | %54 |
| 11-20 Eki 2027 | 13.677 | -3,1% | %17,5 | %56 |
| 21-31 Eki 2027 | 13.542 | -4,0% | %17,6 | %57 |
| 1-10 Kas 2027 | 13.643 | -3,3% | %17,8 | %57 |

## Geriye dönük karne

'Fiyat aynı kalır' tahminine göre hatadaki azalma (%), pozitif = daha iyi.

| Model | 1 ay | 3 ay | 6 ay | 12 ay |
|---|---:|---:|---:|---:|
| ML tahmini (3 modelin ortalaması) | 0,4 | -1,0 | -6,3 | -6,0 |
| ElasticNet (bakır + makro) | 1,9 | -0,9 | -8,3 | -16,8 |
| LightGBM (bakır + makro + haber) | -2,9 | -11,1 | -13,1 | 2,1 |
| ElasticNet (bakır + haber) | 5,1 | 10,4 | 9,9 | -23,4 |
| LME vadeli eğrisi | 6,2 | 1,7 | 2,0 | 1,7 |

## Haber duyarlılığı katkısı

Haberli ElasticNet'in habersiz ElasticNet'e göre hata azalması (%), aynı dönemde.

| Ufuk | Gözlem | Katkı | p |
|---|---:|---:|---:|
| 1 ay | 158 | 0,6 | 0,415 |
| 3 ay | 147 | 6,4 | 0,050 |
| 6 ay | 129 | 7,4 | 0,137 |
| 12 ay | 93 | -44,2 | 0,802 |

## Tahmini en çok etkileyen girdiler (3 ay)

**Yukarı itenler:** Japonya 10 yıllık faiz (12 aylık değişim) (68 $); Çin sabit sermaye yatırımı (12 aylık değişim) (32 $); BYD (3 aylık değişim) (30 $); Çin sanayi üretimi (12 aylık değişim) (29 $); Eaton (elektrik ekipmanı) (3 aylık değişim) (15 $); ABD ÜFE (12 aylık değişim) (6 $); USD/JPY (3 aylık değişim) (5 $); USD/PEN (Peru) (3 aylık değişim) (3 $)

**Aşağı çekenler:** Fed faizi (12 aylık değişim) (-24 $); Altın (12 aylık değişim) (-12 $); ABD 10 yıllık faiz (12 aylık değişim) (-10 $); NVIDIA (12 aylık değişim) (-9 $); Peru dizel fiyatı (12 aylık değişim) (-9 $); Southern Copper (12 aylık değişim) (-7 $); Tesla (12 aylık değişim) (-7 $); ABD ÜFE (3 aylık değişim) (-6 $)

## Sınırlar

- Veriler güncel sürümdür; geçmişte ilk yayımlanan değerler değildir.
- Haber verisi 2017'den başlar.
- Yatırım tavsiyesi değildir.
