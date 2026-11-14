# Bakır tahmini — son veri: 1-10 Kas 2026

Oluşturulma 2026-11-14T16:32:49+00:00 UTC · kod `2b147e6` · 542.8 sn

Son 10 günlük dönem ortalaması: **14.111 USD/t**

## Tahmin (10 günlük dönem ortalamaları)

| Dönem | Tahmin USD/t | Değişim | Geçmiş hata (MAPE) | Yön isabeti |
|---|---:|---:|---:|---:|
| 11-20 Kas 2026 | 14.006 | -0,7% | %1,7 | %68 |
| 21-30 Kas 2026 | 14.048 | -0,4% | %3,2 | %60 |
| 1-10 Ara 2026 | 14.129 | 0,1% | %4,2 | %51 |
| 11-20 Ara 2026 | 14.131 | 0,1% | %4,9 | %50 |
| 21-31 Ara 2026 | 14.142 | 0,2% | %5,6 | %51 |
| 1-10 Oca 2027 | 14.302 | 1,4% | %6,4 | %50 |
| 11-20 Oca 2027 | 14.364 | 1,8% | %6,9 | %51 |
| 21-31 Oca 2027 | 14.392 | 2,0% | %7,5 | %50 |
| 1-10 Şub 2027 | 14.391 | 2,0% | %8,0 | %56 |
| 11-20 Şub 2027 | 14.433 | 2,3% | %8,4 | %57 |
| 21-28 Şub 2027 | 14.452 | 2,4% | %8,9 | %57 |
| 1-10 Mar 2027 | 14.436 | 2,3% | %9,2 | %56 |
| 11-20 Mar 2027 | 14.472 | 2,6% | %9,6 | %57 |
| 21-31 Mar 2027 | 14.246 | 1,0% | %9,9 | %51 |
| 1-10 Nis 2027 | 14.340 | 1,6% | %10,4 | %54 |
| 11-20 Nis 2027 | 14.449 | 2,4% | %10,9 | %53 |
| 21-30 Nis 2027 | 14.276 | 1,2% | %11,2 | %51 |
| 1-10 May 2027 | 14.316 | 1,5% | %11,6 | %55 |
| 11-20 May 2027 | 14.431 | 2,3% | %12,1 | %54 |
| 21-31 May 2027 | 14.320 | 1,5% | %12,6 | %53 |
| 1-10 Haz 2027 | 14.433 | 2,3% | %13,0 | %51 |
| 11-20 Haz 2027 | 14.238 | 0,9% | %13,3 | %49 |
| 21-30 Haz 2027 | 14.554 | 3,1% | %13,5 | %50 |
| 1-10 Tem 2027 | 14.606 | 3,5% | %13,8 | %49 |
| 11-20 Tem 2027 | 14.554 | 3,1% | %13,9 | %51 |
| 21-31 Tem 2027 | 14.495 | 2,7% | %13,9 | %52 |
| 1-10 Ağu 2027 | 14.517 | 2,9% | %14,0 | %52 |
| 11-20 Ağu 2027 | 14.414 | 2,1% | %16,3 | %45 |
| 21-31 Ağu 2027 | 14.536 | 3,0% | %16,5 | %46 |
| 1-10 Eyl 2027 | 14.469 | 2,5% | %16,6 | %46 |
| 11-20 Eyl 2027 | 14.558 | 3,2% | %16,4 | %49 |
| 21-30 Eyl 2027 | 14.403 | 2,1% | %16,5 | %51 |
| 1-10 Eki 2027 | 14.213 | 0,7% | %16,4 | %56 |
| 11-20 Eki 2027 | 14.073 | -0,3% | %16,7 | %57 |
| 21-31 Eki 2027 | 14.049 | -0,4% | %16,9 | %58 |
| 1-10 Kas 2027 | 14.025 | -0,6% | %17,2 | %56 |

## Geriye dönük karne

'Fiyat aynı kalır' tahminine göre hatadaki azalma (%), pozitif = daha iyi.

| Model | 1 ay | 3 ay | 6 ay | 12 ay |
|---|---:|---:|---:|---:|
| ML tahmini (3 modelin ortalaması) | 0,3 | -5,6 | -1,9 | -3,1 |
| ElasticNet (bakır + makro) | 2,0 | -6,2 | -1,4 | -19,8 |
| LightGBM (bakır + makro + haber) | -3,6 | -15,8 | -7,5 | 9,4 |
| ElasticNet (bakır + haber) | 5,1 | 9,1 | 9,9 | -21,3 |
| LME vadeli eğrisi | 6,2 | 1,7 | 2,0 | 1,7 |

## Haber duyarlılığı katkısı

Haberli ElasticNet'in habersiz ElasticNet'e göre hata azalması (%), aynı dönemde.

| Ufuk | Gözlem | Katkı | p |
|---|---:|---:|---:|
| 1 ay | 158 | 0,3 | 0,460 |
| 3 ay | 147 | 18,8 | 0,114 |
| 6 ay | 129 | 7,4 | 0,137 |
| 12 ay | 93 | -26,2 | 0,663 |

## Tahmini en çok etkileyen girdiler (3 ay)

**Yukarı itenler:** Japonya 10 yıllık faiz (12 aylık değişim) (52 $); Çin sanayi üretimi (12 aylık değişim) (44 $); Çin sabit sermaye yatırımı (12 aylık değişim) (32 $); USD/JPY (3 aylık değişim) (15 $); Doğal gaz (3 aylık değişim) (6 $); Euro bölgesi ÜFE (3 aylık değişim) (5 $); Kore 10 yıllık faiz (12 aylık değişim) (4 $); LME alüminyum stoku (3 aylık değişim) (3 $)

**Aşağı çekenler:** Fed faizi (12 aylık değişim) (-33 $); Çin gayrimenkul yatırımı (12 aylık değişim) (-24 $); ABD 2 yıllık faiz (12 aylık değişim) (-15 $); Euro bölgesi bakır ÜFE (3 aylık değişim) (-13 $); Çin ÜFE (12 aylık değişim) (-12 $); Peru dizel fiyatı (12 aylık değişim) (-10 $); Bakır oynaklığı (-7 $); Bakır 3 aylık getiri (-7 $)

## Sınırlar

- Veriler güncel sürümdür; geçmişte ilk yayımlanan değerler değildir.
- Haber verisi 2017'den başlar.
- Yatırım tavsiyesi değildir.
