## Yöntem

**Hedef.** LME bakır nakit fiyatının aylık ortalaması (USD/t), 1–12 ay sonrası. Model fiyatı değil,
log getiriyi tahmin eder: `y_h(t) = log(P[t+h] / P[t])`.

**Tek kural: her tahmin, o gün bilinebilecek bilgiyle yapılır.**

- Her seri için açık bir yayın gecikmesi var. Piyasa verileri 0, risk endeksleri 1–2, resmi makro
  istatistikler 2 ay geriden gelir. Ayın son değeri henüz yayımlanmadıysa en fazla 2 ay eski değer taşınır.
- Süren ay hiçbir zaman kullanılmaz. Tahmin başlangıcı her zaman son tamamlanmış aydır.
- Bir ufuk için model, yalnızca başlangıç anında **gerçekleşmiş** hedeflerle eğitilir (`s + h ≤ t`).
- Makro faktörler (PCA) yalnızca başlangıca kadarki verilerle hesaplanır.
- Üretim tahmini ayrı bir kod değildir. Geriye dönük testin son adımıdır ve aynı fonksiyonla üretilir.

**Uzmanlar.**

| Uzman | Ne yapar |
|---|---|
| Rastgele yürüyüş | Gelecek ayların ortalaması bu ayın ortalamasına eşit |
| Ay sonu fiyatı | Gelecek ayların ortalaması bu ayın **son işlem günü** fiyatına eşit |
| Bakırın kendi sinyalleri | Getiriler, oynaklık, contango, LME stokları → Ridge |
| Makro faktörler | Yukarıdakiler + 55 makro/piyasa serisinden 5 PCA faktörü → Ridge |
| LightGBM | Aynı girdiler, sığ ağaçlar ve güçlü düzenlileştirme |

**Kombinasyon: ağırlık kazanılır.** Bir uzman, yalnızca son değerlendirme penceresinde ay sonu fiyatı
kuralından daha düşük hata yaptıysa ağırlık alır. Ağırlık, ters MSE ile hesaplanır. h aylık tahminler
birbiriyle örtüştüğü için n tahmin yaklaşık n/h bağımsız sonuç taşır. Bu yüzden en az 12 bağımsız
sonuç (12·h gerçekleşmiş tahmin) birikene kadar ağırlığın tamamı ay sonu fiyatı kuralında kalır.

**Belirsizlik.** Aralıklar, kombinasyonun geçmişte gerçekleşmiş hatalarının dağılımından gelir
(boş bir normal dağılım varsayımı yok). Dağılım medyanı nokta tahmine eşitlenir. Böylece 2015–2026
yükselişinden gizli bir yön (drift) aralıklara karışmaz. Kalibrasyon ayrıca test edilir: %80 aralığın
gerçekleşmelerin ~%80'ini kapsaması beklenir.

**Değerlendirme.** Ufuk başına ~130–141 geriye dönük tahmin var (Ocak 2015 – Ekim 2026). Ölçütler:
MAPE, RMSE, sapma, yön isabeti ve rastgele yürüyüşe göre beceri. Anlamlılık, Diebold–Mariano testiyle
ölçülür (Newey–West, h−1 gecikme, HLN düzeltmesi, tek yönlü).

**Sınırlar.** Veriler güncel sürümdür; geçmişte ilk yayımlanan değerler değildir. Yayın gecikmeleri
gerçek yayın takviminden değil, ihtiyatlı varsayımdan gelir. Uzun ufuklarda bağımsız gözlem sayısı
azdır. Bu çalışma yatırım tavsiyesi değildir.
