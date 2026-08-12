---
name: open-questions
description: Kalan açık sorular — 2026-08-12 gap analiziyle güncellendi; termal reset kapandı, veri modeli ve parametre değerleri gibi yeni başlıklar eklendi
metadata:
  type: project
---

Önceki 11 teknik varsayılan tek tek soruldu ve karara bağlandı
([[phase1-settings]]). 13 tasarım ekseni de kapandı. Geriye kalanlar, ancak kod
yazılırken veya ilk ölçümden sonra netleşebilecek konular.

## Kalan açık konular

**1. Hedef karışımının tek sayıya indirgenmesi.**
Hedef `%65 multi-hop doğruluk + %35 Novelty@k`. İki metriğin ölçekleri farklı;
ağırlıklı toplam almadan önce normalizasyon kuralı belirlenmeli. Aksi hâlde "%65"
ve "%35" nominal kalır, gerçek etki farklı olur.

**2. Önerme çıkarımının maliyeti.**
İki katmanlı düğüm yapısı ([[node-layers-and-mass]]) index anında LLM çağrısı
gerektiriyor. MuSiQue sabit ve sınırlı bir corpus olduğu için 1. fazda
yönetilebilir; ama kendi dokümanlarına uygulandığında maliyet ölçülmeli ve
belgelenmeli. Gerekirse "önerme katmanı opsiyonel" hale gelir.

**3. Uyarlanabilir tekrar eşiğinin formülü.**
Karar "uyarlanabilir" ([[phase1-settings]]) ama hesaplama yöntemi seçilmedi —
aktif düğümlerin benzerlik dağılımında yüzdelik dilim mi, sapma temelli mi?
Hata ayıklaması zor bir mekanizma olduğu için UI'da hesaplanan eşik mutlaka
görünmeli.

**4. Emniyet freni değerleri.**
Durma göreli enerji eşiğiyle ([[stopping-and-freshness]]); `max_nodes` ve
`max_hop` üst sınırları taşma koruması olarak gerekli ama değerleri henüz
belirlenmedi. İlk ölçümlerden sonra netleşecek.

**5. Node veri modeli.**
Doküman bazlı oy için kaynak ID, tazelik eşitlik bozucusu için timestamp,
katman etiketi ve tema kümeleri için cluster ID gerekiyor — şema hiç yazılmadı.

**6. Chunk boyutu.**
Yalnızca "300-500 token" olarak telaffuz edildi ([[alternative-directions]]),
karar verilmedi.

**7. Profil parametre değerleri.**
`precise` / `explore` / `compare` için damping, eşik ve seed genişliği
değerleri ([[query-profiles-and-negative-seeds]]) atanmadı.

**8. Atom kütlesi formülü.**
"Uzunlukla orantılı" ötesinde formül yok ([[node-layers-and-mass]]).

**9. Novelty@k alaka yargısı.**
`top-k`'nın hiç getirmediği bir düğümün "yine de ilgili" sayılma yöntemi
tanımlanmadı ([[phase1-settings]]).

**10. Öğrenen katman unutma katsayısı.**
Zorunlu ([[learned-layer-hebbian]]) ama değeri seçilmedi.

**11. NLI model seçimi ve aday çifti eşiği.**
[[contradiction-detection]] kararının uygulama detayları.

## Kapanan başlıklar

**2026-08-12 oturumunda kapananlar:** çelişki tespiti → index anında NLI
([[contradiction-detection]]); durma eşiği → göreli %15
([[stopping-and-freshness]]); termal reset → hibrit
([[conversation-thermal-memory]]); Faz-1 kapısı → iki baseline + HippoRAG
rapor kıyası, LLM sağlayıcı, platform ve paketleme → [[phase1-settings]].

Ayrıntı ve gerekçeler ilgili dosyalarda: [[learned-layer-hebbian]],
[[conversation-thermal-memory]], [[consolidation-pruning]],
[[confidence-and-abstention]], [[corpus-gap-detection]], [[output-contract]],
[[query-profiles-and-negative-seeds]], [[contradiction-handling]],
[[multi-seed-colors]], [[node-layers-and-mass]], [[stopping-and-freshness]],
[[phase1-settings]].

Değerlendirilip ertelenen yönler: [[alternative-directions]] (D — öğrenilmiş
sönümleme, 2. faz).
