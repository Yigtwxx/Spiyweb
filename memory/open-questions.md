---
name: open-questions
description: Kalan açık sorular — 2026-08-13 güncellemesi; #12 (ChunkRef-vs-Node) adım 5'te kapandı, 11 başlık açık
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
`max_hop` üst sınırları taşma koruması olarak gerekli. Kodda **geçici
varsayılan** olarak `max_hop = 6`, `max_nodes = 512` seçildi (bir değer yazmak
zorunluydu); gerçek değerler ilk ölçümlerden sonra netleşecek. Fren devreye
girdiğinde sonuç `stop_reason` alanında görünür — sessiz kırpma yok.

**5. Chunk boyutu.**
Yalnızca "300-500 token" olarak telaffuz edildi ([[alternative-directions]]),
karar verilmedi.

**6. Profil parametre değerleri.**
`precise` / `explore` / `compare` için damping, eşik ve seed genişliği
değerleri ([[query-profiles-and-negative-seeds]]) atanmadı.

**7. Atom kütlesi formülü.**
"Uzunlukla orantılı" ötesinde formül yok ([[node-layers-and-mass]]).

**8. Novelty@k alaka yargısı.**
`top-k`'nın hiç getirmediği bir düğümün "yine de ilgili" sayılma yöntemi
tanımlanmadı ([[phase1-settings]]).

**9. Öğrenen katman unutma katsayısı.**
Zorunlu ([[learned-layer-hebbian]]) ama değeri seçilmedi.

**10. NLI model seçimi ve aday çifti eşiği.**
[[contradiction-detection]] kararının uygulama detayları.

**11. Negatif önerme (polarite) tespit yöntemi.**
[[negative-knowledge-atoms]] için: olumsuz önermeler NLI hattında mı, ayrı bir
polarite sınıflandırıcısıyla mı tespit edilecek? Faz 1 ölçümünden sonra,
ablation uygulamasından önce netleşmeli.

Not (adım 3): `SemanticEdgeConfig.k = 5`, `min_similarity = 0.0` ve
`StructuralEdgeConfig` alt-ağırlıkları (1.0 / 0.6 / 0.0) **geçici el
değerleridir**; katman ağırlıklarıyla aynı grid-search planına dahiller
([[phase1-settings]]).

Not (adım 4): `EntityEdgeConfig.max_df_ratio = 0.5`,
`EntityExtractionConfig.min_entities = 1` ve varsayılan label seti de aynı
statüde — geçici el değerleri, grid-search havuzunda.

## Kapanan başlıklar

**2026-08-13 (uygulama adım 5) — eski #12 kapandı:** `Node` pozisyon
alanlarını EMMEDİ; `ChunkRef` index-zamanı girdi sözleşmesi olarak kaldı
(sahibi seçti). `nodes/chunks.py` her unit için Node + ChunkRef çiftini
birlikte üretir ve iki yapıyı adım adım tutarlı tutan tek yer chunker'dır.
Gerekçe: yayılmanın hiç okumadığı alanlar çekirdek şemasına girmesin
(boundary rule 2). Ayrıca **entity kenar ağırlığı** karara bağlandı:
nadirlik ağırlıklı `Σ 1/df(e)` — [[hybrid-edge-layers]].

**2026-08-13 (uygulama adım 2):** node veri modeli kapandı — `core/graph.py`
içinde `Node` şeması: `id`, `layer` (chunk/proposition), `source_id` (doküman
bazlı oy), `length` (kütle formülünün ham girdisi; formülün kendisi hâlâ açık,
bkz. madde 7), `timestamp` (**UTC epoch float** — tazelik yalnız eşitlik bozucu
olduğundan tek ihtiyaç toplam sıralama; `datetime` aware/naive tuzağı ve ISO
format hassasiyeti nedeniyle elendi), `cluster_id`, D34 `polarity` (+1/−1,
varsayılan +1). Kenar katmanları `LayerWeights` config'iyle ağırlıklı toplamla
birleşiyor; ağırlık 0.0 = katman kapalı (ablation anahtarı).

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
