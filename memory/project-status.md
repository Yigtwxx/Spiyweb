---
name: project-status
description: Tasarım bitti; uygulama sürüyor — adım 1-3 (çekirdek, veri modeli, structural+semantic), adım 4 (entity katmanı: kurucu + spaCy/LLM hibrit çıkarım) ve adım 5 (chunker + gömme + FAISS deposu) tamamlandı
metadata:
  type: project
---

**Durum (2026-08-12): tasarım tamamlandı, uygulama başladı.**

## Uygulama adım 1 (tamamlandı)

Sahibi "çok basit, minicik bir adım, GitHub'da da görünsün" dedi; en küçük
anlamlı dikey dilim seçildi: **saf çekirdek + kanonik izin regresyon testi**.

- `src/spiyweb/config.py` — `PropagationConfig` (seed_energy 10.0, damping 0.60,
  threshold_ratio 0.15, max_hop 6, max_nodes 512) + aralık doğrulaması
- `src/spiyweb/core/graph.py` — seyrek ağırlıklı komşuluk; **0.0 ağırlık
  bastırılmış kenar** demek (dedup sonrası), negatif ağırlık reddediliyor
  (negatif yük ayrı mekanizma)
- `src/spiyweb/core/propagate.py` — çarpımsal sönümleme, oranlı bölüşüm,
  toplamalı birikim, göreli eşik, `max_hop`/`max_nodes` emniyet frenleri
- `tests/` — 25 test; `CLAUDE.md` §2.6 kanonik izi birebir sabitlendi
- `.github/workflows/ci.yml` — ubuntu + macOS + windows matrisi, ruff + pytest

İki tasarım detayı kod yazılırken netleşti ve dokümanlara işlendi:

1. **Eşik birikmiş enerjiye uygulanır**, tek tek katkılara değil. Katkı bazında
   uygulansaydı `D` yalnız 1.75 alırdı; 1.125 + 1.75 = 2.875 birikimi ölürdü —
   yani converging evidence, yani projenin bütün değer önerisi.
2. **Zaten aktif komşular paydadan düşer**, enerji geri sızmaz. Bastırılmış
   kenarın renormalizasyonuyla aynı kural; `F = 1.725` ancak böyle çıkıyor.

Çekirdeğin **sıfır çalışma zamanı bağımlılığı** var — `core/` saflığı kuralı
paketleme seviyesinde de uygulandı.

## Uygulama adım 2 (tamamlandı, 2026-08-13)

Düğüm/kenar veri modeli — açık soru #5 kapandı:

- `core/graph.py` — `Node` şeması: `id`, `layer` (chunk/proposition),
  `source_id` (doküman bazlı oy), `length` (kütle formülünün ham girdisi;
  kütle henüz propagate'e **bağlı değil**), `timestamp` (UTC epoch float,
  yalnız eşitlik bozucu), `cluster_id`, D34 `polarity` (+1/−1). Graf artık
  opsiyonel `node_data` registry taşıyor (`nodes` property'siyle ad çakışmasın
  diye bu ad); kısmi metadata serbest, yinelenen id reddediliyor.
- `config.py` — `EdgeLayer` + `LayerWeights` (semantic .5 / entity 1.0 /
  structural .3 / learned **0.0 = kapalı**; ağırlık 0.0 katmanın ablation
  anahtarı). Faz 1 el ağırlıklarının tek evi burası.
- `Graph.from_layers` — katman başına kenar listelerini **ağırlıklı toplamla**
  tek komşuluğa birleştirir (ortalama değil — iki katmanın onayladığı kenar
  güçlenir, additive evidence). `from_edges` gövdesi `_build_adjacency`'ye
  çıkarıldı; negatif ret, undirected aynalama ve bastırılmış-kenar (0.0)
  sözleşmesi birleştirmeye miras. Katman ağırlığı 0.0 → kenarlar VE yalnız o
  katmanın andığı düğümler grafa hiç girmez.
- 28 yeni test (`test_node.py`, `test_layers.py`) — toplam 53; en kritiği:
  `from_layers` ≡ önceden birleştirilmiş `from_edges` altında `propagate`
  birebir aynı sıralamayı veriyor (birleştirme çekirdeğe görünmez ön-işlem).
  Kanonik §2.6 izi dokunulmadan yeşil.

## Uygulama adım 3 (tamamlandı, 2026-08-13)

İlk iki kenar kurucu — `edges/` paketi açıldı, `core/` DOKUNULMADI
(boundary rule 2):

- `edges/structural.py` — `ChunkRef` girdi sözleşmesi (`id`, `source_id`,
  `position`, `section_id`; Node'a bilerek konmadı — bkz. açık soru 12) +
  `build_structural_edges`. Üç ilişki iç içe (komşu ⊂ bölüm ⊂ doküman),
  çift başına **max kazanır, toplanmaz**; kapalı ilişki (0.0) çifti hiç
  üretmez — 0.0 dedup-bastırma sözleşmesine ayrılı. Komşuluk, pozisyon
  sıralı düzende ardışıklık (boşluklara dayanıklı).
- `config.py` — `StructuralEdgeConfig` (adjacent 1.0 / same_section 0.6 /
  **same_document 0.0 = kapalı**, sahibi seçti: doküman-içi clique hub
  cezasını besler; kapalıyken O(n²) tarama hiç yapılmaz) +
  `SemanticEdgeConfig` (k=5, min_similarity=0.0). Değerler geçici, grid
  search'e tabi.
- `edges/semantic.py` — `build_semantic_edges`: saf stdlib cosine
  (numpy YOK, `dependencies = []` duruyor; FAISS adım 5), **union kNN**
  (mutual reddedildi — meşru ilk temas noktalarını budar), üretim koşulu
  `sim > min_similarity` kesin `>` — negatif cosine ve tam-0.0 grafa asla
  sızmaz. Sınır beraberliği id ile kırılır (3 OS determinizmi). Sıfır-norm
  vektör sessizce atlanmaz, `ValueError` (bozulma gizlenmez).
- 40 yeni test (`test_structural_edges.py`, `test_semantic_edges.py`,
  `test_edges_integration.py`) — toplam 93. Entegrasyon testi katmanlı-hibrit
  iddiasını minyatürde kanıtlıyor: yalnız yapısal komşu olan chunk aktive
  oluyor, `LayerWeights(structural=0.0)` ablation'ında sönüyor; kurucular
  önceden birleştirilmiş grafa denk (çekirdeğe görünmez ön-işlem).
  Kanonik §2.6 izi dokunulmadan yeşil.

## Uygulama adım 4 (tamamlandı, 2026-08-13)

Entity katmanı — tam hibrit, sahibi kapsamı açıkça seçti:

- `edges/entity.py` — `build_entity_edges`: **nadirlik ağırlığı** — ortak
  varlık başına katkı `1/df(e)`, çift ağırlığı = katkıların toplamı (sahibi
  seçti; ham sayı hub-eğilimli, Jaccard bilgi-yoğun chunk'ı cezalandırıyordu).
  Ortak varlık yoksa çift HİÇ üretilmez. Ters indeksle O(mentions + Σ df²);
  sabit sıralı iterasyon = 3 OS bit-özdeş toplamlar. Kurucu saf kaldı —
  varlık string'leri opak anahtar.
- `config.py` — `EntityEdgeConfig.max_df_ratio = 0.5`: 1/df stopword-varlığın
  *ağırlığını* sınırlar ama *kenar sayısını* sınırlamaz; bu knob clique freni
  (same_document'i kapatan hub gerekçesi). `min_shared` bilinçli YOK.
- `llm.py` — `OpenAICompatClient`: tek kod yolu Ollama/Groq/OpenRouter
  (OpenAI-uyumlu `/chat/completions`), **stdlib urllib** (requests yok).
  API anahtarı yalnız env değişkeni ADIYLA (`api_key_env`); değer hiçbir
  mesaja/loga girmez. Retry + üstel backoff, enjekte edilebilir
  transport/sleep. `core/` bu modülü asla import etmez.
- `entities.py` — hibrit çıkarım `edges/` DIŞINDA (paket sözleşmesi:
  kurucular model çağırmaz; çıkarım kenar değil chunk-başına veri üretir).
  spaCy çoğunluk, `min_entities`'ten az bulunan chunk LLM'e (`llm=None` veya
  `min_entities=0` = ablation). Protocol'lerle her şey sahtelenebilir;
  `normalize_entity` iki yolun ortak anahtarı. Prompt `prompts.py`'de
  (mantıktan ayrı), satır-başına-varlık düz metin — JSON/eval yok.
- `EntityExtractionConfig.labels` iki şemanın birleşimi (WikiNER + OntoNotes)
  — model değişince sessiz sıfır-çıkarım olmasın; sayısal/zamansal etiketler
  bilinçli dışarıda ("2019" hop yakıtı değil).

## Uygulama adım 5 (tamamlandı, 2026-08-13)

Chunker + gömme + FAISS deposu; **açık soru #12 kapandı**:

- `nodes/chunks.py` — `TextUnit`/`DocumentInput`/`Chunk`; chunker her unit
  için **Node + ChunkRef çifti** üretir (sahibi seçti: Node yalın kalır,
  ChunkRef index-zamanı girdi sözleşmesi olarak yaşar). Id şeması
  `{source_id}:{position}`. Yeniden bölme YOK — MuSiQue paragraf verir; açık
  soru #5 (chunk boyutu) bu modülle kapanmadı.
- `embedding.py` — e5 önek sözleşmesi API'ye gömülü (`embed_queries` =
  "query: ", `embed_passages` = "passage: "; öneksiz metot yok). Her zaman
  L2-normalize → depo iç çarpımı = cosine. Cihaz: saf `resolve_device`
  (CUDA → MPS → CPU) + ince `detect_device` — CI'da torch gerekmez.
- `store.py` — `VectorStore`: **IndexFlatIP** (exact; Faz 1 ölçeğinde
  yaklaşıklığa gerek yok). **Tek gerçek kaynak vektörler**: save tek `.npz`
  (ids+vectors, dosya-handle formu numpy'ın sessiz `.npz` ekleme büyüsünü
  atlar), load indeksi yeniden kurar — FAISS serialize baytları sürüm/platform
  borcu olurdu.
- Paketleme — granüler extras (sahibi seçti): `store` = numpy+faiss-cpu,
  `embed` = sentence-transformers, `entity` = spacy, `index` = birleşim;
  `dependencies = []` DURUYOR. CI artık `--extra store --extra entity` kurar
  (FAISS gerçek test edilir; torch dışarıda — glue sahte encoder'la kaplı).
- 82 yeni test (entity_edges 15, llm 17, entities 13, chunks 11, embedding 13,
  store 13 + index entegrasyonu) — **toplam 175** (1 skip: spaCy kuruluyken
  ImportError-ipucu yolu erişilmez). `test_index_integration.py` ana iddiayı
  minyatürde kanıtlıyor: cevap chunk'ı başka dokümanda ve semantik görünmez;
  yalnız nadir ortak varlıkla aktive oluyor, `LayerWeights(entity=0.0)`
  ablation'ında sönüyor. Kanonik §2.6 izi dokunulmadan yeşil.

**Sıradaki:** eval harness (MuSiQue yükleyici + `top-k` ve iteratif
baseline'lar + metrik raporu) → **ilk ölçüm**.
Dedup, renkli çok tohum ve çelişki yönetimi baseline karşılaştırmasından sonra.

Güncelleme (2026-08-12): gap analizi yapıldı; 8 karar kapandı ve dokümanlara
işlendi (D26-D33, spec §8.1b): NLI ile çelişki tespiti, göreli eşik (%15),
çift-baseline kapısı, HippoRAG rapor kıyası, lokal-öncelikli LLM sağlayıcı,
`src/` + `evaluation/` + `spiyweb[ui]` paketleme, hibrit termal reset,
3 işletim sistemi desteği. Bayat dosyalar düzeltildi. README / LICENSE /
CONTRIBUTING eklendi ve repo GitHub'a açıldı. **Başlama kuralı aynen geçerli.**

Aynı gün, özgünlük oturumu: 4 yeni karar (D34-D37) — [[negative-knowledge-atoms]]
(tasarım şimdi, uygulama Faz 1 ölçümü sonrası ablation),
[[explained-abstention]] (Faz 1), [[supersession-vs-contradiction]] (Faz 2),
[[corpus-lint]] (Faz 2 ürün adayı / B planı).

Tüm tasarım kararları alındı ve yazıya geçirildi:
- 9 çekirdek karar (D1-D9) + 16 genişletilmiş karar (D10-D25) → `CLAUDE.md` §2 ve
  `docs/specs/2026-08-10-spiyweb-design.md`
- 1. fazın teknik ayarları (benchmark, modeller, ortam, lisans) →
  [[phase1-settings]]
- Gerekçeler ve elenen alternatifler → bu `memory/` klasörü
- Kalan 5 açık soru → [[open-questions]] (hiçbiri başlangıcı bloklamıyor)

## Başlama kuralı (2026-08-12'de kalktı)

Kural şuydu: sahibi açıkça başlama komutu vermeden hiçbir kod yazılmayacak.
**Komut geldi ve ilk adım atıldı**; kural artık geçmiş kayıt. Yerine geçen tek
sıralama kuralı: baseline karşılaştırması diğer her şeyin değerini belirlediği
için önce **çalışan iskelet** (graf + yayılma + eval harness + `top-k`),
sonra dedup / renkli çok tohum / çelişki yönetimi / UI ([[roadmap-and-gates]]).

Hatırlatma: her mekanizma config'ten tek tek kapatılabilir olmak zorunda; ablation
bu projenin kendini kanıtlama yöntemi.
