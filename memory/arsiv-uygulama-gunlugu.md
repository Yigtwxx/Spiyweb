---
name: arsiv-uygulama-gunlugu
description: ARŞİV — Faz 1 uygulama günlüğünün tam anlatımı (adım 1-8, ölçüm turları 1-12, terfi, seed-123 onayı); güncel özet project-status.md'de
metadata:
  type: project
---

**Bu dosya arşivdir (2026-08-14'te donduruldu).** Faz 1 uygulama ve ölçüm
kampanyasının tam, kronolojik anlatımı — gerekçeleriyle. Güncel durum ve
kısa tarihçe: [[project-status]]. Bu dosyaya yeni kayıt eklenmez.

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

## Uygulama adım 6 (tamamlandı, 2026-08-14)

`retrieve.py` + MuSiQue eval harness; **açık soru #1 ve #8 kapandı**
(formüller [[open-questions]] "Kapanan başlıklar" altında):

- `retrieve.py` — uçtan uca yapıştırıcı: `SeedSource` Protocol (`VectorStore`
  yapısal karşılar, faiss'siz import), similarity ≤ 0 temaslar düşer, hiçbiri
  kalmazsa raise; sonuç **bilerek kısmi** (`seeds` + `propagation` +
  `Confidence` üçlüsü; §2.5'in kalanı baseline sonrası toplamalı büyür —
  alan seti testle pinli). `RetrievalConfig(seed_width=5, propagation=...)` —
  D13 profilleri ileride bu sınıfın fabrikası.
- `store.py::build_semantic_edges_fast` — saf kurucunun FAISS'li ikizi
  (union kNN, kesin `>`, kanonik sıra; seçim float32, ağırlık float64);
  saf kurucu semantik oracle olarak duruyor, eşdeğerlik testli. Corpus
  ölçeğinde (11,6k×1024) saf O(n²) Python saatler sürerdi.
- `evaluation/` — `datasets.py` (HF'ten düz HTTPS jsonl, stdlib urllib;
  **leksikografik sıralı id üstünde seed'li örnekleme**; `(title, text)`
  dedup havuzlama → `d{i:05d}` corpus id'leri; gold + köprü-gold haritalama),
  `metrics.py` (support recall, Novelty, bridge recall, S@k — hepsi el-izli
  testli), `baseline.py` (top-k + IRCoT-sadık iteratif: ilk cümle = sonraki
  sorgu, stop-phrase, union sıralaması; `max_steps=0` = düz top-k ablation),
  `cache.py` (**prompt-SHA256 deterministik LLM disk cache** — hibrit entity
  fallback ve iteratif baseline aynı mekanizmadan; crash-resume +
  tekrarlanabilirlik), `index.py` (aşama-başına artifact + varsa-atla resume;
  graf yükte `from_layers` ile yeniden birleşir → `LayerWeights` ablation'ı
  re-index'siz; pasaj = `title\ntext`, embed + çıkarım aynı string'i görür;
  LLM'e gidecek chunk sayısı çağrıdan önce loglanır), `run.py` (CLI:
  download/index/evaluate/report/all; `--no-entity-llm`, `--skip-iterative`;
  `per_query.jsonl` + `results.json`; markdown rapor: üç-sistem tablosu,
  hop-katmanlı S, **stop_reason dağılımı** (açık soru #4'ün verisi), HippoRAG
  referans satırı R@2 .409 / R@5 .519 "reported, not reproduced" — D29).
- Sahibi seçti: **1.000 soru HippoRAG-kıyas rejimi** (sample_seed=42,
  `sample_size=0` = tam dev), **tam hibrit entity varsayılan** (ablation
  bayrağı ters: `--no-entity-llm`), S@5 birincil sayı.
- 58 yeni test (retrieve 10, fast-semantic 10, datasets 11, metrics 10,
  baseline 11, e2e run 7 dahil minyatür kanıt: **web S@2 .825 > top-k .325**
  — köprü dokümanı yalnız entity hop'uyla) — **toplam 233 + 1 skip**.
  pyproject/CI değişmedi. Kanonik §2.6 izi dokunulmadan yeşil.

## İlk ölçüm (2026-08-14, varsayılan config — NEGATİF/DÜZ sonuç)

Koşuldu: 1.000 soru, 11.835 pasaj (HippoRAG'ın 11.656'sıyla aynı rejim),
llama3.1:8b (Ollama), e5 CUDA'da index / CPU'da evaluate. Süre: index ~4 dk
(yalnız 27 chunk LLM'e düştü), evaluate ~1,5 saat (~3.450 LLM çağrısı, hepsi
cache'te). Sonuç tabloları `data/musique/results.json`, görseller repo kökünde
(`olcum_1_sistem_kiyasi.png`, `olcum_2_hop_kirilimi.png`, `olcum_3_teshis.png`).

**Sayılar (S@5):** top-k **0.309** | web **0.309** (birebir aynı!) | iteratif
**0.463**. Support R@5: .475/.475/.622; HippoRAG rapor R@5 .519. **Web 1.000
sorgunun 998'inde hop 0'da öldü** — web = 5 seed = top-5, Novelty 0.000.

**Teşhis (per_query.jsonl'den doğrulandı, iki bileşik kök neden — ikisi de
önceden belgelenmiş riskler):**
1. **Eşik–seed genişliği çatışması:** e5 benzerlikleri dar bantta (~0.77-0.91,
   ort. 0.81) → 5 seed'e bölünen 10.0 enerji seed başına ~2.0; ileri giden en
   fazla 2.0×0.6=1.2 < eşik 1.5. Tek yoldan aktivasyon **yapısal olarak
   imkânsız**; kanonik örnek 2 seed'le çalışıyordu, 5 seed + %15 göreli eşik
   birlikte matematiksel olarak ölü doğuyor.
2. **Entity clique şişmesi:** `max_df_ratio=0.5` → 2.497.920 kenar, ortalama
   derece 438, maks 3.656 — 1.2'lik enerji yüzlerce komşuya bölünüp toza
   dönüyor (bilinen hub cezası, `known-risks` #2).

**Sıradaki: grid/ablation turu** (zaten planlıydı — "elle başla, küçük grid
search"). Oynatılacaklar: `threshold_ratio` (0.15 → ~0.03-0.05),
`max_df_ratio` (0.5 → ~0.02-0.05); gerekirse seed benzerliklerinin
keskinleştirilmesi (dar e5 bandı bölüşümü düzleştiriyor — `sim ** alpha`
seçeneği `known-risks`'te zaten var). Maliyet düşük: entity kenarları
`edges_entity.json` silinip index'in resume'uyla ~1 dk'da yeniden kurulur;
iteratif baseline cache'ten bedava döner; evaluate ~10 dk. Dedup, renkli çok
tohum ve çelişki yönetimi hâlâ baseline karşılaştırması netleşince.

## Grid/ablation turu (2026-08-14 — üç tur, plato ~.35)

Üç grid koşuldu (scratchpad script'leri, kütüphane API'siyle; repo artifact'ları
ezilmedi; top-k/iteratif kolonları ilk ölçümün `per_query.jsonl`'inden alındı):

1. **Tur 1 — threshold {.05,.03} × max_df {.05,.02}:** 4/4 kombinasyonda
   S@5 = **.309** (top-k'ya birebir eşit, novelty 0). Yayılım canlandı
   (thr .03'te sorguların ~%47'si hop 1) ama sıralama değişmedi. **Yeni kök
   neden:** `seed_width=5 = k=5` — seed'ler S@5 penceresinin tamamını
   dolduruyor; hop-1 düğümü (enerji ≤ 1.2'nin kırıntısı) seed'i (~2.0)
   yapısal olarak geçemiyor. Umut sinyali: web recall@10 .475→.509,
   novelty@10 .023 — mekanizmanın nabzı ilk kez attı.
2. **Tur 2 — seed_width {2,3} × thr {.03,.01} (df .02 sabit):** sayı İLK KEZ
   kımıldadı. En iyi: sw2+thr.01 → **S@5 .347** (recall .491, novelty .079).
   Teşhis doğrulandı: pencere açılınca hop düğümleri top-5'e giriyor.
3. **Tur 3 — split_alpha {2,3} × sw {2,3} (thr .01, df .02):** yeni
   `PropagationConfig.split_alpha` knob'u eklendi (pay ∝ w**alpha; 1.0 =
   eski davranış, kanonik iz yeşil; +5 test → toplam 237). Alpha yayılımı
   derinleştiriyor (hop 2-3 yaygın, a3'te hop 5'e kadar) ama S@5 platoda:
   en iyi a3+sw3 → **.349** (recall .502, novelty .066).

**Durum:** web artık top-k'yı geçiyor (.349 vs .309, recall .502 vs .475) ama
iteratif .463'ün çok altında. Bridge@5 web .633 ≈ top-k .641 — köprü kazancı
YOK. Yorum: yayılım mekaniği artık çalışıyor; kalan açık muhtemelen kenar
kalitesi / sorgu ayrıştırma tarafında. Doğal sıradaki adaylar: **renkli çok
tohum** (D12 — zaten Faz 1 ablation planında; çok-hop sorunun parçalarını ayrı
seed yapıp köprüde buluşturmak IRCoT'un yaptığı işin LLM'siz karşılığı),
katman ağırlığı grid'i, dedup. Ayrıntılı sayılar: scratchpad
`grid_summary.json` / `grid2_summary.json` / `grid3_summary.json`.

## Renkli çok tohum ablation'ı (2026-08-14 — İLK GERÇEK SİNYAL)

`core/colors.py` yazıldı (D12): `propagate_colored` — renk başına ayrı yayılım
(enerji renkler arasında EŞİT bölünür, toplam korunur; renk-başı eşik kendi
payına göre ölçeklenir), ≥2 rengin ulaştığı düğüm = köprü. Çekirdek saf kaldı;
ayrıştırmayı çağıran yapar. 7 yeni test (`test_colors.py`) — **toplam 244**,
ruff temiz, kanonik iz yeşil. Commit edilmedi.

Ölçüm: MuSiQue'nin kendi `question_decomposition`'ı iki oracle seviyesinde
(O1: `#N` referansları atılır — saf renkli köprüleme, zincirsiz; O2: `#N`
yerine gold ara-cevap — mükemmel zincir tavanı). Ayarlar: thr .01, alpha 3,
df .02, renk-başı seed {2,3}. Sonuçlar (S@5):

| Sistem | S@5 | recall@5 | novelty@5 | bridge@5 |
|---|---|---|---|---|
| top-k | .309 | .475 | 0 | .641 |
| düz web (en iyi) | .349 | .502 | .066 | .633 |
| iteratif (IRCoT, 4 LLM çağrısı) | .463 | .622 | .168 | .709 |
| **renkli O1 sw2 (LLM'siz!)** | **.466** | .613 | **.191** | **.774** |
| renkli O2 sw2 (tavan) | .705 | .868 | .404 | .879 |

**O1, iteratif baseline'ı sorgu anında SIFIR LLM çağrısıyla yakalıyor/geçiyor**
(.466 vs .463) ve novelty + köprü recall'da net önde. Köprü istatistiği:
O1'de 374 soruda köprü oluştu, 189'unda köprüde gold var. "Bridge-first"
sıralama her yerde "sum"dan kötü → enerji toplamı sinyali zaten taşıyor,
köprü bilgisi açıklama katmanı olarak kalmalı. O2 tavanı (.705) ayrıştırma
kalitesinin büyük kazanç alanı olduğunu gösteriyor.

**Dürüstlük şerhi:** O1 gold ayrıştırma kullanıyor (oracle). Dürüst sayı için
sorgu başına 1 LLM çağrısıyla (IRCoT'un 4'üne karşı) LLM ayrıştırması koşulmalı
— cache mekanizması hazır. Ayrıntı: scratchpad `grid4_summary.json`.

## LLM ayrıştırmalı dürüst ölçüm (2026-08-14 — .358, aşırı ayrıştırma sorunu)

llama3.1:8b, soru başına 1 çağrı (20,6 dk, scratchpad `llm_decomp_cache.jsonl`
+ `grid5_llm_sw2_sum.json`): **S@5 .358** (recall .487, novelty .120,
bridge@5 .630) — düz webin (.349) üstünde ama iteratif .463 ve oracle O1
.466'nın belirgin altında. **Kök neden teşhis edildi: aşırı ayrıştırma** —
llama 1000 sorunun 954'ünü hop sayısına bakmadan 4 parçaya böldü (renk
dağılımı {1:12, 2:2, 3:32, 4:954}). Sonuçları: (a) enerji 4'e bölünüyor
(renk başı 2.5), (b) 4×2=8 seed top-5 penceresini yine dolduruyor —
seed_width=k sorunu renk üzerinden geri geldi, (c) alakasız parçalar yanlış
bölgelere tohum atıyor (k=2 recall .330 < top-k .380 bunun kanıtı; k=10'da
renkli .583 en iyi — sinyal derinlerde var, pencereye sığmıyor). Köprü sayısı
şişti (870 soruda köprü, 646'sında gold) ama bridge@5 .630 — köprüler var,
sıralamada öne geçemiyorlar. Görseller güncellendi (olcum_1/2/3*.png,
scratchpad `plot_olcum2.py`). Doğal sıradaki adımlar: ayrıştırma prompt'unu
"olabildiğince az parça, çoğunlukla 2" diye sıkılaştırmak, qwen3.5:9b denemek,
veya renk sayısını 2-3'e kırpmak.

**Tur 9 (2026-08-14, LLM zincirli renkler): S@5 .460 — İTERATİFLE BAŞA BAŞ,
YARI BÜTÇEYLE.** Renk-0 retrieval'ının top-1 pasajından +1 LLM çağrısıyla
ara-cevap çıkarılıp (NONE guard'lı; 766/1000 çıkarım) sonraki renklerin
sorgusuna eklendi — toplam 2 çağrı/soru vs IRCoT ~4. Sonuç: recall .606
(iteratif .622), **novelty .187 > iteratif .168**, **bridge@5 .730 >
iteratif .709**. Tur 8'in kör başlık zehirlenmesi LLM filtresiyle çözüldü
(.404 → .460). O1 oracle'ı (.466) fiilen yakalandı; kalan büyük alan O2
(.705) = tam ardışık zincirleme (renk başına çıkarım, ~3.3 çağrı). Faz 1
kapısı İÇİN DURUM: top-k net geçildi, iteratifle istatistiksel beraberlik —
"anlamlı fark" henüz yok. Görseller tur 9 ile güncellendi.
Çıktılar `grid9_*`, cache `llm_chain_cache.jsonl`.

**Tur 7 (2026-08-14, few-shot prompt): S@5 .413** (recall .561, novelty .138,
bridge@5 .731, köprüde gold 810/936) — en iyi dürüst sayı. Renk dağılımı
neredeyse değişmedi ({2:79, 3:579, 4:342}); kazanç parça sayısından değil,
örneklerin öğrettiği **kendine-yeterli anahtar kelime üslubundan** geldi.

## Adım 7 — Konsolidasyon (2026-08-14, bitti): kazanan kütüphanede

Tur 7+9'un scratchpad boru hattı kütüphaneye taşındı; artık
`uv run python -m spiyweb.evaluation.run evaluate` varsayılan olarak kazanan
renkli+zincirli akışı koşuyor ve **S@5 .460 birebir yeniden üretildi**
(recall .606, novelty .187, bridge .730, köprü 850/966, non-NONE 766 —
tur 9 ile aynı). Yapılanlar:

- `prompts.py`: `QUERY_DECOMPOSITION_PROMPT` (few-shot, tur 7) +
  `INTERMEDIATE_ANSWER_PROMPT` (NONE guard'lı, tur 9).
- `config.py`: **`ColoredRetrievalConfig`** — ölçülmüş kazanan varsayılan
  (renk-başı seed_width 2, thr .01, alpha 3, max_colors 4, zincir açık,
  cevap ≤10 kelime). Sahibi seçti: core `PropagationConfig` (0.15/1.0) ve
  kanonik §2.6 izi DOKUNULMADI — kazanan işletme noktası bu config'de yaşar.
  `EntityEdgeConfig.max_df_ratio` varsayılanı 0.5 → **0.02** (grid kazananı).
- `retrieve.py`: **`retrieve_colored()`** + `ColoredRetrievalResult`
  (bridges, confidence). Tur script'lerindeki ilk-renk "ham temas fallback"i
  ölü koddu (pozitif temas yoksa ham toplam da ≤0 → core zaten patlar; 1000
  soruda hiç tetiklenmedi) — açık, açıklayıcı ValueError'a çevrildi.
- `evaluation/decompose.py` (yeni): parse_subqueries / decompose_question /
  parse_intermediate_answer / extract_intermediate_answer.
- `evaluation/run.py`: fazlı yapı (LLM fazları embed fazlarından ayrık —
  8 GB VRAM disiplini), `--web {colored,plain}` (varsayılan colored),
  results.json'a `combo` reçetesi + köprü/renk istatistikleri.
- Artifact: `edges_entity.json` df .02 ile yeniden üretildi (**2.5M →
  632.838 kenar**), meta.json güncel. Scratchpad LLM cache'leri
  `data/musique/llm_cache.jsonl`'e birleştirildi (4014 → 5903 satır) —
  yeniden koşum LLM'siz, ~6 dk (CPU embed).
- Testler: `test_retrieve_colored.py`, `test_decompose.py`, run/entity test
  güncellemeleri — **278 test + 3 skip**, ruff temiz, kanonik iz yeşil.

Commit YOK (sahibi erteledi). Sıradaki adaylar: tam ardışık zincir (O2 .705
yönü), dedup→oy ablation'ı, katman ağırlığı grid'i.
İteratife ara .050.

## Tur 10 (2026-08-14, tam ardışık zincir — O2 yönü): S@5 .469

Seviyeli zincirleme (renk i'nin top-1 pasajından çıkarılan ara-cevap renk
i+1'in sorgusuna eklenir; ~3.26 LLM çağrısı/soru vs IRCoT ~4): **S@5 .469**
(recall .619, novelty .192, bridge@5 .744, köprü 971/1000 — 854'ünde gold,
non-NONE 904). Tur 9'un (.460) üstünde, iteratifin (.463) ilk kez nominal
üstünde. **Ama istatistik beraberlik diyor:** eşleştirilmiş bootstrap
(10k örnek) fark = **+0.006, %95 CI [−0.011, +0.024], P(fark≤0)=.226**;
soru-bazlı 226 galibiyet / 228 mağlubiyet / 546 beraberlik. Hop kırılımı:
2-hop .546 vs .539, 3-hop .421 vs .426, **4-hop .300 vs .274** — derin
sorularda web önde. O2 tavanı .705'e mesafe hâlâ büyük; kalan açık
ayrıştırma kalitesi. Çıktılar `grid10_*`, script `grid_tour10.py`.

Koşu sırasında not: makine termal nedenle kapandı (rerank baseline
koşarken); tur 10 sonuçları diske yazılmıştı, kayıp yok.

## Rakip rerank baseline (2026-08-14 — bitti): S@5 .409

Grafiklerdeki "güçlü rakip" için LLM'siz endüstri standardı iki aşamalı
sistem: e5 dense top-50 aday + **BAAI/bge-reranker-v2-m3** cross-encoder
(aynı store, aynı metrikler; "web" kolonunda ama RAKİP etiketiyle).
İlk deneme termal kapanmayla öldü; script artımlı checkpoint + resume +
termal fren (batch 8, soru arası 0.25 s bekleme) ile sağlamlaştırılıp
yeniden koşuldu (~13 dk, 0.8 s/soru, GPU ≤73°C). **Sonuç: S@5 .409**
(recall .562, novelty .124, bridge@5 .720) — top-k .309'u ezip geçiyor ama
Spiyweb .469 ve iteratif .463'ün altında; bridge'de Spiyweb .744 önde.
Sıralama: Spiyweb .469 > iteratif .463 > rerank .409 > düz web .349 >
top-k .309. Çıktı: scratchpad `baseline_rerank.json` +
`baseline_rerank_per_query.jsonl`.

**Görseller güncellendi (tur 10 + rerank):** üç PNG (`olcum_1/2/3*.png`)
yeniden üretildi — sistem kıyası ve hop kırılımına 5. çubuk olarak rerank
RAKİP girdi, yolculuk paneline tur 9 ara basamağı ve rerank çizgisi eklendi,
köprü grafiğinde rerank .720 kolonu var. Script: scratchpad `plot_olcum2.py`
(tur 10'u otomatik tercih ediyor).

## Tur 11 (2026-08-14, dedup→oy ablation'ı): mekanizma kütüphanede, MuSiQue'de nötr

Projenin en özgün iddiası (tekrar → bağ 0 + oy) ilk kez uygulandı ve ölçüldü.
Kütüphane tarafı: `core/dedup.py` (SimilarityFn protokolü, `adaptive_threshold`
`tau = max(floor, mean + sigma*std)`, `find_survivor`), `DedupConfig`
(`config.py`), `propagate`/`propagate_colored`/`retrieve*` opsiyonel
`similarity` + `dedup` + `source_of` parametreleri; sonuçta `votes` (kaynak
bazlı, taban 1 + bastırma sayısı), `suppressed`, `dedup_thresholds` (UI için
görünür). Bastırılan komşunun payı yeniden dağıtılır (enerji korunur). Kanonik
iz artık canlı .95 kenarla dinamik testte yeniden üretiliyor
(`tests/test_dedup.py`, 16 test; toplam süit 295, ruff temiz).

**Ölçüm (tur 10 tabanı üstünde, her şey aynı, tek fark dedup açık):**
doz-yanıt eğrisi floor ile monoton —
floor .80 → S@5 .452, fark −.018 CI[−.026,−.010] ANLAMLI ZARAR (219k bastırma);
floor .85 → .461, −.009 CI[−.014,−.003] anlamlı zarar (37k);
floor .90 → .468, −.002 CI[−.005,+.001] beraberlik (5k bastırma, 837 soruda);
floor .95 → .469, −.001 beraberlik (842 bastırma).
Uyarlanabilir kısım (mean+2σ ≈ .88) hiç devreye girmedi — tau hep tabanda;
e5 uzayında aktif set benzerlik dağılımı tabanın altında kalıyor.

**Dürüst yorum:** MuSiQue havuzu zaten metin bazında dedup'lu — corpus'ta
gerçek tekrar YOK; bu ölçüm mekanizmanın FAYDASINI değil MALİYETİNİ sınırlar.
Sonuç: floor ≥ .90'da maliyet sıfır, oy sinyali bedava üretiliyor; agresif
bastırma benzer-ama-farklı gold pasajları da yuttuğu için zarar. Fayda kanıtı
tekrarlı bir corpus ister (gerçek dünya RAG) — Faz 2 ölçüm adayı. Varsayılan
`floor=0.95`'e çekildi (ölçülen güvenli nokta, gerekçe docstring'de).
Çıktılar: scratchpad `grid11_dedup_*` + `grid11b_floor_summary.json`,
scriptler `grid_tour11_dedup.py` + `grid_tour11b_floorgrid.py`.

## Tur 12 (2026-08-14, ayrıştırma kalitesi): FAZ 1 KAPISI GEÇİLDİ — S@5 .512

Adım 8. Teşhis-önce yaklaşımı (`tour12_diag.py`, dört kaynağı id'de join):
renk sayısı uyumsuzluğu **hop-bazında bakınca zararsız** (ham ortalamalardaki
fark hop karışımı yanılsamasıydı); gerçek kaldıraç **zincir ara-cevabı +
alt-sorgu metni** (gold köprü cevabı yakalanınca 2-hop S@5 .631, yakalanmayınca
.400; 3-4 hop'ta yakalama 14/307 ve 2/156). Varyantlar (hepsi tur-10 tabanı,
aynı 1000 soru, eşleştirilmiş bootstrap):

- **V1 clamp2: .431 ANLAMLI ZARAR** (derin zincirleri kesiyor); clamp3 .470
  beraberlik (−0.34 çağrı/soru) — teşhisi doğruladı, sayı kaldıraç değil.
- **V3 — qwen3.5:9b ayrıştırma, llama çıkarım: S@5 .512 KAZANAN** — tur 10'a
  +.043 CI [+.029,+.057], **iteratif .463'e +.049 CI [+.033,+.065], P=.000 —
  İTERATİF İLK KEZ ANLAMLI GEÇİLDİ**; recall .667 (> iteratif .622), novelty
  .225, bridge .779, **2.6 çağrı/soru** (IRCoT ~4). Renk histogramı
  {1:16, 2:497, 3:362, 4:125}, gold hop uyumu **%72.9** (llama %35.7).
- V2b train-split few-shot (llama): .474 beraberlik — renk dağılımı düzeldi
  ama kazanç gelmedi → kazanç sayı dağılımından değil **ayrıştırma metni
  kalitesinden**.
- V2c tam qwen (çıkarım da qwen): .507 < V3 → qwen çıkarımı katkısız,
  llama çıkarım yeterli (ucuz model yeterli olduğu yerde kalır).
- V4-lite qwen+clamp3: .507, 2.47 çağrı → −.005 sayı için −0.13 çağrı değmez;
  `max_colors=4` kalır.

Sıralama: **SPIYWEB .512** > iteratif .463 > rerank .409 > düz web .349 >
top-k .309. O2 tavanı .705'e kalan ~.19'un adresi teşhise göre zincir-cevap
doğruluğu (sonraki tur adayı).

Teknik notlar: qwen3.5:9b **thinking modeli** — OpenAI-uyumlu uçta tüm token
bütçesi reasoning'e gidip content boş dönüyor; çözüm native `/api/chat` +
`think:false` (`tour12_ollama.py`, 1.3 s/çağrı). Cache anahtarı prompt-only
olduğu için model başına **ayrı cache dosyası** kullanıldı (kalıcı çözüm
terfi setinde: model-başına-yol). Görseller (üç PNG) tur 12 kazananıyla
yeniden üretildi (`plot_olcum3.py`). Çıktılar bu oturumun scratchpad'inde:
`grid12_*`, `subqueries_qwen.jsonl`, `llm_qwen_decomp_cache.jsonl`,
`tour12_diag.json`, `v2b_prompt.txt`.

**Commit YOK. Terfi seti UYGULANDI (aynı gün):** (1) ardışık zincir →
`run.py::_run_colored_web`, `ColoredRetrievalConfig.chain_mode`
("none"/"single"/"sequential", varsayılan sequential); (2) qwen3.5:9b
`decomposition_model` + `decomposition_no_think` + `llm.py::NativeOllamaClient`
(native /api/chat, think:false); (3) `main()` ColoredRetrievalConfig'i geçiriyor,
CLI `--decomp-model`/`--max-colors`/`--chain-mode`; (4) `_llm_cache_path` —
model-başına cache dosyası, `llm_cache_qwen3.5-9b.jsonl` repoya kopyalandı.
per_query alanı `intermediate_answer` → `intermediate_answers` (liste).
**Harness uçtan uca S@5 .5123'ü BİREBİR üretti**; 301 test (298+3 skip) +
ruff yeşil.

**Seed-123 onayı GEÇTİ (2026-08-14, protokolün ilk uygulaması):** taze
1000 soruluk örneklem (`--sample-seed 123`, `data/musique_seed123`, tam
yeniden index + tüm baseline'lar) — SPIYWEB **S@5 .5073** (recall .659,
novelty .226, bridge .760) > iteratif .4420 > top-k .3046. Eşleştirilmiş
bootstrap web−iteratif: **+.0653, %95 CI [+.0483, +.0824], P(diff≤0)=.000**,
W/L/T 309/170/521 — seed-42'deki +.049'dan bile büyük. Kazanan ayara özgü
overfitting bulgusu YOK; .512 (seed 42) ile .507 (seed 123) pratikte aynı.
Hop kırılımı taze örneklemde de aynı hikâye: 2-hop .576/.526, 3-hop .477/.417,
4-hop .374/.253 (web/iteratif). Maliyet 2.63 çağrı/soru. Görsellere şerh
gerekmedi (sonuç hikâyeyi değiştirmiyor; seed-42 sayıları geçerli kalıyor).

## A2 + A3 + A4 kütüphanede (2026-08-14, seed-123 onay koşusuna paralel)

Özgünlük öncelikli üç mekanizma sahibi kararlarıyla kodlandı — formüller,
tasarım gerekçeleri ve kalan kuyruklar **[[faz1-kalanlar]]** "Bitmiş
sayılanlar" bölümünde (bu dosyaya kopyalanmadı):

- **A2 çelişki:** `core/conflict.py` + `ConflictConfig` + `questions.py`
  (D16 şablon soru) + `edges/nli.py`; hop başına yük nötrleşmesi.
- **A3 negatif tohum:** `core/negative.py` yayılan emici alan +
  `NegativeSeedConfig`; `retrieve*` `negative_queries` API'si.
- **A4 dürüstlük çıktıları:** `output.py` — yollar, tema kümeleri (bağlı
  bileşenler), D18 boşluk uyarıları, D35 red raporu; `OutputConfig`.

Toplam **349 test**, ruff yeşil, kanonik iz dokunulmadı, commit yok.
Doğrulama protokolü ayrı kayıtta: [[olcum-protokolu]].

**Tur 8 (2026-08-14, başlık zincirleme — NEGATİF): S@5 .404** — renk i'nin
sorgusuna renk i-1'in top-1 başlığını eklemek (LLM'siz O2 yaklaşıklaması)
tur 7'yi GEÇEMEDİ (recall .546 < .561, bridge .686 < .731). Neden: hop-1
top-1 yanlışsa hata sonraki renge zehir olarak yayılıyor; IRCoT'taki filtre
görevini kör başlık ekleme yapamıyor. O2 yönü gerçek ara-cevap çıkarımı
(soru başına +1 LLM çağrısı) istiyor — açık deney adayı. Görseller tur 7 ile
güncellendi.

**Tur 6 (2026-08-14, sıkı prompt — tek değişken): S@5 .386** (recall .520,
novelty .135, bridge@5 .664, köprüde gold 573/812). Renk dağılımı
{4:954} → {2:82, 3:549, 4:369}; hipotez doğrulandı — renk azaldıkça sayı
yükseliyor (.358 → .386), ama llama "çoğunlukla 2" talimatına rağmen %92
soruyu hâlâ 3-4'e bölüyor. Oracle .466 ile ara: ~.08. Trend, gerçek 2-3
parçalı ayrıştırmayla .44-.47 bandını işaret ediyor. Görseller tur 6 +
SPIYWEB/RAKİP etiketleriyle yenilendi (sahibi tercihi, kalıcı hafızada).
Sıradaki adaylar: few-shot örnekli prompt v3, qwen3.5:9b, parse'ta 3'e kırpma.

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

## Arşiv eki: open-questions.md "Kapanan başlıklar" tam metni (2026-08-14'te taşındı)

**2026-08-14 (uygulama adım 6) — #1 ve #8 kapandı (sahibi seçti):**

- **#1 Hedef normalizasyonu:** doğruluk := **support recall@k** (HippoRAG'ın
  R@k'sı) ve Novelty@k aynı `|gold|` paydasında hesaplanır — iki terim de
  [0,1] recall olduğundan paylaşılan payda normalizasyon kuralının kendisidir.
  `S@k = 0.65·recall + 0.35·novelty`; **birincil sayı S@5** (4-hop sorunun 4
  gold'u k=2'ye sığmaz; 5 HippoRAG'ın manşet kesimi). Tablo k ∈ {2, 5, 10}.
- **#8 Novelty alaka yargısı:** ilgili := MuSiQue'nin gold `is_supporting`
  paragrafı. `Novelty@k(X) = |(topk(X,k) ∩ gold) \ topk(dense,k)| / |gold|` —
  dense baseline'ın **tüm** top-k'sı çıkarılır, yalnız gold isabetleri değil
  (baseline'ın zaten gösterdiği doküman, sırası ne olursa olsun, yeni değildir).
  Sıfır ek etiketleme, sıfır LLM yargıcı; LLM'li alaka yargısı ileride ablation
  olabilir, asla kapı olamaz. Formüllerin evi `evaluation/metrics.py`,
  ağırlıkların evi `EvaluationConfig`.

**2026-08-13 (uygulama adım 5) — eski #12 kapandı:** `Node` pozisyon
alanlarını EMMEDİ; `ChunkRef` index-zamanı girdi sözleşmesi olarak kaldı
(sahibi seçti). `nodes/chunks.py` her unit için Node + ChunkRef çiftini
birlikte üretir ve iki yapıyı adım adım tutarlı tutan tek yer chunker'dır.
Gerekçe: yayılmanın hiç okumadığı alanlar çekirdek şemasına girmesin
(boundary rule 2). Ayrıca **entity kenar ağırlığı** karara bağlandı:
nadirlik ağırlıklı `Σ 1/df(e)` — [[hybrid-edge-layers]].

**2026-08-13 (uygulama adım 2):** node veri modeli kapandı — `core/graph.py`
içinde `Node` şeması: `id`, `layer` (chunk/proposition), `source_id` (doküman
bazlı oy), `length` (kütle formülünün ham girdisi; formülün kendisi hâlâ açık),
`timestamp` (**UTC epoch float** — tazelik yalnız eşitlik bozucu
olduğundan tek ihtiyaç toplam sıralama; `datetime` aware/naive tuzağı ve ISO
format hassasiyeti nedeniyle elendi), `cluster_id`, D34 `polarity` (+1/−1,
varsayılan +1). Kenar katmanları `LayerWeights` config'iyle ağırlıklı toplamla
birleşiyor; ağırlık 0.0 = katman kapalı (ablation anahtarı).

**2026-08-12 oturumunda kapananlar:** çelişki tespiti → index anında NLI
([[contradiction-detection]]); durma eşiği → göreli %15
([[stopping-and-freshness]]); termal reset → hibrit
([[conversation-thermal-memory]]); Faz-1 kapısı → iki baseline + HippoRAG
rapor kıyası, LLM sağlayıcı, platform ve paketleme → [[phase1-settings]].
