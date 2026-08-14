---
name: project-status
description: FAZ 1 KAPISI GEÇİLDİ ve seed-123 onayı geçti — S@5 .512 (seed 42) / .507 (seed 123), iteratif anlamlı geride; terfi + A2/A3/A4 kütüphanede (349 test); tam anlatım arsiv-uygulama-gunlugu.md'de
metadata:
  type: project
---

**Durum (2026-08-14): FAZ 1 KAPISI GEÇİLDİ ve MÜHÜRLENDİ.** Tam kronolojik
anlatım ve tüm tur gerekçeleri: [[arsiv-uygulama-gunlugu]] (arşiv, dondu).
Kalan işler: [[faz1-kalanlar]]. Doğrulama kuralları: [[olcum-protokolu]].

## Güncel sayılar

Kazanan reçete: **qwen3.5:9b ayrıştırma + llama3.1:8b ardışık zincir**
(renk-başı seed_width 2, thr .01, alpha 3, max_colors 4; 2.63 çağrı/soru —
IRCoT ~4'e karşı). MuSiQue 1000 soru, S@5 = 0.65·recall + 0.35·novelty:

| Sistem | seed 42 (ayar) | seed 123 (onay) |
|---|---|---|
| **SPIYWEB** | **.5123** | **.5073** |
| iteratif (IRCoT) — RAKİP | .463 | .4420 |
| rerank (bge-v2-m3) — RAKİP | .409 | — |
| top-k — RAKİP | .309 | .3046 |

- Fark web−iteratif: seed 42 **+.049 CI [+.033,+.065]**, seed 123
  **+.065 CI [+.048,+.082]**, ikisi de P=.000 — overfitting bulgusu YOK
  (protokolün ilk uygulaması, 2026-08-14).
- Köprü recall .78/.76; novelty .225; 4-hop'ta fark en büyük (.374 vs .253).
- Kalan alan: O2 tavanı .705'e ~.20 = zincir-cevap doğruluğu (sonraki tur adayı).
- Görseller (`olcum_1/2/3*.png`) tur 12 kazananıyla güncel.

## Kütüphane durumu

- Kazanan akış terfi edildi: `ColoredRetrievalConfig` (chain_mode
  "sequential", decomposition_model qwen3.5:9b), `NativeOllamaClient`
  (native /api/chat, think:false), CLI bayrakları, model-başına LLM cache.
  Harness uçtan uca .5123'ü birebir üretti.
- Özgünlük mekanizmaları kütüphanede: **A2 çelişki** (`core/conflict.py`,
  yük nötrleşmesi + D16 şablon soru + `edges/nli.py`), **A3 negatif tohum**
  (`core/negative.py`, yayılan emici alan), **A4 dürüstlük çıktıları**
  (`output.py`: yollar, tema kümeleri, D18 boşluk uyarısı, D35 red raporu).
  Formüller ve tasarım kararları: [[faz1-kalanlar]] "Bitmiş sayılanlar".
- **A1 dedup→oy fayda KANITI (2026-08-14):** kopya-enjeksiyonlu ölçüm önce
  mevcut mekanizmayı nötr buldu (kanal: tohum yakalama), sahibi kararıyla
  **tohum dedup (`include_seeds`) + elastik temas doldurma
  (`contact_overfetch`)** eklendi → doz %40'ta **+.027 CI [+.019,+.036]**,
  %10'da +.007; temiz corpus'ta maliyet beraberlik; floor .95 doğrulandı;
  oy sinyali gold lehine ayrıştı (3.18 vs 2.49). Sayılar ve protokol:
  [[faz1-kalanlar]] "Bitmiş sayılanlar". Kalan: V1 parafraz doğrulaması.
- **Ara işler (2026-08-14, V1 parafraz koşusu sürerken):** A4'ün son
  kalıntısı kapandı (`output.py::entity_edge_labels` — en nadir ortak
  varlık etiketi) + **öğrenen katman** kütüphanede (`edges/learned.py`
  Hebbian: orantılı+doygunluklu artış, her turda yaşlandırma, `prune`,
  taban graf dokunulmaz; `LayerWeights.learned=0` varsayılan kapalı).
  Kararlar: [[faz1-kalanlar]] "Bitmiş sayılanlar".
- **Ara işler 2 (2026-08-14, V1 koşusu sürerken):** **sorgu profilleri**
  (`profiles.py` — sahibi kararı: tek `Profile` sınıfı, `as_retrieval()` +
  `as_colored()`; yalnız üç knob'u tabanın üstüne yazar, renkli kazananın
  `split_alpha`/chain ayarları korunur; PRECISE .45/.25/3, EXPLORE .75/.05/8,
  COMPARE .60/.10/6 — GEÇİCİ el değerleri, açık soru #6 ölçüm bekliyor) +
  **konsolidasyon budaması** (`edges/consolidate.py` — `EdgeUsage` kullanım
  sayacı, `prune_layers` kept/removed raporu GERİ YÜKLENEBİLİR,
  `ConsolidationConfig.min_runs=100` freni; "learned" kapsam dışı).
  **391 test** (388+3 skip). Kararlar: [[faz1-kalanlar]] "Bitmiş sayılanlar".
- **Ara işler 3 (2026-08-14 gecesi, V1 sürerken):** **termal sohbet
  hafızası** (D22/D32 — çekirdek `propagate(residue=...)`, enjekte toplam ve
  göreli eşik birlikte ölçeklenir; `ThermalSession` + `ThermalConfig`
  residue_ratio .25 / auto_reset kapalı / temas-örtüşmesi sinyali — sahibi
  kararları; kapsam önce düz yol) + **D34 negatif bilgi atomları**
  (`core/polarity.py`, oransal-tam emme `coefficient=1.0`, `disputes`
  ledger'ı = yok-edilen-enerji defterinin 3. kolu, `DisputeWarning` şablon
  uyarısı; tespit = açık soru #11). CLAUDE.md §2.1/§2.3 + mimari ağaç
  güncellendi. **405 test** (402+3 skip). Kararlar: [[faz1-kalanlar]]
  "Bitmiş sayılanlar".
- **Ara işler 4 (2026-08-14 gecesi):** **önerme katmanı KODU** (D10 —
  `nodes/propositions.py`, chunk başına tek LLM çağrısı, `{chunk}#p{n}`,
  paylaşılan `parse_listing`) + **derivation** 5. kenar katmanı (sahibi
  kararı; `LayerWeights.derivation=1.0` geçici) + **kütle mekaniği** (D11,
  sahibi "tam ve sağlam" istedi — `core/mass.py`, katman-içi normalize
  `μ = clamp((len/ort)**exp, .5, 2)`, kapı `threshold·μ` + iletim
  `damping**(1/μ)`; `MassConfig` varsayılan KAPALI, açılış ölçüme bağlı) +
  index hattı `--propositions` (varsayılan kapalı, maliyet #2) + `nodes.json`
  7 alan (polarity kaybı düzeltildi). ÖLÇÜM YAPILMADI — faz1-kalanlar madde
  3b. **423 test.** Ayrıntı: [[faz1-kalanlar]] "Bitmiş sayılanlar".
- **Çapraz-dataset desteği (2026-08-14):** harness'e `--dataset 2wiki`
  eklendi — `datasets.py::load_2wiki` (orijinal format, kamelliao HF
  aynası; id'ler tip-önekli `comparison__...`; köprü kuralı YAKLAŞIK:
  comparison/evet-hayır → köprü=gold, diğerleri → cevabı içermeyen gold,
  docstring'de belgeli). Protokol: kazanan config AYARSIZ koşulur.
  Koşu V1 parafraz doğrulaması bitince sıraya girecek (~4-6 saat GPU).
- **Router simülasyonu (2026-08-14, CPU, mevcut veriden):** "önce ucuz düz
  web, güven sinyali zayıfsa zincire tırman" fikri seed-42 verisinde test
  edildi — DÜRÜST SONUÇ: MuSiQue'de ÇALIŞMIYOR. Oracle %57 sorguyu LLM'siz
  bırakabilirdi (.5305) ama ucuz sinyaller onları bulamıyor (tek sinyal
  corr ≤ .12; 9-özellikli lojistik AUC .625; %10 LLM'siz = −.010 S@5,
  %20 = −.022). Yapısal neden: MuSiQue %100 çok-hop — kolay tek-hop sorgu
  yok; gerçek trafikte durum farklı olabilir ama bu veriyle kanıtlanamaz.
  Router Faz 2'ye ertelendi; sessiz-kayıp riskine tasarlanan çözüm (gölge
  denetim: düz cevap anında, zincir arka planda örneklemle, fark loglanır →
  kayıp ölçülür hale gelir) tasarım notu olarak kayıtlı. Scratchpad:
  `router_sim.py` + `router_sim_combo.py` + çıktı json'ları.
- **377 test + ruff yeşil; kanonik §2.6 izi dokunulmadı. Commit YOK**
  (sahibi erteledi).

## Kısa tarihçe (tam anlatım arşivde)

- **Adım 1-6 (2026-08-12/14):** saf çekirdek + kanonik iz → veri modeli →
  semantic/structural/entity kenar kurucuları → hibrit entity (spaCy+LLM) →
  chunker + e5 + FAISS store → `retrieve.py` + MuSiQue eval harness
  (3 baseline, metrikler, cache, CLI).
- **Ölçüm kampanyası (tur 1-12, 2026-08-14):** ilk ölçüm düz .309 (eşik-seed
  çatışması + entity clique teşhisi) → grid .349 → renkli çok tohum O1 oracle
  .466 (LLM'siz, iteratifi yakaladı) → LLM ayrıştırma .358 (aşırı bölme) →
  few-shot .413 → zincirli .460 → tam ardışık zincir .469 (iteratifle
  beraberlik) → **tur 12: qwen ayrıştırma .512, ilk anlamlı üstünlük**.
- **Tur 11 (dedup→oy):** MuSiQue'de nötr (havuz zaten dedup'lu; floor .95
  güvenli varsayılan) — FAYDA kanıtı tekrarlı corpus istiyor (A1, açık).
- **Adım 7-8:** kazanan akışın kütüphaneye konsolidasyonu + tur 12 ve terfi.
- **Seed-123 onayı (2026-08-14): GEÇTİ** — yukarıdaki tablo.
