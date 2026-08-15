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
  [[faz1-kalanlar]] "Bitmiş sayılanlar".
- **A1 V1 parafraz doğrulaması (2026-08-15 gecesi): A1 TAMAMEN KAPANDI.**
  Aynı ön-kayıtla kopya içeriği llama parafrazı (4773 üretim, %0.4 fallback;
  cosine ort .949, yalnız %58'i ≥.95). Doz %40 f95−off **+.0128 CI
  [+.0069,+.0190] P=.000**; f90−f95 anlamsız → **floor .95 varsayılan
  kalır**; doz %10 nötr; oy ayrışması korundu (2.92 vs 2.29). Etki V0'ın
  ~yarısı — parafrazın %42'si floor altı olduğu hâlde fayda anlamlı.
  Üretim 25/5 dk termal duty-cycle ile. Ayrıntı: [[faz1-kalanlar]].
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
  docstring'de belgeli). Protokol: kazanan config AYARSIZ koşuldu.
- **2Wiki çapraz-dataset SONUCU (2026-08-15 gecesi, 1000 soru, ayarsız):**
  S@5 — **SPIYWEB .657** | iteratif .687 | top-k .468. **web−topk +.189 CI
  [+.174,+.204] P=.000** (genelleme güçlü); ama **web−iteratif −.030 CI
  [−.042,−.017]** — MuSiQue'deki üstünlük 2Wiki'ye TAŞINMADI. Hop kırılımı:
  2-hop (784 soru) web **+.031 önde**; 4-hop (216) web **−.249 geride**.
  **Teşhis (per_query'den):** qwen ayrıştırması 4-hop soruların %57'sinde
  yalnız 2 alt-soru üretiyor → 2 renk, 4 zincirli köprüyü kaçırıyor
  (bridge_hit .505 vs 2-hop'ta .710); IRCoT sabit 4 adım atıyor. Kayıp
  yayılımda değil AYRIŞTIRMA DERİNLİĞİNDE — düzeltme adayı (ör. few-shot'a
  2wiki tarzı kompozisyonel örnek, max_colors'a hop ipucu) sahibi kararıyla
  ayrı tur olur; protokol gereği bu koşuda ayar YAPILMADI. Koşu 25/5
  termal duty-cycle'lı sarmalayıcıyla (`run_2wiki_cooled.ps1`); evaluate
  tamamlandı, sarmalayıcının PS 5.1 ExitCode tuhaflığı sahte "FAILED"
  loglattı, rapor elle alındı — sonuçlar sağlam.
- **Tur 13 ÖN-KAYIT (2026-08-15 sabahı, sahibi onayı "Yap"):** tek değişiklik
  `QUERY_DECOMPOSITION_PROMPT` — 4-sorgulu türetilmiş-kıyas örneği + "her
  taraf için tam zincir" kural satırı; başka hiçbir knob dokunulmadı.
  Hipotez: 2Wiki 4-hop'ta ayrıştırma derinliği → bridge_hit ve S@5 artar;
  MuSiQue gerilememeli. Koşular kopya dizinlerde (`data/2wiki_t13`,
  `data/musique_t13` — kazanan artifact'lar korunur); eşleştirilmiş
  bootstrap t13−mevcut iki dataset'te. Karar kuralı: MuSiQue anlamlı
  gerilemezse VE 2Wiki anlamlı iyileşirse terfi, aksi hâlde revert.
  DÜRÜSTLÜK NOTU: bu turdan sonra 2Wiki artık kör çapraz set değil —
  gelecekteki genelleme iddiası üçüncü dataset ister.
- **Tur 13 SONUÇ (2026-08-15): TERFİ.** 2Wiki t13 web S@5 .657→**.713**
  (t13−eski **+.0557 CI [+.047,+.065] P=.000**); **t13−iteratif +.0262 CI
  [+.016,+.037] P=.000 — 2Wiki'de de ÖNE GEÇİLDİ**. 4-hop .474→**.733**
  (+.2595, iteratifin .723'ü de geçildi); 2-hop −.0004 (yan hasar yok);
  4-renk üretimi 124/216→**210/216**. MuSiQue kapısı: t13−eski **−.0029 CI
  [−.0125,+.0069] P=.717 NÖTR** (hop-4'te −.031 hafif bedel, 4-renk
  125→217 — izlenecek). Karar kuralı sağlandı → prompt kütüphanede kaldı;
  `pyproject.toml`'a `prompts.py` E501 istisnası (ölçülen prompt bayt-bayt
  korunur, cache anahtarı). Görsel olcum_5 t13 serisiyle güncellendi.
  Koşu artifact'ları `data/2wiki_t13`, `data/musique_t13`. 423 test yeşil.
- **Termal duty-cycle kaldırıldı (sahibi, 2026-08-15 sabahı):** gece
  koşuları 25/5 döngüyle koştu; bundan sonra koşular KESİNTİSİZ full hız.
- **Ara işler 5 (2026-08-15): UI dışı KOD TAMAMEN BİTTİ (sahibi: "kod
  kısmını tamamen bitir, koşu/ölçüm kalsın").** Üç parça: (1) **gerçek NLI**
  — `nli.py` `TransformersNLIModel` (sahibi seçimi #10: mDeBERTa-v3-base-xnli,
  `spiyweb[nli]` extra, lazy import, device CUDA→MPS→CPU, contradiction
  sınıfı modelin kendi id2label'ından; `NLIModelConfig`+`NLICandidateConfig`)
  + index `--nli` aşaması (aday = yüksek-cosine çiftler, önermeler varsa
  önermeler — D26 keskinlik; `edges_nli.json` + meta makbuzu) +
  `load_nli_edges` → evaluate artifact görünce `conflict_adjacency` ile
  çatışma mekanizmasını otomatik açar; (2) **polarite TESPİTİ** (sahibi
  seçimi #11: LLM piggyback) — `PROPOSITION_EXTRACTION_POLARITY_PROMPT`
  (`NEG:` öneki, aynı çağrı, sıfır ek maliyet), `PropositionConfig.
  tag_polarity` ablation anahtarı, temiz-metin dedup anahtarı, polarity
  propositions.json'da; (3) **HotpotQA loader** — `load_hotpotqa`
  (2wiki ile paylaşılan `_load_hotpot_family` gövdesi), `--dataset
  hotpotqa`, HF aynası URL (CMU http+dengesiz). **447 test** (444+3 skip),
  ruff+format temiz. Kalan: yalnız KOŞULAR (HotpotQA mührü, önerme/kütle
  ölçümü, NLI ölçümü uygun corpus'ta) + Dev UI.
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
