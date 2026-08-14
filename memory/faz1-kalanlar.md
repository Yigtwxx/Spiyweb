# Faz 1 — kalan iş listesi (2026-08-14 itibarıyla)

Kapı (çift baseline'ı anlamlı geçmek) tur 12 ile aşıldı (S@5 .512; seed-123
taze örneklem onayı GEÇTİ — .5073, CI [+.048, +.082] iteratife karşı). Modül bazında Faz 1'in ~%60-65'i bitti; kalanlar aşağıda.
Sahibi kararı: bunlar ilerleyen zamanlarda yapılacak — sıra/öncelik o gün
belirlenir.

**Bakım kuralı (sahibi, 2026-08-14):** bir madde tamamlanınca bu listeden
ÇIKARILIR ve "Bitmiş sayılanlar" bölümüne taşınır — liste her zaman yalnız
gerçekten kalan işi gösterir.

## Öncelik kararı (2026-08-14, sahibi)

**Önce özgünlük kanıtlayan maddeler** yapılacak (bağımlılık sırası varsa ona
uyulur). Projeyi bilinen graph-RAG'lerden ayıran iddialar bunlar; kanıtlanmazsa
proje "iyi uygulanmış bilinen fikirler" olarak kalır:

- **A1 — Dedup→oy FAYDA kanıtı: KANITLANDI (2026-08-14, V0)** — bkz. "Bitmiş
  sayılanlar". Kalan kuyruk: V1 parafraz doğrulaması (aşağıda madde 1b).
- **A2 — Çelişki mekaniği: KÜTÜPHANEDE (2026-08-14)** — bkz. "Bitmiş
  sayılanlar"; kalan kuyruğu aşağıda madde 1.
- **A3 — Negatif tohum: KÜTÜPHANEDE (2026-08-14)** — bkz. "Bitmiş sayılanlar".
  Negatif bilgi atomları (D34) da 2026-08-14 gecesi kütüphaneye girdi —
  mekanizma tamam, polarite TESPİTİ (açık soru #11) hâlâ açık.
- **A4 — Dürüstlük çıktıları: TAMAMEN KAPANDI (2026-08-14)** — kenar-etiket
  üreticisi de eklendi (`output.py::entity_edge_labels`); bkz. "Bitmiş
  sayılanlar".

Geri kalanlar (B): yalnız UI (+önerme ölçümü, madde 3b). Profiller, öğrenen
katman, budama, termal hafıza, D34 atomları ve önerme katmanı+kütle kodu
2026-08-14'te bitti ("Bitmiş sayılanlar").

## Kalanlar

1. **Çelişki — kalan kuyruk:** gerçek NLI model sarmalayıcısı (model seçimi =
   açık soru #10) + `evaluation/index.py`'ye index-time NLI aşaması (aday çift
   üretimi: yüksek benzerlikli çiftler) + uygun corpus'ta ölçüm (MuSiQue'de
   doğal çelişki yok — dedup'la aynı durum).
1b. **A1 kuyruğu — V1 parafraz doğrulaması:** V0 (birebir kopya) kanıtı geldi;
   gerçekçi tekrar = LLM parafraz kopyaları (cos ~.90-.97 bandı) ile aynı
   protokolün tekrarı. Ön-kayıtlı kapı sağlandı, sahibi onayı bekliyor
   (~3-5 saat GPU: üretim + embed + zincir).
2. **Sorgu-anı hız sorunu — ÇÖZÜLMEDİ (2026-08-14, sahibi: ileride dönülecek).**
   Kazanan akış ~2,6 LLM çağrısı/soru → ~5-8 sn gecikme; en sert benimseme
   bariyeri. **Denenen ve ÇALIŞMAYAN:** tırmanma router'ı ("önce ucuz düz web,
   güven sinyali zayıfsa zincir") — simülasyon çürüttü: oracle %57 sorguyu
   LLM'siz bırakabilirdi ama ucuz sinyaller bulamıyor (tek sinyal corr ≤ .12,
   9-özellik lojistik AUC .625; %10 LLM'siz = −.010 S@5); yapısal neden
   MuSiQue'nin %100 çok-hop olması. **Park edilen adaylar (denenmedi):**
   (a) spekülatif paralel + akış (iki yol aynı anda, sonuç aşamalı rafine —
   tahminsiz, algılanan gecikme ~0,2 sn, maliyet aynı); (b) ekstraktif QA
   modeliyle ara-cevap çıkarımı (llama yerine ~100 MB span modeli, CPU'da
   ölçülebilir); (c) ayrıştırma damıtması (cache'teki ~2.000+ qwen
   ayrıştırması bedava eğitim seti); (d) gölge denetim (sessiz kayıp
   çözümü, router'a bağlı değil). Simülasyon artifact'ları: scratchpad
   `router_sim*.py/json`.
3. **Dev UI** — `ui/` Streamlit inspector + force-directed graf görünümü;
   `spiyweb[ui]` extra'sı. (Dedup eşiği UI'da görünür olmalı — kural.)
3b. **Önerme katmanı ÖLÇÜMÜ** — kod tamam (bkz. "Bitmiş sayılanlar") ama
   hiç koşulmadı: `--propositions` ile index maliyeti ölçülür (açık soru #2),
   sonra derivation/kütle ablation'ları S@5'e vurulur (#7 formül değerleri).

## Bitmiş sayılanlar (referans)

**Önerme katmanı + derivation + kütle mekaniği — D10/D11 (2026-08-14 gecesi,
V1 sürerken; KOD tamam, ölçüm madde 3b'de):**
- `nodes/propositions.py::extract_propositions` — chunk başına TEK LLM
  çağrısı (`PROPOSITION_EXTRACTION_PROMPT`, satır-başına düz metin;
  `entities.py` parser'ı `parse_listing` olarak paylaşıldı), id
  `{chunk}#p{n}` boşluksuz, source/timestamp mirası, chunk-içi normalize
  dedup (D6 disiplini), `PropositionConfig(max_per_chunk=12, min_chars=15
  — GEÇİCİ)`. `texts=` override'ı harness'ın composed metnini besler.
- **Derivation** = 5. EdgeLayer (sahibi kararı): `edges/derivation.py`,
  chunk→önerme uniform 1.0; `LayerWeights.derivation=1.0` GEÇİCİ, `0.0` =
  katmanları ayırma ablation'ı.
- **Kütle (D11), sahibi kararı "tam ve sağlam":** `core/mass.py::node_masses`
  — `μ = clamp((len/katman_ort)**exponent, floor, cap)`; İKİ etki: varış
  kapısı `energy >= threshold·μ` ("geç canlanır") + iletim
  `damping**(1/μ)` ("uzağa taşır", enerji defteri korunur). Kütle asla
  kenar ağırlığına/bölüşüme girmez (= "katmanlar arası bağda kütle devre
  dışı" kuralının yapısal hali). `MassConfig(enabled=False — kazanan
  kütlesiz ölçüldü, açılış ölçüme bağlı; exponent 1/floor .5/cap 2 GEÇİCİ)`
  `PropagationConfig.mass` içinde. Uniform uzunluk/`exponent=0`/kapalı →
  bugünkü davranış birebir (testli); kanonik iz dokunulmadı.
- **Index hattı:** `--propositions` bayrağı (varsayılan KAPALI, maliyet #2;
  çağrı sayısı koşudan önce loglanır), `propositions.json` + embed/entity
  aşamalarına önermeler dahil + `edges_derivation.json` (boşken de yazılır);
  `build_index(entity_llm=)` ile iki LLM tüketicisi bağımsız ablation.
  `nodes.json` 4→7 alan (timestamp/cluster_id/polarity artık korunuyor —
  sessiz polarity kaybı düzeltildi, D34 beslemesi); eski indexler geriye
  uyumlu (derivation dosyası yoksa boş sayılır). 18 yeni test → **423 test**
  (420+3 skip), ruff yeşil, commit yok.

**Termal sohbet hafızası — D22/D32 (2026-08-14 gecesi, V1 sürerken):**
çekirdek `propagate(residue=...)` — kalıntı ham enerji olarak tohum
bölüşümünün ÜSTÜNE enjekte edilir, enjekte TOPLAM = seed + kalıntı ve göreli
eşik toplama göre ölçeklenir (D5/D27'nin varlık sebebi; kanonik iz kalıntısız
yolda birebir korunur) + `thermal.py::ThermalSession` (durum tutan kabuk,
çekirdek turlardan habersiz) + `ThermalConfig(residue_ratio=.25 — %20-30
bandı, auto_reset=False, min_overlap=.05 — hepsi GEÇİCİ)`. Sahibi kararları:
otomatik konu-değişimi sinyali = **temas örtüşmesi** (yeni sorgunun index
temasları ∩ sıcak küme; embedding'siz, dil-bağımsız; varsayılan "hiç sıcak
temas yoksa sıfırla"), kapsam = **önce düz retrieve()** (renkli yol
belgelenmiş erteleme). Sıcak düğüm hop-0'da aktive olduğu için yayılımdan
yeni enerji ALMAZ (mevcut "aktif düğüm almaz" kuralının doğal sonucu).

**Negatif bilgi atomları — D34 (2026-08-14 gecesi, V1 sürerken):**
`core/polarity.py::DisputeRecord` + `propagate(polarity=...)` —
`Node.polarity == -1` atoma ulaşan enerji atomda YOK EDİLİR (sahibi kararı:
**oransal, varsayılan tam** — `coefficient=1.0`, düğüm başına tek ateşleme,
eşik altına düşen yayılamaz: karşıt iddianın kanıtı atomda ölür, akıp
geçmez). Kutup atomları çatışma/dışlama alanından ÖNCE ateşler (en kalıcı
çevre). Ledger `disputes` = yok-edilen-enerji defterinin ÜÇÜNCÜ ve son kolu
(CLAUDE.md §2.1 satırı güncellendi); `output.py::DisputeWarning` +
`dispute_warnings` şablon-kurulu "corpus disputes this" uyarısı (LLM'siz,
D17: politika çağıranın). Renkli yol da `polarity` geçirir. Polarite TESPİTİ
(index anında, NLI mi ayrı sınıflandırıcı mı) açık soru #11 olarak AÇIK —
çekirdek yalnız etiketi tüketir. İki iş 17 yeni test → **405 test**
(402+3 skip), ruff yeşil, commit yok.

**Sorgu profilleri — D13 (2026-08-14, V1 koşusu sürerken):**
`profiles.py::Profile` — sahibi kararı: TEK sınıf, `as_retrieval()` +
`as_colored()` ile iki yola da çevrilir (compare için ayrı fabrika yerine).
Profil YALNIZ üç knob'u (damping/eşik oranı/seed genişliği)
`dataclasses.replace` ile tabanın üstüne yazar — renkli tabanın ölçülmüş
kazananı (`split_alpha=3.0`, chain ayarları) asla sessizce kaybolmaz.
Hazır profiller (HEPSİ GEÇİCİ el değeri, açık soru #6 ölçüm bekliyor):
`PRECISE` .45/.25/3 (küçük hızlı sönen top), `EXPLORE` .75/.05/8 (büyük
yavaş sönen), `COMPARE` .60/.10/6 (renkli yolla doğal eşleşme; orada
genişlik RENK BAŞINA). `PROFILES` isim haritası UI seçicisi için.

**Konsolidasyon budaması — D23 (2026-08-14, V1 koşusu sürerken):**
`edges/consolidate.py` — `EdgeUsage` (koşular arası kullanım sayacı;
learned ile AYNI contributor-çifti kanıtı, kanonik ikili, `to_dict/from_dict`)
+ `prune_layers(layers, usage, config) -> ConsolidationReport` (kept +
removed katman başına; **kept ∪ removed = girdi → geri yüklenebilir**, memory
şartı) + `ConsolidationConfig(min_runs=100 — GEÇİCİ el freni: altındayken
HİÇBİR kenar silinmez, "sorulmamış soruların yolları" uyarısı)`. Kenar üçlüsü
listeden çıkar, 0.0'a çekilmez (0.0 = dedup işareti). "learned" katmanı
kapsam dışı (kendi `prune`'u; docstring'deki D22 yazım hatası D23 yapıldı).
İki iş birlikte 14 yeni test → **391 test** (388+3 skip), ruff yeşil,
commit yok.

**Kenar-etiket üreticisi — A4 kapanışı (2026-08-14, V1 koşusu sürerken):**
`output.py::entity_edge_labels(pairs, entities)` — çift başına ortak
varlıkların **en nadiri** etiket olur (`shared entity 'X'`; df verilen
entities haritasından, eşitlik leksikografik — entity katmanının 1/df
sezgisiyle tutarlı). Ortak varlık yoksa/öz-çiftse/id bilinmiyorsa girdi yok;
`ActivationPath.rendered` iki yönü de aradığı için anahtar verilen sırada.

**Öğrenen katman — Hebbian (2026-08-14, V1 koşusu sürerken):**
`edges/learned.py::LearnedLayer` + `LearnedLayerConfig` (enabled /
learning_rate .1 / forgetting .99 / max_strength 1.0 — hepsi GEÇİCİ, açık
soru #9). Sahibi kararları: pekiştirme = **enerji taşıyan tüm bağlar**
(contributor çiftleri; opsiyonel `accepted=` filtresi), artış = **orantılı +
doygunluk** (sahibi melezi: `lr·min(1, carried/injected)`, tavan
max_strength; `carried` ≈ düğüm enerjisi / contributor sayısı — çekirdek
per-kenar akış tutmaz, yaklaşım docstring'de), unutma = **her reinforce'ta
yaşlandırma** (tüm kenarlar × forgetting) + `prune()` konsolidasyon kancası.
Taban graf DOKUNULMAZ; çıktı `Graph.from_layers`'ın "learned" katmanı,
okuma anahtarı `LayerWeights.learned` (0.0 varsayılan = kapalı).
`to_dict/from_dict` ile kalıcılık çağıranın. Uçtan uca ablation testi:
öğrenilmiş kenar ölü komşuyu yalnız `learned>0` ağırlıkta canlandırıyor.
16 yeni test (+4 etiket testi) → **372 test** (369+3 skip), ruff yeşil,
commit yok.

**Dedup→oy FAYDA kanıtı — A1 V0 (2026-08-14):** ön-kayıt: seed-42 havuzuna
birebir kopya enjeksiyonu (doz %10 = 1184 / %40 = 4734 kopya, rastgele
seçim gold dahil, `#dup1` kimliği ayrı source_id), çökme puanlaması (kopya
kaynağa sayılır, top-5 penceresi İÇİNDE çöker — zarar metrik hilesi değil
pencere israfı; her sistem aynı kuralla). **1. tur dürüst sonuç:** mevcut
komşu-dedup her dozda istatistiksel NÖTR (%40'ta +.0005 CI [−.0040,+.0046]);
teşhis hasar kanalını buldu — **tohum yakalama**: %40 dozda 1056/2596 rengin
iki temas slotu da aynı pasajın kaynak+kopyası; komşu bastırma enjekte edilmiş
ikizlere dokunamıyor. **Sahibi kararı (B):** çekirdeğe tohum-seviyesi dedup
(`DedupConfig.include_seeds`, enjeksiyondan ÖNCE bastır → pay yeniden bölüşümle
korunur + oy) + `retrieve`'e **elastik temas doldurma**
(`contact_overfetch=3`: ikiz atlanır→oy, slot SIRADAKI FARKLI fikre gider;
`contact_suppressed`/`contact_votes`/`contact_tau` ledger'ları,
`votes()` iki aşamayı birleştirir). **2. tur sonuç:** doz %40 ON−OFF
**+.0274 CI [+.0190,+.0357] P=.000** (.4740→.5014; hasarın ~%71'i geri),
doz %10 **+.0068 CI [+.0013,+.0123] P=.008**; temiz corpus maliyeti f95
−.0020 beraberlik (sigorta bedava), f90 −.0068 anlamlı zarar → **floor .95
doğrulandı**. Pencere israfı 1110→242 slot; oy sinyali ilk kez ayrıştı
(gold kaynak ort. 3.18 oy vs diğer 2.49). Doz-yanıt eğrisi: fayda dozla
büyür (0: −.002 | %10: +.007 | %40: +.027). CLAUDE.md §2.3'e "Seed twins"
kuralı eklendi. **356 test** (353+3 skip), ruff yeşil, commit yok.
Script'ler scratchpad: `a1_build_corpus.py`, `grid_a1.py`, `a1_stats.py`,
`a1_diag.py`; hücre çıktıları `a1_d{0,10,40}_{off,f95,f90}*.json`.

Çekirdek graf+yayılım, dedup→oy (tur 11), renkli çok tohum + ardışık zincir
(tur 12 kazananı kütüphanede), semantic/entity/structural kenarlar, hibrit
entity, embedding+FAISS store, eval harness + 3 baseline, ölçüm görselleri,
ölçüm protokolü ([[olcum-protokolu]]).

**Çelişki mekaniği — A2 çekirdeği (2026-08-14):** `core/conflict.py`
(NegativeEdge, ConflictRecord, `neutralize` — sahibi formülü: yük nötrleşmesi,
`absorbed = k·s·min(E_a,E_b)`, iki taraf da aynı miktarı kaybeder; hop başına,
yayılım İÇİNDE uygulanır — sönen atom sonraki hop'ta yayılamaz, çift koşu
başına EN FAZLA BİR kez ateşler), `ConflictConfig` (enabled/coefficient,
ablation anahtarı), `propagate`/`propagate_colored`/`retrieve*` `negative` +
`conflict` opsiyonel parametreleri, sonuçta `conflicts` ledger'ı + `disputed`
(sıfırlanan taraf disputed DEĞİL — sıralamadan zaten düştü), `questions.py`
(D16 şablon soru, 3. seçenek = cevapsız varsayılan "ikisi de disputed"),
`edges/nli.py` (`NLIModel` Protocol + `build_nli_edges`, çift yönlü max,
eşik dahil; `NLIEdgeConfig.contradiction_threshold=0.9` el varsayılanı).
Kanonik iz dokunulmadı; 24 yeni test → **325 test**, ruff yeşil, commit yok.

**Negatif tohum — A3 (2026-08-14):** `core/negative.py` (`negative_field` —
negatif sorgu AYNI kurallarla kendi alanını örer, sahibi seçti: bölge emici
olur, "yolları da söndürür"; `AbsorptionRecord` ledger'ı) +
`NegativeSeedConfig` (enabled/seed_width/`energy_ratio` ayrı knob — sahibi
seçti, varsayılan 1.0 = simetri/coefficient), `propagate` hop başına
`absorb` uygulaması (düğüm başına TEK ateşleme, `min(E, k·alan)` yok edilir,
eşik altına düşen yayılamaz; alan çatışmalardan ÖNCE ateşler — çevre önce),
`retrieve*` `negative_queries=[embedding]` API'si (temassız dışlama sessiz
no-op — pozitif sorgunun aksine hata değil). 12 yeni test → **337 test**,
ruff yeşil, commit yok.

**Dürüstlük çıktıları — A4 (2026-08-14):** `output.py` (yeni) — sahibi
kararları: tema kümesi = **aktif alt-grafın bağlı bileşenleri** (sorgu anında;
renkli koşuda `color_composition` ile renk bileşimi işlenir), yoğunluk =
**≥3 düğüm VE ≥%15 enerji** (`OutputConfig`, iki knob), D35 tetiklemesi =
**çağıran tetikler kütüphane kurar** (D17 kazandı; ablation = çağırmamak),
kenar gerekçesi = **opsiyonel etiket haritası** (harita yoksa düz zincir).
İçerik: `ActivationPath` (`contributors` geriye yürünür, en güçlü besleyici;
`converging` görünür; `rendered()` "A -> shared entity 'X' -> B"),
`theme_clusters`, `gap_warnings` (D18 — bileşen ayrımı = köprüsüzlük testi,
bedava), `build_refusal_report` (D35 — 4 slot: kümeler/boşluk/enerjinin
öldüğü yer/eksik kaynak türü; şablonlar LLM'siz). 12 yeni test → **349 test**,
ruff yeşil, commit yok.
