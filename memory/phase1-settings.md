---
name: phase1-settings
description: 1. fazın teknik ayarları — hedef karışımı, benchmark, modeller, ortam ve ölçüm metrikleri, hepsi karara bağlandı
metadata:
  type: project
---

Tek tek sorulup karara bağlandı. Hiçbiri geri dönülemez değil; hepsi `config.py`
ya da ortam seviyesinde yaşıyor.

| Başlık | Karar | Not |
|---|---|---|
| Birincil hedef | **%65 multi-hop doğruluk + %35 serendipity** | Tek bir ağırlıklı hedef fonksiyonu; ikisi de ölçülüyor. **Adım 6'da somutlaştı:** doğruluk = support recall@k, birleşim `S@k = 0.65·recall + 0.35·novelty`, **birincil sayı S@5**; iki terim de aynı `|gold|` paydasında recall olduğu için 65/35 gerçek, nominal değil (açık soru #1 kapandı) |
| Serendipity ölçümü | **Novelty@k** | Ağın getirdiği ama `top-k`'nın hiç getirmediği, yine de ilgili çıkan düğümlerin oranı. **Adım 6 formülü:** ilgili := gold `is_supporting`; `|(topk(X,k) ∩ gold) \ topk(dense,k)| / |gold|` — dense'in TÜM top-k'sı çıkarılır (açık soru #8 kapandı) |
| Ek metrik | **Köprü düğüm recall'ü** | Cevabın gerektirdiği ara doküman ağda kaçıncı sırada? Standart `recall@k` bunu ödüllendirmiyor — iddianın ölçüldüğü tek metrik bu |
| Benchmark | **MuSiQue** | Kasten çok-hop; "cevap zaten tek chunk'ta" sızıntısı en az olan set. **Adım 6 rejimi (sahibi seçti):** dev'den seed=42 ile **1.000 soru** (leksikografik sıralı id üstünde deterministik örnekleme), corpus = örneklemin paragraf havuzu `(title, text)` dedup'lu (~11,6k pasaj) — HippoRAG ile birebir kıyas; `sample_size=0` = tam dev. Edinim: HF `dgslibisey/MuSiQue`'den düz HTTPS jsonl, stdlib urllib, bağımlılık yok. Lisans CC BY 4.0 |
| Baseline | **`top-k` + iteratif retrieval** | Gerçek rakip iteratif retrieval; sadece `top-k`'yı geçmek ikna edici değil. **Kapı: ikisini de anlamlı farkla geçmek** |
| Referans kıyas | **HippoRAG** (rapor amaçlı) | Kapı kriteri değil; "graph-RAG'lerden farkın ne" sorusuna tablo cevabı (2026-08-12) |
| LLM sağlayıcı | **Lokal-öncelikli (Ollama)** + opsiyonel ücretsiz API'ler (Gemini, OpenRouter, Groq) | Soyutlama `core/` dışında (`llm.py`), config'ten seçilir; secret'lar ortam değişkeninden (2026-08-12) |
| Durma eşiği | **Göreli — enjekte enerjinin %15'i** | 10.0 seed'de 1.5'e denk; [[stopping-and-freshness]] (2026-08-12) |
| Platform | **macOS + Windows + Linux** | Device sırası CUDA → MPS → CPU; OS'e özgü path/çağrı yok (2026-08-12) |
| Paketleme | **`src/spiyweb/` layout**; `eval/` → `evaluation/`; `ui/` opsiyonel extra `spiyweb[ui]` | Public'e çıkmadan alınan yerleşim kararları — sonradan değiştirmek pahalı (2026-08-12) |
| Varlık çıkarımı | **spaCy + LLM hibrit** | Çoğunluk spaCy ile, belirsiz durumlar LLM'e. İki yolu da bakımda tutmak gerekiyor. **İlk ölçümde de hibrit AÇIK** (sahibi seçti, 2026-08-14); belirsizlik `evaluation/cache.py`'nin prompt-SHA256 deterministik disk cache'iyle giderildi — LLM bir kez ödenir, koşular birebir tekrarlanır; ablation `--no-entity-llm` |
| Embedding | **multilingual-e5-large** | Türkçe dahil 100 dil destekliyor; hem benchmark hem kendi dokümanların için çalışır |
| Vektör deposu | **numpy + FAISS** | Sunucusuz, tek dosya. Qdrant/pgvector 2. fazın adaptör konusu |
| Katman ağırlıkları | **Elle başla** (`semantic .5 / entity 1.0 / structural .3`), sonra küçük grid search | UI'da görerek ayarlanacak |
| Seed genişliği | **5 atom** | UI'da kaydırıcı olacak |
| Tekrar eşiği | **Uyarlanabilir** | Aktif düğümlerin benzerlik dağılımından hesaplanır. Sabit eşikten sağlam ama hata ayıklaması zor — UI'da hesaplanan eşik görünmeli |
| Ortam | **Python 3.11 + uv** | |
| Repo / lisans | **Baştan public + Apache-2.0** | Açık geliştirme; olgunlaşmamış fikrin erken yargılanması kabul edildi |

**Public repo kararının pratik sonucu:** `CLAUDE.local.md` ilk günden itibaren
`.gitignore`'da olmalı ve orada kalmalı. Kişisel gerekçeler ve henüz
netleşmemiş fikirler dışarı sızmasın.

**Ağır bağımlılıkların paketlenmesi (adım 4-5, sahibi seçti):** granüler
extras — `spiyweb[store]` = numpy+faiss-cpu, `[embed]` = sentence-transformers,
`[entity]` = spacy, `[index]` = üçünün birleşimi; **`dependencies = []`
değişmedi** (çekirdek saflığının paketleme yüzü). CI `--extra store --extra
entity` kurar: FAISS gerçek test edilir, spaCy import yolu ucuza kapsanır;
torch CI dışında — embedding glue'su sahte encoder'la test ediliyor, cihaz
çözümü (`resolve_device`) saf fonksiyon. spaCy modeli (`xx_ent_wiki_sm`)
PyPI paketi değil, `python -m spacy download` manuel adımdır.

İlgili: [[open-questions]], [[roadmap-and-gates]].
