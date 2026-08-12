---
name: roadmap-and-gates
description: 1 → 2 → 3 sırayla ilerlenecek; her terfi takvime değil somut bir sinyale bağlı
metadata:
  type: project
---

Üç aşama, sırayla ve zamanla gerekenler eklenerek. Sahibinin kararı bu; 3. faz
hedefte kalıyor.

| Faz | İçerik | Bırakma kapısı |
|---|---|---|
| **1. Araştırma prototipi** | Graf + yayılma + dedup + eval harness + geliştirici UI, tek hardcoded store | MuSiQue'de **iki baseline'ı da** (`top-k` + iteratif retrieval) **anlamlı farkla** geçmek; HippoRAG rapor amaçlı kıyaslanır, kapı değildir |
| **2. Kütüphane** | Temiz public API, vektör store adaptörleri, config, docs, paketleme | **Dışarıdan gerçek talep** — issue, kullanıcı, istek. Kendi tahmini değil |
| **3. Framework** | Ingestion, LLM çağrısı, orkestrasyon | — |

**Terfi takvime değil sinyale bağlanır.** "İki ay oldu, artık kütüphane yapalım"
geçerli bir gerekçe değil.

Ne taşınır: graf + yayılma + oy mekanizması 1'den 2'ye **olduğu gibi** gider;
eval harness regression testine dönüşür. Eklenen şey adaptör, API, config ve
docs — yeniden yazım değil.

**1 → 2 bir terfidir, 2 → 3 bir pivottur.** 3. faz farklı bir problem alanı:
framework işi retrieval problemi değil, **API tasarımı ve topluluk** problemidir.
Tartışıldı ve yine de yol haritasında tutulmasına karar verildi. Bağlayıcı tek
kısıt: geriye sızmaması ([[architecture-boundaries]]).

Not: aynı erişimin ucuz yolu, framework yazmadan LangChain/LlamaIndex'e bir
**retriever adaptörü** vermektir. 3. faza girmeden önce bu seçenek bir kez daha
masaya konmalı.

---

## Revizyon (geçici, kesinleşmedi)

Sıralama şöyle güncellendi: **basit çalışan versiyon → tarayıcı yüzü (UI) →
framework / ekosistem.** Ekosistem hedefi, projenin bir *skill* olarak da
eklenebilmesini kapsıyor — günümüzde çoğu proje bu şekilde dağıtılıyor.

**Bu revizyon kesin değil**, sahibi tarafından açıkça "sonradan değiştirilebilir,
kesin olarak düşünme" notuyla verildi. Bu dosyadaki fazlar hâlâ geçerli; değişen
şey vurgu ve sıra.

Kayda değer tek sonucu: **kütüphaneleşme adımı ortadan kalkmıyor, adı
değişiyor.** Tarayıcı yüzü, arkasında sabit bir API olmadan yazılamaz; yani "UI
fazı" pratikte 2. fazın (temiz public API + config) işini de içeriyor. Sıra
değişse de [[architecture-boundaries]] kuralı aynı sebeple geçerli kalıyor:
`core/` saf olmazsa UI de framework de aynı duvara çarpar.

Terfi kapıları değişmedi — hâlâ takvime değil sinyale bağlı, ve 1. fazın çıkışı
hâlâ tek bir ölçüm sayısı ([[phase1-settings]]).
