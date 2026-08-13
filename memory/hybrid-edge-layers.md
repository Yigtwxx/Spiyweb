---
name: hybrid-edge-layers
description: Kenarlar katmanlı hibrit — cosine yalnız seed temasında, sonraki hop'lar varlık ve yapı kenarlarından yürür
metadata:
  type: project
---

Atomlar arası bağ tek bir ölçüye dayanmıyor. Dört katman var ve **hangi katmanın
nerede kullanıldığı** kritik:

| Katman | Nerede kullanılır | Ne getirir |
|---|---|---|
| `semantic` (cosine kNN) | Yalnız topçuğun **ilk teması** + fallback | Sorguya doğrudan benzeyen giriş noktaları |
| `entity` (ortak varlık/kavram) | Asıl **hop yakıtı** | "Aynı X'ten bahseden başka doküman" — gerçek multi-hop |
| `structural` (aynı doküman/bölüm/komşu chunk) | Destekleyici | Bağlam bütünlüğü, parçalanmış anlatının toparlanması |
| `learned` (kullanımla güçlenen) | Ayrı, **kapatılabilir** katman | Sık doğrulanan yollar; ana graf asla mutasyona uğramaz ([[learned-layer-hebbian]]) |

Ayrıca [[contradiction-detection]] index anında **negatif kenarlar** üretir
(`edges/nli.py`); bunlar ağırlıklı bir yayılma katmanı değil, çelişki işaretidir.

**Neden saf cosine olmaz:** cosine ile komşu olan chunk, aynı şeyi söyleyen bir
**parafrazdır**. Böyle kenarlar üzerinden sıçramak "yeni bilgi" değil "aynı şeyin
başka cümlesi" toplar. O durumda [[propagation-rules]] içindeki `D` terfisi kâğıt
üstünde kalır — ağ çalışıyor görünür ama `top-k`'dan farklı bir şey getirmez.
Metaforun "canlanma" kısmı ilk temasla olur, "örümcek ağı" kısmı ise farklı
türden bağlarla anlam kazanır.

Katmanlar index anında ayrı ayrı üretilir, sonra yapılandırmadaki ağırlıklarla
tek bir seyrek matrise birleştirilir. Bunun mimari sonucu: **hibrit karar koda
değil config'e gömülüdür**; yeni bir katman eklemek `core/`'a dokunmayı asla
gerektirmemeli ([[architecture-boundaries]]).

Başlangıç ağırlıkları belirlendi: `semantic .5 / entity 1.0 / structural .3`,
sonrasında küçük bir grid search — [[phase1-settings]].

**Katman-içi kural (adım 3):** structural katmanın üç ilişkisi iç içedir
(komşu ⊂ aynı bölüm ⊂ aynı doküman) — bağımsız kanıt değiller, o yüzden çift
başına **en güçlü etkin ilişki kazanır (max), toplanmaz**. Katmanlar ARASI
birleştirme ise toplamalı kalır (`from_layers`, additive evidence).

**Entity ağırlık kuralı (adım 4, sahibi seçti):** ortak varlık `e` başına
katkı **`1/df(e)`** (df = o varlığı anan chunk sayısı); çift ağırlığı =
katkıların **toplamı** — iki farklı ortak varlık bağımsız kanıttır, structural
ilişkilerin aksine toplanır. Nadir varlık (df=2) güçlü bağ, her yerde geçen
varlık neredeyse hiç. Ortak varlık yoksa çift hiç üretilmez.
`max_df_ratio` (varsayılan 0.5) buna ek clique frenidir: 1/df stopword'ün
ağırlığını sınırlar ama ürettiği kenar sayısını sınırlamaz. Kurucu saf —
çıkarım (spaCy + LLM hibrit) `entities.py`'de, `edges/` paketinin dışında.
