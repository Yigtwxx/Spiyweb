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
