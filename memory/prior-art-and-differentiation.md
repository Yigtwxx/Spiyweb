---
name: prior-art-and-differentiation
description: PPR tabanlı graph-RAG zaten var; Spiyweb'in farkı yayılma algoritmasında değil, yayılma sırasındaki dinamik tekrar-oy dönüşümünde
metadata:
  type: reference
---

Bu alan boş değil. Bilinen çalışmalar:

- **HippoRAG** — varlık grafı üzerinde Personalized PageRank; fikrin yayılma
  kısmının en yakın akrabası
- **GraphRAG** (Microsoft) — varlık + topluluk özetleri üzerinden graf tabanlı
  retrieval
- **LightRAG** — hafif graf indeksleme + çift seviyeli retrieval
- **RAPTOR** — hiyerarşik özet ağacı üzerinden retrieval
- **MMR** — çeşitlilik için tekrar eleme (klasik, graf değil)

Sonuç: **yayılma mekanizması özgün değil** ve proje öyleymiş gibi anlatılmamalı.
Matematiği damping faktörlü PPR'ye denk ([[propagation-rules]]).

**Farklılaşma tek bir yerde:** yayılma sırasında tekrarın bağını kesip **oya**
çevirmek, üstelik bunu **dinamik olarak, sorgu anında** yapmak
([[redundancy-as-vote]]). MMR tekrarı atar ve sinyali kaybeder; bu tasarım sadece
token'ı atar, sinyali tutar. Bunun yayınlanmış bir karşılığı bilinmiyor.

Bunun pratik sonucu: projenin değeri "graf RAG yaptım"da değil, "tekrarı kanıta
çevirdim"de. Eval tasarımı da bunu ölçmeli — yani ablation'da tekrar-oy
mekanizmasını kapatıp açmak, tek başına bir deney olmalı.

**Baseline uyarısı:** rakip sadece `top-k` değil. LLM ile sorgu yeniden yazımına
dayalı **iteratif retrieval** (IRCoT tarzı) da güçlü ve kurması çok daha ucuz.
Dürüst bir değerlendirme onu da baseline'a koymalı ([[open-questions]]).
