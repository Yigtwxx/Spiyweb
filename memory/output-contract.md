---
name: output-contract
description: retrieve() düz liste döndürmez — aktivasyon yolları LLM'e açıklama olarak verilir, sonuçlar tema kümelerine göre gruplanır
metadata:
  type: project
---

İki karar, ikisi de çıktının şekliyle ilgili.

**1. Yollar LLM'e verilir.** Her aktive düğüm hangi zincirle geldiğini taşır:
*"A → ortak varlık 'X' → D"*. Bu zincir sadece debug bilgisi değil, bir
**gerekçe**. Chunk'larla birlikte LLM'e geçer: "bu bilgi şu yol üzerinden ilgili".

Neden önemli: bu, retrieval'ı **kendini açıklayan** hale getirir. `top-k` bunu asla
veremez — orada tek gerekçe "cosine skoru yüksekti"dir ve bu bir açıklama değildir.
Graf tabanlı olmanın en görünür, en somut faydası bu. Aynı zamanda hata ayıklamayı
ve kullanıcı güvenini doğrudan besliyor ([[confidence-and-abstention]]).

**2. Sonuçlar tema kümelerine göre gruplanır.** Ağda 3 küme ve 2 köprü varsa,
çıktı da bunu yansıtır: "bunlar ayrı temalar, şurada kesişiyorlar". Düz sıralı
liste yerine yapılı çıktı.

Elenen alternatif: ağ yapısını doğrudan **cevap taslağı** olarak verip LLM'e
doldurtmak. Güçlü fikir ama 3. faza (LLM orkestrasyonu) kayıyor ve `core/`
saflığını tehdit ediyor ([[architecture-boundaries]]). Yapıyı aktarmak yeterli;
cevabı kurmak çağıranın işi.

Sonuç olarak `retrieve()` şunu döndürür: sıralı düğümler + enerjileri + oy
sayıları + aktivasyon yolları + tema kümeleri + güven skoru + boşluk uyarıları +
çelişki kayıtları ([[contradiction-handling]]).
