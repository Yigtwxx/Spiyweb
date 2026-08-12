---
name: supersession-vs-contradiction
description: NLI çelişkisi + sıralı timestamp = çelişki değil güncelleme; eski atom sönümlenir, yenisi "geçerli" işaretlenir (D36, uygulama Faz 2)
metadata:
  type: project
---

**Karar (2026-08-12, D36): süpersesyon ile çelişki ayrılır.** NLI bir çelişki
bulduğunda ([[contradiction-detection]]) iki atomun timestamp'ı **sıralıysa ve
aynı kaynağın/konunun evrimi** ise, bu bir çatışma değil **güncellemedir**:
eski atom sönümlenir (negatif yük almaz), yenisi "geçerli" işaretlenir ve
kullanıcıya çelişki sorusu ÜRETİLMEZ. Timestamp sırası belirsizse normal
çelişki akışı ([[contradiction-handling]]) işler.

**Why:** "Eski bilgi + yeni bilgi" karışımı RAG'in gerçek bir ağrısı; ama
dürüst not — **alan bakir değil**: T-GRAG (2508.01680), Temporal Validity
(2606.26511), TruthfulRAG (2025-26) tam bu bölgede çalışıyor. Bizim farkımız
mekanizmanın enerji ağına gömülü olması; pazarlama "ilk biz" değil "bizde
böyle çözülüyor" diliyle yapılmalı.

**Zamanlama:** Tasarıma şimdi girer (bu dosya + şemadaki timestamp alanı,
[[open-questions]] node modeli); **uygulama Faz 2**. MuSiQue'de zamansal
çatışma azdır — Faz 1'de ölçülemez, emek görünmez.

**How to apply:** Karar `edges/nli.py` çıkışında verilir (çelişki mi,
süpersesyon mu); `core/` iki farklı hazır etiketi işler. Tazelik kuralıyla
karışmamalı: [[stopping-and-freshness]] eşitlik bozucu olarak kalır,
süpersesyon ise NLI kanıtı ister — sadece "daha yeni" olmak yetmez.
