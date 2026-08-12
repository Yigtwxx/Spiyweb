---
name: corpus-gap-detection
description: İki yoğun küme aktive olup aralarında köprü yoksa corpus'ta bilgi boşluğu vardır — uyarı olarak döndürülür
metadata:
  type: project
---

Sorgu iki ayrı yoğun kümeyi canlandırdı ama bu iki küme birbirine hiç bağlanmadı.
Bu, corpus'ta o iki konuyu **birbirine bağlayan içeriğin olmadığı** anlamına gelir.

**Karar: sonuçla birlikte uyarı olarak döndürülür.** Ek hesap maliyeti yok —
bilgi zaten yayılma sırasında ortaya çıkıyor, sadece raporlanıyor.

Bunun değeri retrieval'ın ötesinde: bir **corpus teşhisi**. "Dokümantasyonunda X
ile Y arasında hiçbir bağlantı yok" bilgisi, doküman yazarına retrieval
sonucundan daha faydalı olabilir.

2. fazda bağımsız bir "teşhis modu"na dönüşme potansiyeli var — o zaman ürün
argümanı retrieval değil, *bilgi tabanı kalitesi analizi* olur. Şimdilik kapsam
dar tutuldu ([[roadmap-and-gates]]).

[[confidence-and-abstention]] ile birlikte okunur: güven skoru "cevap zayıf" der,
bu mekanizma "çünkü şu köprü eksik" der.
