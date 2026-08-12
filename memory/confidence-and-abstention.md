---
name: confidence-and-abstention
description: Ağın toplam enerjisi ve genişliği doğal bir güven skorudur — skor döndürülür, "bilmiyorum" kararını çağıran verir
metadata:
  type: project
---

Ağ neredeyse hiç yayılmadıysa, corpus o soruyu kapsamıyor demektir. Bu bilgi
yayılmanın **bedava yan ürünü**: toplam aktive olan enerji, aktive olan düğüm
sayısı ve ağın kaç hop derinleştiği birlikte bir **güven skoru** verir.

**Karar: skor sonuçla birlikte döndürülür, kararı çağıran verir.** Kütüphane
"bilmiyorum" politikasını dayatmaz; eşiği uygulama belirler.

Gerekçe: bir retriever'ın "cevap yok" demesi ürün kararıdır, kütüphane kararı
değil. Bir chatbot bunu kullanıcıya söylemek isteyebilir, bir analiz aracı zayıf
sonuçla devam etmek isteyebilir.

**Bu neden önemli:** `top-k` bunu **yapısal olarak** veremez. Her zaman en iyi N'i
döndürür, N ne kadar kötü olursa olsun; benzerlik skorları da mutlak bir anlam
taşımaz. Kalibre bir "kapsam dışı" sinyali RAG'in en büyük eksiklerinden biri ve
bu tasarım onu ek maliyet olmadan üretiyor.

Pratik sonucu: bu özellik tek başına projeyi satabilir — multi-hop kazancı marjinal
çıksa bile ([[known-risks]] #4) güven skoru ayakta kalır.

İlgili: [[corpus-gap-detection]] — düşük skorun *nedenini* söyleyen tamamlayıcı
mekanizma.
