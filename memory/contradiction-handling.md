---
name: contradiction-handling
description: Çelişen atomlar negatif yükle birbirini söndürür; ayrıca kullanıcıya seçenekli bir soru üretilir, cevap gelmezse iki taraf da işaretlenerek verilir
metadata:
  type: project
---

Tasarımın en ciddi kör noktası kapatıldı ([[known-risks]] #1). Cosine `0.9` olan
iki chunk zıt şey söyleyebilir ve [[redundancy-as-vote]] bunu "mutabakat" sayıp
yanlış konsensüsü güçlendirir.

**Karar: çelişki negatif yük olarak modellenir.** Zıt şey söyleyen iki aktif atom
birbirinin enerjisini artırmaz, söndürür ya da "çelişkili" etiketiyle işaretlenir.
Metafora oturuyor — pozitif ve negatif yük ([[query-profiles-and-negative-seeds]]).

**Revizyon: çelişki kullanıcıya seçenekli bir soru olarak sunulabilir.**
Sistem sessizce bir tarafı seçmez; "bu iki kaynak çelişiyor, hangisini baz alalım?"
diye sorar.

Soruyu **kütüphane üretir, şablonla, LLM'siz.** `retrieve()` yapılandırılmış
çelişki verisiyle birlikte hazır soru metnini ve seçeneklerini döndürür. Şablon
`core/` dışında durur, böylece çekirdek saflığı korunur
([[architecture-boundaries]]) ama her kullanıcı aynı işi baştan yazmak zorunda
kalmaz.

**Cevap gelmezse (otomatik/batch kullanım): iki taraf da context'e girer,
"burada anlaşmazlık var" notuyla.** LLM çelişkiyi cevapta belirtebilir. Hiçbir
bilgi sessizce kaybolmaz.

Elenen alternatif: güçlü olanı otomatik seçmek. Reddedildi çünkü azınlıkta kalan
doğru bilgi sessizce silinir — ve bu, mekanizmanın en tehlikeli olduğu senaryonun
tam kendisi.

Tespit yöntemi ayrı bir kararla kapandı (2026-08-12): index anında NLI —
[[contradiction-detection]].

Bu paket ([[confidence-and-abstention]] + [[output-contract]] + bu dosya) birlikte,
"dürüst retriever" diye ayrı bir konumlandırma oluşturuyor.
