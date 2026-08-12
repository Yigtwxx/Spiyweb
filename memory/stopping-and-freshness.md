---
name: stopping-and-freshness
description: Ağ göreli enerji eşiğiyle durur — enjekte edilen enerjinin %15'i (token bütçesi değil); tazelik yalnızca eşitlik bozucu olarak devreye girer
metadata:
  type: project
---

**1. Durma ölçütü: enerji eşiği.** Enerji eşiğin altına inince dal ölür.
Token bütçesine bağlı durma seçeneği değerlendirildi ve **seçilmedi** — mekanizma
saf ve tek boyutlu kalıyor, davranış context penceresinden bağımsız oluyor.

**Revizyon (2026-08-12): eşik görelidir** — enjekte edilen **toplam enerjinin
%15'i** (10.0 seed'de 1.5'e denk gelir; kanonik örnek değişmez). Gerekçe:
[[conversation-thermal-memory]] artık enerjisi (%20-30) ve profil farkları
toplam enjekte edilen enerjiyi turdan tura değiştirir; mutlak bir eşik aynı
sorguya farklı turlarda farklı davranırdı. Göreli eşik bu tutarsızlığı kapatır.

Bunun kabul edilen sonucu: `retrieve()`'in döndürdüğü içerik miktarı, model
context bütçesiyle ilişkili değil. Güçlü bir sorgu beklenenden çok düğüm aktive
edebilir.

**Bu yüzden ayrı bir emniyet freni gerekiyor:** `max_nodes` (ve/veya `max_hop`)
sert bir üst sınır olarak config'te bulunmalı. Enerji eşiği birincil durdurucu,
bunlar taşma koruması. Yoksa çağıran uygulama context taşmasını kendi başına
yönetmek zorunda kalır ve bu, kütüphaneden beklenmeyecek bir sürpriz olur.

**2. Tazelik yalnızca eşitlik bozucu.** Yeni/güncellenmiş dokümanlar sürekli bir
iletkenlik çarpanı almaz; tazelik sadece iki sonuç başa baş kaldığında devreye
girer.

Gerekçe: sürekli etki, güncel ama alakasız içeriği sessizce yukarı iter — ve bu
hata türünü fark etmek neredeyse imkânsızdır, çünkü sonuçlar makul görünür.
Eşitlik bozucu olarak kullanıldığında ise etkisi sınırlı ve denetlenebilir kalır.

İlgili: [[propagation-rules]], [[node-layers-and-mass]].
