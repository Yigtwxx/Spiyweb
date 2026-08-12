---
name: conversation-thermal-memory
description: Sohbette kutu her turda soğutulmaz — önceki turun enerjisinin %20-30'u kalır, takip soruları sıcak bölgeye düşer
metadata:
  type: project
---

Takip sorusu geldiğinde önceki turun ağı hâlâ sıcaktır. Kutuyu her seferinde
sıfırlamak, elde olan bilgiyi atmak demek.

**Karar: önceki turun enerjisinin %20-30'u korunur.** Yeni sorgu bu artık
enerjinin üstüne enjekte edilir; sıcak bölgedeki atomlar daha kolay canlanır.

Gerekçe: takip sorularının coğrafyası gerçekten aynı bölgede yaşar. "Peki bunun
maliyeti ne?" sorusu, bir önceki turda aktive olan alanla neredeyse tamamen
örtüşür. Bu hem tutarlılık hem hız kazandırır — yayılma daha az hop'ta hedefe
varır.

Elenen alternatifler:
- **Her sorgu sıfırdan:** öngörülebilir ve hata ayıklaması kolay, ama sohbet
  bağlamı tamamen boşa gider.
- **Sadece bağlantı sinyalinde ("peki ya…", "onun…"):** dil tespitine bağımlı;
  Türkçe ve İngilizce için ayrı kural yazmayı gerektirir ve bağlantı ifadesi
  içermeyen takip soruları kaçar.

Dikkat edilecek risk: konu değiştiğinde artık enerji **yanlış bölgeyi** ısıtır.
Bu yüzden oran düşük tutuldu (%20-30).

**Karar (2026-08-12): sıfırlama hibrit.** Varsayılan, çağırana bırakılan
`reset()` çağrısı; ek olarak config'te opsiyonel **otomatik konu-değişimi
tespiti** bayrağı (yeni sorgu ile önceki aktif set arasındaki benzerlik düşükse
sıfırla). Otomatik yol kapalı gelir — iki kod yolunun maliyeti bilinerek kabul
edildi. ([[open-questions]] eski #4 böylece kapandı.)

Bu, hiçbir mevcut graph-RAG sisteminde bulunmayan bir boyut
([[prior-art-and-differentiation]]).
