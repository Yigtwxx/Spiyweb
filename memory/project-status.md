---
name: project-status
description: Tasarım aşaması bitti, kod yazımı bekletiliyor — sahibi "projeye başlayalım" diyene kadar hiçbir kod yazılmayacak
metadata:
  type: project
---

**Durum (2026-08-12): tasarım tamamlandı, uygulama beklemede.**

Güncelleme (2026-08-12): gap analizi yapıldı; 8 karar kapandı ve dokümanlara
işlendi (D26-D33, spec §8.1b): NLI ile çelişki tespiti, göreli eşik (%15),
çift-baseline kapısı, HippoRAG rapor kıyası, lokal-öncelikli LLM sağlayıcı,
`src/` + `evaluation/` + `spiyweb[ui]` paketleme, hibrit termal reset,
3 işletim sistemi desteği. Bayat dosyalar düzeltildi. README / LICENSE /
CONTRIBUTING eklendi ve repo GitHub'a açıldı. **Başlama kuralı aynen geçerli.**

Tüm tasarım kararları alındı ve yazıya geçirildi:
- 9 çekirdek karar (D1-D9) + 16 genişletilmiş karar (D10-D25) → `CLAUDE.md` §2 ve
  `docs/specs/2026-08-10-spiyweb-design.md`
- 1. fazın teknik ayarları (benchmark, modeller, ortam, lisans) →
  [[phase1-settings]]
- Gerekçeler ve elenen alternatifler → bu `memory/` klasörü
- Kalan 5 açık soru → [[open-questions]] (hiçbiri başlangıcı bloklamıyor)

## Başlama kuralı

**Sahibi açıkça "projeye başlayalım" (ya da eşdeğeri bir başlama komutu) diyene
kadar hiçbir kod, iskelet, dosya ya da bağımlılık oluşturulmayacak.** Tasarım
üzerine konuşmak, karar revize etmek ve dokümanları güncellemek serbest;
uygulamaya geçmek değil.

Başlama komutu geldiğinde ilk adım kod yazmak değil, **uygulama planı** yazmaktır
(`writing-plans`). Kapsam önerisi: önce **çalışan iskelet** — graf kurulumu +
yayılma + eval harness + `top-k` baseline. Dedup, renkli çok tohum, çelişki
yönetimi ve UI ondan sonra, çünkü ilk baseline karşılaştırması diğer her şeyin
değerini belirliyor ([[roadmap-and-gates]]).

Hatırlatma: her mekanizma config'ten tek tek kapatılabilir olmak zorunda; ablation
bu projenin kendini kanıtlama yöntemi.
