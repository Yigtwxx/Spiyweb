---
name: project-status
description: Tasarım bitti; uygulama başladı — ilk adım (yayılma çekirdeği + kanonik iz testi + 3-OS CI) tamamlandı
metadata:
  type: project
---

**Durum (2026-08-12): tasarım tamamlandı, uygulama başladı.**

## Uygulama adım 1 (tamamlandı)

Sahibi "çok basit, minicik bir adım, GitHub'da da görünsün" dedi; en küçük
anlamlı dikey dilim seçildi: **saf çekirdek + kanonik izin regresyon testi**.

- `src/spiyweb/config.py` — `PropagationConfig` (seed_energy 10.0, damping 0.60,
  threshold_ratio 0.15, max_hop 6, max_nodes 512) + aralık doğrulaması
- `src/spiyweb/core/graph.py` — seyrek ağırlıklı komşuluk; **0.0 ağırlık
  bastırılmış kenar** demek (dedup sonrası), negatif ağırlık reddediliyor
  (negatif yük ayrı mekanizma)
- `src/spiyweb/core/propagate.py` — çarpımsal sönümleme, oranlı bölüşüm,
  toplamalı birikim, göreli eşik, `max_hop`/`max_nodes` emniyet frenleri
- `tests/` — 25 test; `CLAUDE.md` §2.6 kanonik izi birebir sabitlendi
- `.github/workflows/ci.yml` — ubuntu + macOS + windows matrisi, ruff + pytest

İki tasarım detayı kod yazılırken netleşti ve dokümanlara işlendi:

1. **Eşik birikmiş enerjiye uygulanır**, tek tek katkılara değil. Katkı bazında
   uygulansaydı `D` yalnız 1.75 alırdı; 1.125 + 1.75 = 2.875 birikimi ölürdü —
   yani converging evidence, yani projenin bütün değer önerisi.
2. **Zaten aktif komşular paydadan düşer**, enerji geri sızmaz. Bastırılmış
   kenarın renormalizasyonuyla aynı kural; `F = 1.725` ancak böyle çıkıyor.

Çekirdeğin **sıfır çalışma zamanı bağımlılığı** var — `core/` saflığı kuralı
paketleme seviyesinde de uygulandı.

**Sıradaki:** düğüm/kenar veri modeli (kaynak ID, timestamp, katman, cluster ID)
→ kenar kurucular → gömme + FAISS deposu → eval harness + `top-k` baseline.
Dedup, renkli çok tohum ve çelişki yönetimi baseline karşılaştırmasından sonra.

Güncelleme (2026-08-12): gap analizi yapıldı; 8 karar kapandı ve dokümanlara
işlendi (D26-D33, spec §8.1b): NLI ile çelişki tespiti, göreli eşik (%15),
çift-baseline kapısı, HippoRAG rapor kıyası, lokal-öncelikli LLM sağlayıcı,
`src/` + `evaluation/` + `spiyweb[ui]` paketleme, hibrit termal reset,
3 işletim sistemi desteği. Bayat dosyalar düzeltildi. README / LICENSE /
CONTRIBUTING eklendi ve repo GitHub'a açıldı. **Başlama kuralı aynen geçerli.**

Aynı gün, özgünlük oturumu: 4 yeni karar (D34-D37) — [[negative-knowledge-atoms]]
(tasarım şimdi, uygulama Faz 1 ölçümü sonrası ablation),
[[explained-abstention]] (Faz 1), [[supersession-vs-contradiction]] (Faz 2),
[[corpus-lint]] (Faz 2 ürün adayı / B planı).

Tüm tasarım kararları alındı ve yazıya geçirildi:
- 9 çekirdek karar (D1-D9) + 16 genişletilmiş karar (D10-D25) → `CLAUDE.md` §2 ve
  `docs/specs/2026-08-10-spiyweb-design.md`
- 1. fazın teknik ayarları (benchmark, modeller, ortam, lisans) →
  [[phase1-settings]]
- Gerekçeler ve elenen alternatifler → bu `memory/` klasörü
- Kalan 5 açık soru → [[open-questions]] (hiçbiri başlangıcı bloklamıyor)

## Başlama kuralı (2026-08-12'de kalktı)

Kural şuydu: sahibi açıkça başlama komutu vermeden hiçbir kod yazılmayacak.
**Komut geldi ve ilk adım atıldı**; kural artık geçmiş kayıt. Yerine geçen tek
sıralama kuralı: baseline karşılaştırması diğer her şeyin değerini belirlediği
için önce **çalışan iskelet** (graf + yayılma + eval harness + `top-k`),
sonra dedup / renkli çok tohum / çelişki yönetimi / UI ([[roadmap-and-gates]]).

Hatırlatma: her mekanizma config'ten tek tek kapatılabilir olmak zorunda; ablation
bu projenin kendini kanıtlama yöntemi.
