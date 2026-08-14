---
name: open-questions
description: Kalan açık sorular — 2026-08-14 sadeleştirmesi; 8 başlık açık (#2 önerme maliyeti, #4 fren değerleri, #5 chunk boyutu, #6 profil değerleri, #7 kütle formülü, #9 unutma katsayısı, #10 NLI seçimi, #11 polarite tespiti); kapananların tam metni arşivde
metadata:
  type: project
---

Önceki 11 teknik varsayılan ve 13 tasarım ekseni karara bağlandı
([[phase1-settings]]). Geriye kalanlar, ancak kod yazılırken veya ölçümden
sonra netleşebilecek konular.

## Kalan açık konular

**2. Önerme çıkarımının maliyeti.**
İki katmanlı düğüm yapısı ([[node-layers-and-mass]]) index anında LLM çağrısı
gerektiriyor. 2026-08-14: kod hazır ve zaten OPSİYONEL (`--propositions`
bayrağı, varsayılan kapalı; çağrı sayısı koşudan önce loglanır) — maliyet
ölçümü hâlâ yapılmadı, soru AÇIK.

**4. Emniyet freni değerleri.**
`max_hop = 6`, `max_nodes = 512` geçici varsayılan. Veri birikiyor: 12 turluk
kampanyada ve seed-123 onayında 1000/1000 sorgu `threshold` ile durdu —
frenler mevcut kazanan ayarda hiç devreye girmedi. Kalıcı değerler farklı
corpus/profillerde test edilince atanacak; `stop_reason` raporu kuralı geçerli.

**5. Chunk boyutu.**
Yalnızca "300-500 token" olarak telaffuz edildi ([[alternative-directions]]);
MuSiQue paragraf verdiği için henüz zorlanmadı.

**6. Profil parametre değerleri.**
`precise` / `explore` / `compare` için damping, eşik ve seed genişliği
değerleri ([[query-profiles-and-negative-seeds]]). 2026-08-14: `profiles.py`
yazıldı ve GEÇİCİ el değerleri atandı (PRECISE .45/.25/3, EXPLORE .75/.05/8,
COMPARE .60/.10/6) — soru ölçümle kapanır, henüz kapanmadı.

**7. Atom kütlesi formülü.**
2026-08-14: formül ÖNERİLDİ ve kodda (`core/mass.py`:
`μ = clamp((len/katman_ort)**exp, .5, 2)`; kapı `threshold·μ` + iletim
`damping**(1/μ)`) — GEÇİCİ, varsayılan KAPALI; değerler ve açılış kararı
ölçüm bekliyor, soru o zamana dek açık.

**9. Öğrenen katman unutma katsayısı.**
Zorunlu ([[learned-layer-hebbian]]) ama değeri seçilmedi.

**10. NLI model seçimi ve aday çifti eşiği.**
[[contradiction-detection]] uygulama detayları; `edges/nli.py` Protocol'ü
hazır, gerçek model sarmalayıcısı [[faz1-kalanlar]] madde 1.

**11. Negatif önerme (polarite) tespit yöntemi.**
[[negative-knowledge-atoms]] için: NLI hattında mı, ayrı polarite
sınıflandırıcısıyla mı? 2026-08-14: emme MEKANİZMASI kütüphanede
(`core/polarity.py` + `PolarityConfig`, çekirdek yalnız `polarity=-1`
etiketini tüketir); tespit tarafı hâlâ açık — ölçümlü ablation tespit
gelmeden koşulamaz.

Not: kenar/çıkarım config varsayılanlarının çoğu grid'le ölçüldü ve kazanan
değerler kütüphane varsayılanı oldu (örn. `max_df_ratio` .02, renk-başı
seed_width 2, thr .01, alpha 3); `SemanticEdgeConfig.k` ve yapısal
alt-ağırlıklar hâlâ el değeri statüsünde.

## Kapanan başlıklar (özet — tam metin [[arsiv-uygulama-gunlugu]] ekinde)

- **#1 hedef normalizasyonu + #8 Novelty alaka yargısı** (adım 6):
  `S@k = 0.65·recall + 0.35·novelty`, iki terim aynı `|gold|` paydasında;
  ilgili := gold `is_supporting`. Formüller `evaluation/metrics.py` +
  `EvaluationConfig`'te yaşıyor.
- **#3 uyarlanabilir tekrar eşiği** (tur 11): formül seçildi ve ölçüldü —
  `tau = max(floor, mean + sigma·std)` (`core/dedup.py`), floor varsayılanı
  .95 (ölçülen güvenli nokta); eşik sonuçta `dedup_thresholds` alanıyla
  görünür, UI'da gösterme kuralı [[faz1-kalanlar]] madde 6'da sürüyor.
- **Eski #12 Node/ChunkRef ayrımı** (adım 5) ve **node veri modeli** (adım 2).
- **2026-08-12 kapanışları:** NLI ile çelişki tespiti, göreli %15 eşik,
  hibrit termal reset, çift-baseline kapısı → ilgili karar dosyaları.

Değerlendirilip ertelenen yönler: [[alternative-directions]] (D — öğrenilmiş
sönümleme, 2. faz).
