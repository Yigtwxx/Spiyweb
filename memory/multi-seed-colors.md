---
name: multi-seed-colors
description: Sorgu parçalanıp her parça ayrı renkte tohum olarak atılır; iki rengin buluştuğu atom köprüdür ve multi-hop cevabı oradadır
metadata:
  type: project
---

Tek topçuk yerine sorguyu alt-sorulara veya varlıklarına ayır, her parçayı ayrı
bir **renkte** tohum olarak at. Renkler ayrı ayrı yayılsın.

Bir atomda **iki farklı renk buluşuyorsa**, o atom sorunun iki ayrı parçasını
birbirine bağlayan **köprüdür** — ve çok-hop bir sorunun cevabı tam olarak orada
durur.

**Karar: 1. faza girer, ablation olarak** (açık/kapalı ölçülür).

Neden en yüksek potansiyelli fikir: mevcut toplamalı birikim
([[propagation-rules]]) zaten yakınsayan yolları ödüllendiriyor, ama **renk ayrımı
olmadan**. Renk eklendiğinde iki şey birden kazanılıyor:
- **Daha güçlü sinyal:** "iki farklı alt-sorudan geldi" ile "aynı alt-sorudan iki
  yoldan geldi" artık ayırt ediliyor. İlki gerçek köprü, ikincisi sadece yoğunluk.
- **Açıklanabilirlik:** bir düğümün *neden* yükseldiği renk kompozisyonundan
  okunabiliyor. [[output-contract]] içindeki yol açıklamalarını doğrudan
  besliyor.

Maliyeti: sorgu ayrıştırma adımı. Kural tabanlı (varlık çıkarımı ile) başlanabilir,
LLM'e ihtiyaç duyulursa `core/` dışında kalır ([[architecture-boundaries]]).

Bu, [[prior-art-and-differentiation]] içinde sayılan sistemlerin hiçbirinde yok ve
[[redundancy-as-vote]] ile birlikte projenin "PPR varyantı" olmaktan çıkma
ihtimalini taşıyan ikinci ayak.
