# Spiyweb — Proje Hafızası

Bu klasör, projenin **neden bu şekilde tasarlandığını** tutar. Koddan veya git
geçmişinden çıkarılamayan kararlar buraya yazılır. Her dosya tek bir karar veya
tek bir gerçek içerir.

## Temel tasarım

- [Çekirdek fikir ve metafor](core-concept.md) — kutu, oksijen atomları, canlı topçuk, örümcek ağı
- [Yayılma kuralları](propagation-rules.md) — çarpımsal sönümleme, oranlı bölüşüm, toplamalı birikim, eşik
- [Tekrarın oya dönüşmesi](redundancy-as-vote.md) — projenin en özgün parçası; bağ 0'lanır, oy artar
- [Hibrit kenar katmanları](hybrid-edge-layers.md) — cosine sadece ilk temasta, hop'lar varlık + yapı kenarlarından
- [Düğüm katmanları ve atom kütlesi](node-layers-and-mass.md) — hem chunk hem önerme; kütle katman içinde normalize
- [Renkli çok tohum](multi-seed-colors.md) — sorgu parçalanır, iki rengin buluştuğu atom köprüdür
- [Durma ve tazelik](stopping-and-freshness.md) — sabit enerji eşiği, tazelik yalnız eşitlik bozucu

## Zaman boyutu

- [Öğrenilmiş katman](learned-layer-hebbian.md) — kullanılan bağ güçlenir, ama ayrı ve kapatılabilir katmanda
- [Sohbet termal hafızası](conversation-thermal-memory.md) — kutu her turda soğutulmaz, %20-30 kalır
- [Konsolidasyon](consolidation-pruning.md) — periyodik budama; atom birleştirme 2. faza ertelendi

## Dürüstlük ve çıktı

- [Güven skoru ve bilmiyorum](confidence-and-abstention.md) — ağ yayılmadıysa corpus kapsamıyordur
- [Corpus boşlukları](corpus-gap-detection.md) — köprüsüz iki küme = bilgi boşluğu uyarısı
- [Çıktı sözleşmesi](output-contract.md) — yollar LLM'e açıklama olarak gider, sonuçlar temaya göre gruplanır
- [Çelişki yönetimi](contradiction-handling.md) — negatif yük + kullanıcıya seçenekli soru
- [Çelişki tespiti](contradiction-detection.md) — index anında çok-dilli NLI, `edges/` içinde; core yalnız işaretli veriyi işler
- [Sorgu profilleri ve negatif tohum](query-profiles-and-negative-seeds.md) — precise/explore/compare, ve "X hariç"

## Süreç ve kapsam

- [Mimari sınırlar](architecture-boundaries.md) — `core/` saflığı; terfiyi bedava yapan tek kural
- [Yol haritası ve terfi kapıları](roadmap-and-gates.md) — 1 → 2 → 3, sinyale bağlı (+ ekosistem revizyonu, geçici)
- [1. faz ayarları](phase1-settings.md) — hedef, benchmark, modeller, metrikler, ortam, lisans
- [Geliştirici UI kapsamı](dev-ui-scope.md) — ürün arayüzü değil, ayar için görsel teftiş aracı
- [İsim kararı](naming-spiyweb.md) — Spiyweb; PyPI/npm/GitHub'da müsait, riskleri bilinerek seçildi

## Durum

- [Proje durumu ve başlama kuralı](project-status.md) — tasarım bitti; "projeye başlayalım" denmeden kod yazılmayacak

## Uyarılar ve bekleyenler

- [Literatür ve farklılaşma](prior-art-and-differentiation.md) — HippoRAG/GraphRAG var; fark yayılmada değil dedup'ta
- [Bilinen riskler](known-risks.md) — çelişki körlüğü, hub cezası, tekrar ≠ doğruluk
- [Alternatif yönler](alternative-directions.md) — değerlendirilen yönler; D (öğrenilmiş sönümleme) 2. faza ertelendi
- [Açık sorular](open-questions.md) — 2026-08-12'de güncellendi; veri modeli ve parametre değerleri uygulama sırasında netleşecek
