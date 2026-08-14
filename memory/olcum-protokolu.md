# Ölçüm protokolü — overfitting koruması (kalıcı kural)

**Karar (2026-08-14, sahibi):** 12 tur boyunca tüm ayarlar aynı 1000 soruluk
örneklemde (seed 42) yapıldı; en iyi sayı o örnekleme göre *seçilmiş* kazanan.
Seçim yanlılığına karşı kalıcı üç-katmanlı protokol:

| Katman | Set | Ne zaman |
|---|---|---|
| **Ayar** | `sample_seed=42`, 1000 soru | Tüm tur/grid/ablation çalışması; seed sabit — turlar arası kıyas ancak böyle mümkün |
| **Onay (holdout)** | `sample_seed=123`, 1000 soru | YALNIZ kazanan ilan edilince; üstünde ayar/grid **yasak**. Onayda düşen kazanan → seed 42'ye yeni fikirle dönülür |
| **Faz kapanışı** | Tam dev (2417 soru) | Faz gate sayısı, tek sefer |

## Ek korumalar

- **Ön-kayıt:** varyant, koşudan önce tanımlanır (tur planına yazılır);
  rapor kaç varyant denendiğini söyler ("N varyantın en iyisi").
- **CI zorunlu:** nokta tahmini değil eşleştirilmiş bootstrap %95 CI raporlanır.
- **Seed rotasyonu reddedildi:** 2417'lik dev'den çekilen iki 1000'lik
  örneklem ~%41 çakışır; rotasyon hem zayıf koruma hem turlar arası
  karşılaştırılabilirliği bozar.

## Faz 2 adayları (bu kararla not edildi)

- Ayar havuzunu MuSiQue **train split'ine** taşımak (dev'den tamamen ayrık —
  en temiz kurulum; index maliyeti nedeniyle Faz 2'ye).
- Çapraz-dataset genelleme testi: 2WikiMultihopQA / HotpotQA.

İlk uygulama: tur 12 kazananı (S@5 .512, seed 42) için seed-123 onay koşusu
2026-08-14'te başlatıldı; sonucu `project-status.md`'ye işlenir.
