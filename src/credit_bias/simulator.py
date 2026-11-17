"""Sintetik kredit müraciətçiləri populyasiyasının generatoru.

Əsas dizayn prinsipi: hər iki qrupun (A — əsas, B — qorunan qrup) HƏQİQİ
ödəmə qabiliyyəti eyni paylanmadan gəlir. Deməli, modelin qruplar arasında
yaratdığı istənilən fərq "real riskin" deyil, verilənlərə daxil edilmiş
qərəz mexanizmlərinin nəticəsidir. Simulyasiyanın üstünlüyü də budur:
real həyatda görünməyən "həqiqi" etiketi (y_true) biz bilirik.

Qərəz mexanizmləri Suresh & Guttag (2021) təsnifatına uyğundur:
  * historical  — tarixi qərəz: keçmişdə B qrupundan olan etibarlı
                  müştərilərin bir hissəsi "pis" kimi qeydə alınıb;
  * measurement — ölçmə qərəzi: B qrupunun gəliri və kredit tarixçəsi
                  sənədlərdə real səviyyədən aşağı əks olunur
                  (məsələn, qeyri-formal məşğulluq);
  * representation — təmsil qərəzi: B qrupu təlim verilənlərində azdır;
  * proxy       — qrupla korrelyasiyalı, amma ödəmə ilə əlaqəsiz əlamət
                  (məsələn, yaşayış zonası kodu).

Bütün ədədi parametrlər şərtidir (illüstrativdir) və heç bir ölkənin real
statistikasını təmsil etmir.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd

FEATURES = ["log_income", "debt_to_income", "credit_history_years",
            "late_payments", "employment_years", "zone_code"]
GROUP = "group"            # 0 = A qrupu, 1 = B qrupu (qorunan əlamət)


@dataclass(frozen=True)
class BiasConfig:
    """Bir simulyasiya ssenarisinin parametrləri."""
    name: str = "neytral"
    minority_share: float = 0.30   # B qrupunun populyasiyada payı
    label_bias: float = 0.0        # B-nin etibarlı müştərilərinin "pis" qeydə alınma ehtimalı
    measurement_bias: float = 0.0  # B-nin sənədləşmiş göstəricilərinin sürüşməsi (std vahidi ilə)
    proxy_strength: float = 0.0    # zona kodu ilə qrup arasında əlaqənin gücü (0..1)

    def with_(self, **kw) -> "BiasConfig":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        return asdict(self)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_population(cfg: BiasConfig, n: int, rng: np.random.Generator,
                        force_group: int | None = None) -> pd.DataFrame:
    """`n` müraciətçidən ibarət populyasiya yaradır.

    force_group — "əkiz" (counterfactual) test üçün: bütün təsadüfi çəkilişlər
    eyni qalır, yalnız qrup dəyişdirilir. Eyni toxumla (seed) iki dəfə çağırmaqla
    eyni insanların başqa qrupda olsaydı necə görünəcəyini alırıq.

    Qaytarılan sütunlar:
      FEATURES   — bankın gördüyü əlamətlər;
      group      — qorunan əlamət (0/1);
      y_true     — həqiqi nəticə (1 = krediti qaytarır), yalnız simulyasiyada məlumdur;
      y_hist     — tarixi qeydlərdəki etiket (tarixi qərəzlə təhrif oluna bilər).
    """
    g = (rng.random(n) < cfg.minority_share).astype(int)
    if force_group is not None:
        g = np.full(n, int(force_group))

    # 1) Latent ödəmə qabiliyyəti: HƏR İKİ QRUP ÜÇÜN EYNİ paylanma.
    z = rng.normal(0.0, 1.0, n)
    p_repay = _sigmoid(1.4 + 1.3 * z)              # orta hesabla ~75% qaytarır
    y_true = (rng.random(n) < p_repay).astype(int)

    # 2) Bankın gördüyü əlamətlər z-nin küylü ölçmələridir.
    mb = cfg.measurement_bias * g                  # yalnız B qrupuna təsir edir
    log_income = 6.9 + 0.30 * z + rng.normal(0, 0.30, n) - 0.30 * mb
    debt_to_income = np.clip(0.35 - 0.07 * z + rng.normal(0, 0.10, n) + 0.07 * mb, 0.0, 1.5)
    credit_history_years = np.clip(5.0 + 1.6 * z + rng.normal(0, 2.0, n) - 1.6 * mb, 0.0, None)
    late_payments = rng.poisson(np.exp(0.1 - 0.7 * z))
    employment_years = np.clip(6.0 + 1.5 * z + rng.normal(0, 3.0, n), 0.0, None)

    # 3) Proksi: qrupla korrelyasiyalı, ödəmə qabiliyyəti ilə ƏLAQƏSİZ.
    s = cfg.proxy_strength
    zone_code = s * (2 * g - 1) + np.sqrt(1 - s ** 2) * rng.normal(0, 1, n)

    # 4) Tarixi etiket: B-nin etibarlı müştərilərinin bir hissəsi "pis" yazılıb.
    flip = (g == 1) & (y_true == 1) & (rng.random(n) < cfg.label_bias)
    y_hist = np.where(flip, 0, y_true)

    return pd.DataFrame({
        "log_income": log_income,
        "debt_to_income": debt_to_income,
        "credit_history_years": credit_history_years,
        "late_payments": late_payments,
        "employment_years": employment_years,
        "zone_code": zone_code,
        GROUP: g,
        "y_true": y_true,
        "y_hist": y_hist,
    })


# Tədqiqatda istifadə olunan ssenarilər (ablasiya: hər mexanizm ayrıca, sonra birlikdə)
SCENARIOS: dict[str, BiasConfig] = {
    "S0": BiasConfig(name="Qərəzsiz (nəzarət)"),
    "S1": BiasConfig(name="Tarixi qərəz", label_bias=0.25),
    "S2": BiasConfig(name="Ölçmə qərəzi", measurement_bias=1.0),
    "S3": BiasConfig(name="Təmsil qərəzi", minority_share=0.05),
    "S4": BiasConfig(name="Proksi + tarixi qərəz", label_bias=0.25, proxy_strength=0.8),
    "S5": BiasConfig(name="Birgə (real ssenari)", minority_share=0.20, label_bias=0.20,
                     measurement_bias=0.6, proxy_strength=0.8),
}
