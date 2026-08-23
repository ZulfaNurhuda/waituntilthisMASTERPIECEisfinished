# ground-truth layer: Arrhenius + first-order degradation
# kinetics capsaicin content di cabai giling.

# Source param:
#     Renate, D., Pratama, F., Yuliati, K., & Priyanto, G. (2014).
#     "Model Kinetika Degradasi Capsaicin Cabai Merah Giling pada
#     Berbagai Kondisi Suhu Penyimpanan." Agritech, 34(3), 330-336.

#     - Degradation first-order kinetics: C(t) = C0 * exp(-k*t)
#     - Arrhenius fit: ln(k) = 27.836 - 9356.3 * (1/T[K])   (R = 0.76)
#     - Aktivasi energi Ea ~= 18.58 kcal/mol
#     - shelf life: 10.62 wk @ 20C, 8.62 wk @ 30C, 8.45 wk @ 40C


from dataclasses import dataclass
import math

R_GAS_CAL = 1.987204  # cal / (mol * K)


@dataclass
class ArrheniusParams:
    ln_A: float          # ln of pre-exponential factor
    Ea_over_R: float      # Ea / R, in Kelvin (this is the Arrhenius plot slope magnitude)
    threshold_fraction: float  # fraction of C0 at which product is "end of shelf life"
    time_unit: str = "week"    # the time unit the rate constant k is expressed in

    @property
    def Ea_kcal_mol(self) -> float:
        return self.Ea_over_R * R_GAS_CAL / 1000.0

    def k_at(self, temp_celsius: float) -> float:
        """First-order rate constant k at a given temperature (deg C)."""
        T_kelvin = temp_celsius + 273.15
        ln_k = self.ln_A - self.Ea_over_R / T_kelvin
        return math.exp(ln_k)


def _calibrate_threshold_from_30C(ln_A: float, Ea_over_R: float,
                                   shelf_life_30C_weeks: float) -> float:
    """Solve for the critical retention fraction that reproduces the
    paper's reported shelf life at 30C, given the fitted k(30C)."""
    T = 30.0 + 273.15
    k_30 = math.exp(ln_A - Ea_over_R / T)
    # first order: C(t)/C0 = exp(-k*t)  =>  threshold = exp(-k*t)
    threshold = math.exp(-k_30 * shelf_life_30C_weeks)
    return threshold


_THRESHOLD = _calibrate_threshold_from_30C(
    ln_A=27.836, Ea_over_R=9356.3, shelf_life_30C_weeks=8.62
)

CABAI_GILING = ArrheniusParams(
    ln_A=27.836,
    Ea_over_R=9356.3,
    threshold_fraction=_THRESHOLD,
    time_unit="week",
)


@dataclass
class MicrobialSafetyParams:

    # Source: Esmeralda, M., Renate, D., & Rahmi, S. L. "Pengaruh Suhu dan
    # Lama Penyimpanan terhadap Karakteristik Produk Cabai Merah Giling."
    # Universitas Jambi. (Shares an author with the Renate et al. 2014
    # capsaicin paper -- same product formulation: cabai giling with 6%
    # garam, 0.05% natrium benzoat, 0.5% asam sitrat.)

    t_ref_days: float
    T_ref_celsius: float
    Ea_over_R: float  # K

    @property
    def Ea_kcal_mol(self) -> float:
        return self.Ea_over_R * R_GAS_CAL / 1000.0

    def shelf_life_hours_at(self, temp_celsius: float) -> float:
        T = temp_celsius + 273.15
        T_ref = self.T_ref_celsius + 273.15
        days = self.t_ref_days * math.exp(self.Ea_over_R * (1.0 / T - 1.0 / T_ref))
        return days * 24.0


# Calibrated on the two measured points (5C -> 28.5 days, 30C -> 14.2 days);
# t_ref anchored at 30C since that's the directly-measured ambient point.
CABAI_GILING_MICROBIAL = MicrobialSafetyParams(
    t_ref_days=14.2,
    T_ref_celsius=30.0,
    Ea_over_R=2335.3,
)


def remaining_shelf_life_hours_microbial(temp_history_celsius, hours_per_step: float, params: MicrobialSafetyParams = CABAI_GILING_MICROBIAL) -> float:

    if len(temp_history_celsius) == 0:
        raise ValueError("temp_history_celsius must not be empty")

    consumed_fraction = 0.0
    for temp in temp_history_celsius:
        sl_hours = params.shelf_life_hours_at(temp)
        consumed_fraction += hours_per_step / sl_hours

    if consumed_fraction >= 1.0:
        return 0.0

    last_temp = temp_history_celsius[-1]
    sl_hours_last = params.shelf_life_hours_at(last_temp)
    return max(0.0, (1.0 - consumed_fraction) * sl_hours_last)


def remaining_shelf_life_hours(temp_history_celsius, hours_per_step: float, params: ArrheniusParams = CABAI_GILING) -> float:

    if len(temp_history_celsius) == 0:
        raise ValueError("temp_history_celsius must not be empty")

    step_weeks = hours_per_step / (24.0 * 7.0)

    remaining_fraction = 1.0
    for temp in temp_history_celsius:
        k_i = params.k_at(temp)
        remaining_fraction *= math.exp(-k_i * step_weeks)

    if remaining_fraction <= params.threshold_fraction:
        return 0.0

    last_temp = temp_history_celsius[-1]
    k_last = params.k_at(last_temp)
    if k_last <= 0:
        return float("inf")

    remaining_weeks = math.log(remaining_fraction / params.threshold_fraction) / k_last
    return max(0.0, remaining_weeks * 7.0 * 24.0)


if __name__ == "__main__":
    p = CABAI_GILING
    print(f"Ea = {p.Ea_kcal_mol:.2f} kcal/mol")
    print(f"calibrated threshold fraction (anchored at 30C) = {p.threshold_fraction:.3f}")
    for t in (10, 20, 30, 40):
        print(f"k({t}C) = {p.k_at(t):.5f} per week")

    const_30 = [30.0] * 24 
    rem = remaining_shelf_life_hours([30.0], hours_per_step=1.0, params=p)
    print(f"remaining shelf life after 1 hour at 30C (fresh product): {rem/24:.2f} days "
          f"(~{rem/24/7:.2f} weeks, sanity check vs 8.62 wk reference)")
