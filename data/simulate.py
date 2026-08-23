
# Usage:
#     python simulate.py                                  # default
#     python simulate.py,seed 7,n-per-scenario 600     # reproducible
#     python simulate.py,min-hours 6,max-hours 120,output data/v2.csv


import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kinetics import remaining_shelf_life_hours_microbial, CABAI_GILING_MICROBIAL as SHELF_LIFE_PARAMS

HOURS_PER_STEP = 1.0  # one temperature reading per hour


# ============================================================================
# KONFIGURASI PARAMETER SKENARIO
#
# [DATA]   = rentang bersumber dari studi/data resmi yang dikutip di komentar
# [ASUMSI] = keputusan rekayasa berbasis pengetahuan umum (iklim, praktik
#            umum refrigerasi pangan), dinyatakan eksplisit karena tidak ada
#            pengukuran langsung untuk parameter tersebut
# ============================================================================

@dataclass
class ScenarioConfig:
    #,- stable_cold,-
    cold_setpoint_range_c: tuple = (2.0, 6.0)
    # [DATA] Esmeralda, Renate, & Rahmi (2019) menguji cabai giling langsung
    # pada rentang cold storage 4-6C. Diperluas sedikit ke 2-6C mengikuti
    # rentang refrigerasi pangan segar yang umum dipakai industri.
    cold_noise_std_c: float = 0.4
    # [ASUMSI] simpangan kecil di sekitar setpoint untuk cold storage yang
    # terawat baik. Tidak ada data spesifik; nilai dipilih agar variasi
    # terlihat realistis tanpa mendominasi sinyal setpoint itu sendiri.

    #,- power_outage,-
    outage_ambient_range_c: tuple = (28.0, 34.0)
    # [ASUMSI] suhu ambien dataran rendah tropis Indonesia saat listrik padam.
    outage_duration_hours_range: tuple = (2, 6)
    # [DATA] ESDM/PLN: durasi pemadaman terencana umumnya 2-6 jam, gangguan
    # besar tak terencana bisa mencapai 6-8 jam.
    outage_count_range: tuple = (1, 3)
    # [ASUMSI - STRESS TEST] frekuensi ini SENGAJA lebih tinggi dari rata-rata
    # nasional (SAIDI 5,34 jam/pelanggan/TAHUN, ESDM 2024) untuk menguji
    # ketahanan model pada kondisi wilayah dengan keandalan listrik rendah
    # (Sulawesi 11,76 jam/tahun, Sumatera 11,10 jam/tahun per data yang sama).
    # Ini adalah skenario uji-batas (worst case), bukan kondisi tipikal,
    # nyatakan ini eksplisit di proposal, jangan disajikan sebagai rata-rata.
    outage_noise_std_c: float = 0.3
    # [ASUMSI] simpangan sensor selama/setelah periode outage.

    #,- hot_truck_transit,-
    truck_base_temp_range_c: tuple = (22.0, 30.0)
    # [ASUMSI] suhu bak truk non-refrigerasi, berdasar pengetahuan umum iklim
    # tropis dataran rendah Indonesia. Belum ada studi pengukuran suhu
    # langsung di dalam truk pengangkut cabai yang ditemukan.
    truck_diurnal_amplitude_c: float = 4.0
    # [ASUMSI] amplitudo variasi suhu siang-malam untuk kendaraan tanpa isolasi.
    truck_noise_std_c: float = 0.8
    truck_dock_spike_start_range_c: tuple = (3.0, 8.0)
    truck_dock_spike_end_range_c: tuple = (2.0, 6.0)
    # [ASUMSI] lonjakan suhu singkat saat bongkar-muat; titik awal (memuat,
    # biasanya lebih lama terpapar) diberi rentang sedikit lebih tinggi
    # dari titik akhir (bongkar, biasanya lebih cepat).

    #,- market_display,-
    market_temp_range_c: tuple = (26.0, 33.0)
    # [ASUMSI] suhu rak pasar terbuka tanpa pendingin, iklim tropis.
    market_noise_std_c: float = 1.2

    #,- sensor gaps & noise,-
    max_gap_fraction: float = 0.15
    # [ASUMSI - STRESS TEST] hingga 15% pembacaan hilang, untuk menguji
    # robustness pipeline terhadap sensor mati/paket data hilang. Bukan
    # diturunkan dari statistik kegagalan sensor IoT yang terukur, pilihan
    # rekayasa untuk stress-test, nyatakan sebagai itu.
    temp_plausible_range_c: tuple = (-5.0, 45.0)
    # [ASUMSI] batas fisik akal sehat untuk memotong outlier ekstrem
    # (bukan dari spesifikasi sensor tertentu).


def _clip_temp(t, cfg: ScenarioConfig):
    lo, hi = cfg.temp_plausible_range_c
    return float(np.clip(t, lo, hi))


def scenario_stable_cold(rng, n_hours, cfg: ScenarioConfig):
    ##Cold storage yang berfungsi normal: pita sempit di sekitar titik-set dingin.

    # Distribusi: setpoint ~ Uniform(rentang cfg), dipilih karena kita hanya
    # tahu batas bawah/atas yang masuk akal dari literatur, tanpa alasan untuk
    # menganggap nilai tengah lebih mungkin (prinsip entropi maksimum untuk
    # parameter yang cuma diketahui rentangnya). Noise ~ Normal(0, sigma),
    # variasi sensor/termostat di sekitar titik kerja biasanya mendekati
    # Gaussian (banyak sumber gangguan kecil independen, mengikuti central
    # limit theorem), bukan uniform.
    setpoint = rng.uniform(*cfg.cold_setpoint_range_c)
    noise = rng.normal(0, cfg.cold_noise_std_c, n_hours)
    return [_clip_temp(setpoint + n, cfg) for n in noise]


def scenario_power_outage(rng, n_hours, cfg: ScenarioConfig):
    # Cold storage dengan 1-2 periode listrik padam, suhu drift ke ambien.

    # Distribusi: jumlah outage ~ DiscreteUniform(cfg range), durasi tiap
    # outage ~ DiscreteUniform(cfg range, bersumber data ESDM/PLN). Selama
    # outage, suhu naik linear (bukan langsung lompat) menuju ambien,
    # merepresentasikan kenaikan suhu bertahap akibat kehilangan pendinginan
    # aktif, bukan perubahan instan.
    
    setpoint = rng.uniform(*cfg.cold_setpoint_range_c)
    ambient = rng.uniform(*cfg.outage_ambient_range_c)
    temps = [setpoint] * n_hours
    n_outages = rng.integers(*cfg.outage_count_range)
    for _ in range(n_outages):
        start = rng.integers(0, max(1, n_hours - 6))
        dur = rng.integers(*cfg.outage_duration_hours_range)
        for h in range(start, min(start + dur, n_hours)):
            progress = (h - start + 1) / dur
            temps[h] = setpoint + (ambient - setpoint) * progress
    noise = rng.normal(0, cfg.outage_noise_std_c, n_hours)
    return [_clip_temp(t + n, cfg) for t, n in zip(temps, noise)]


def scenario_hot_truck_transit(rng, n_hours, cfg: ScenarioConfig):
    # Truk tanpa pendingin/isolasi buruk: panas siang hari, dingin malam, ditambah
    # lonjakan suhu bongkar-muat di awal/akhir perjalanan.

    # Distribusi: suhu dasar ~ Uniform(cfg range). Pola diurnal dimodelkan
    # sebagai gelombang sinus terhadap jam-dalam-hari.
    base = rng.uniform(*cfg.truck_base_temp_range_c)
    hours_of_day = np.arange(n_hours) % 24
    diurnal = cfg.truck_diurnal_amplitude_c * np.sin((hours_of_day - 9) / 24.0 * 2 * np.pi)
    temps = base + diurnal
    if n_hours > 2:
        temps[0] += rng.uniform(*cfg.truck_dock_spike_start_range_c)
        temps[-1] += rng.uniform(*cfg.truck_dock_spike_end_range_c)
    noise = rng.normal(0, cfg.truck_noise_std_c, n_hours)
    return [_clip_temp(t + n, cfg) for t, n in zip(temps, noise)]


def scenario_market_display(rng, n_hours, cfg: ScenarioConfig):
    # Produk dipajang di rak pasar terbuka tanpa pendingin.

    # Distribusi: suhu dasar ~ Uniform(cfg range), noise ~ Normal dengan sigma
    # lebih besar dari skenario cold storage, rak terbuka lebih rentan pada
    # gangguan eksternal (angin, sinar matahari langsung, keramaian) sehingga
    # variasi dibuat lebih besar.
    
    base = rng.uniform(*cfg.market_temp_range_c)
    noise = rng.normal(0, cfg.market_noise_std_c, n_hours)
    return [_clip_temp(base + n, cfg) for n in noise]


def scenario_mixed_chain(rng, n_hours, cfg: ScenarioConfig):
    # Cold storage -> transit panas -> rak pasar, digabung berurutan
    # (kasus end-to-end paling realistis untuk satu perjalanan produk).

    # Proporsi durasi tiap fase ~ Uniform pada rentang wajar (bukan dari data
    # logistik terukur, rantai distribusi nyata bisa lebih rumit).
    
    cold_h = int(n_hours * rng.uniform(0.3, 0.5))
    transit_h = int(n_hours * rng.uniform(0.2, 0.4))
    market_h = n_hours - cold_h - transit_h
    parts = []
    parts += scenario_stable_cold(rng, cold_h, cfg) if cold_h > 0 else []
    parts += scenario_hot_truck_transit(rng, transit_h, cfg) if transit_h > 0 else []
    parts += scenario_market_display(rng, market_h, cfg) if market_h > 0 else []
    return parts[:n_hours]


SCENARIOS = {
    "stable_cold": scenario_stable_cold,
    "power_outage": scenario_power_outage,
    "hot_truck_transit": scenario_hot_truck_transit,
    "market_display": scenario_market_display,
    "mixed_chain": scenario_mixed_chain,
}


def inject_sensor_gaps(rng, temps, cfg: ScenarioConfig):
    
    # Jumlah empty ~ Uniform(0, max_gap_fraction) * panjang riwayat, posisi
    # empty dipilih tanpa return (setiap jam sama-sama berpeluang
    # hilang), model paling sederhana untuk kegagalan acak yang tidak
    # berkorelasi dengan waktu, dipakai karena tidak ada data pola kegagalan
    # sensor IoT nyata untuk kasus ini (lihat catatan [ASUMSI - STRESS TEST]).
    
    temps = list(temps)
    n_gaps = int(len(temps) * rng.uniform(0, cfg.max_gap_fraction))
    if n_gaps == 0:
        return temps
    gap_idx = rng.choice(len(temps), size=n_gaps, replace=False)
    for i in gap_idx:
        temps[i] = np.nan
    return temps


def generate_dataset(n_per_scenario=400, min_hours=2, max_hours=96, seed=42,
                      cfg: ScenarioConfig = None):
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(seed)
    rows = []
    for scen_name, scen_fn in SCENARIOS.items():
        for _ in range(n_per_scenario):
            n_hours = int(rng.integers(min_hours, max_hours + 1))
            clean_temps = scen_fn(rng, n_hours, cfg)
            true_remaining_h = remaining_shelf_life_hours_microbial(
                clean_temps, hours_per_step=HOURS_PER_STEP, params=SHELF_LIFE_PARAMS
            )
            observed_temps = inject_sensor_gaps(rng, clean_temps, cfg)
            rows.append({
                "scenario": scen_name,
                "n_hours": n_hours,
                "temp_history_celsius": json.dumps(
                    [None if (isinstance(t, float) and np.isnan(t)) else round(t, 2)
                     for t in observed_temps]
                ),
                "true_remaining_shelf_life_hours": round(true_remaining_h, 2),
            })
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic cold-chain temperature scenarios.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducibility")
    p.add_argument("--n-per-scenario", type=int, default=400, help="samples per scenario type")
    p.add_argument("--min-hours", type=int, default=2, help="minimum history length (hours)")
    p.add_argument("--max-hours", type=int, default=96, help="maximum history length (hours)")
    p.add_argument("--output", type=str, default="synthetic_dataset.csv", help="output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = generate_dataset(
        n_per_scenario=args.n_per_scenario,
        min_hours=args.min_hours,
        max_hours=args.max_hours,
        seed=args.seed,
    )
    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} synthetic scenarios (seed={args.seed}) -> {args.output}")
    print(df.groupby("scenario")["true_remaining_shelf_life_hours"].describe()[["mean", "min", "max"]])