# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:29:48 2026

@author: dominika
"""

import time


def je_prvocislo(n):
    """Deterministický Miller-Rabin (spolehlivý pro n < 3.3 * 10^24)."""
    if n < 2:
        return False
    male = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in male:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in male:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# ============================ NASTAVENÍ ============================
OD = 999999999900017
DO = 999999999999989
JEN_PRVOCISLA = False     # True = vypisuj jen čísla, která vyšla True

# ============================ KONTROLA =============================
od = OD if OD % 2 else OD + 1          # začni na lichém čísle
pocet = pocet_prvocisel = 0
t0 = time.perf_counter()

for n in range(od, DO + 1, 2):
    t = time.perf_counter()
    vysledek = je_prvocislo(n)
    dt = time.perf_counter() - t
    pocet += 1
    pocet_prvocisel += vysledek
    if vysledek or not JEN_PRVOCISLA:
        print(f"{n}  {str(vysledek):<5}  {dt * 1e6:8.1f} µs   "
              f"(celkem {time.perf_counter() - t0:.4f} s)")

celkem = time.perf_counter() - t0
print(f"\nZkontrolováno {pocet} lichých čísel od {od} do {DO}, "
      f"prvočísel {pocet_prvocisel}, celkem {celkem:.4f} s")