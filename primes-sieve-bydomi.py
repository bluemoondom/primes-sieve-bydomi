# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:19:33 2026

@author: dominika

Block sieve (a port of the T-SQL procedure) with optional Miller-Rabin and GPU.

The blocks (4n-1)(2n-1): block n tests the odd divisor d = 2n-1. One pass of the
blocks, started from a prime, marks every composite number in the window
[mynum, @firstblock), where @firstblock = mynum + 2*n0 + 2. The first unmarked
odd number is the next prime; each iteration starts again from that prime.

BACKEND:
  "numba"  - FASTEST: the same loop compiled to machine code (Numba),
             @a stored as a byte map of the window instead of a set
  "auto"   - Numba if installed, otherwise GPU (CuPy), otherwise NumPy
  "gpu"    - force the GPU (CuPy, NVIDIA cards only)
  "numpy"  - batched computation on the CPU
  "python" - the original row-by-row version (the only one that supports
             sql_compatible=True)
All variants give the same result (checked by compare()).

MR (Miller-Rabin) is optional:
  MR = True  - START does not have to be prime (the nearest prime below it is
               used), every number found is verified, both checks run at the end
  MR = False - pure sieve: START must be prime, nothing is verified
               (e.g. when checking against your own list of primes)

MODE:
  "window" - one pass returns ALL primes in the window [p, @firstblock);
             the next pass starts from the largest of them (fast: about 1 µs
             per prime at 10^15 instead of ~0.9 s)
  "prime"  - the original way: one pass = one next prime

Output: every iteration prints the prime found (or the window summary), the gap
and the width of the window; the final summary shows the smallest and largest
gap and the smallest and largest window.
"""
import math
import os
import time
from math import isqrt
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING, getcontext
import numpy as np

getcontext().prec = 80

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    GPU_OK = True
except Exception:
    cp = None
    GPU_OK = False

try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

INT64_LIMIT = 3 * 10**18    # the batched/numba versions use int64: mynum must be smaller


def _sql_div(a, b, scale):
    """Mimic SQL Server: quotient to 6 decimal places, then stored as NUMERIC(x, scale)."""
    q = (Decimal(a) / Decimal(b)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return q.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_UP)


def is_prime(n):
    """Deterministic Miller-Rabin (reliable for n < 3.3 * 10^24)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
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


def previous_prime(x):
    """Largest prime smaller than x (Miller-Rabin)."""
    x -= 2 if x % 2 else 1
    while not is_prime(x):
        x -= 2
    return x


def first_block(mynum):
    """Smallest numblock with (4n-1)(2n-1) >= mynum.
    Same result as the SQL loop WHILE @maxblock < @mynum, but instant."""
    n = max(1, isqrt(mynum // 8) - 2)
    while (4 * n - 1) * (2 * n - 1) < mynum:
        n += 1
    return n


if NUMBA_OK:
    def _jit(f):
        # cache on disk when possible; otherwise (e.g. Spyder, unsaved file,
        # folder without write access) compile in memory only - same result
        try:
            return njit(cache=True)(f)
        except RuntimeError:
            return njit(f)
else:
    def _jit(f):
        return f


@_jit
def _procedure_kernel(mynum, numblock, firstblock, window_map, base, warn, nwarn):
    """Body of the procedure for the "numba" backend - line by line the same as
    _procedure_python / _scalar. @a is a byte map of odd numbers:
    window_map[(x - base) // 2] = 1  <=>  x is in @a.
    Returns (lownumber, number of warnings)."""
    nmax = mynum // 2
    length = window_map.shape[0]
    low1 = 0
    low2 = 0
    attention = 0
    stop = False
    low = -1
    while numblock <= nmax and not stop and attention < 2:
        d = 2 * numblock - 1
        q = (mynum + d - 1) // d
        if q % 2 == 0:
            q += 1
        low = d * q                      # @lownumber
        high = low - 2 * d               # @highnumber
        if high == d:
            attention += 1
        if high <= firstblock:
            i = (low - base) >> 1
            if i < length:
                window_map[i] = 1
        if low1 - low2 == low - low1:
            diff1 = low - low1
            if diff1 > 0:
                x = low + diff1
                while x < firstblock:
                    window_map[(x - base) >> 1] = 1
                    x += diff1
            else:
                if nwarn < warn.shape[0]:
                    warn[nwarn, 0] = numblock
                    warn[nwarn, 1] = 1
                nwarn += 1
            de = abs(diff1) - 4
            if de == 0:
                if nwarn < warn.shape[0]:
                    warn[nwarn, 0] = numblock
                    warn[nwarn, 1] = 2
                nwarn += 1
            else:
                jump = (mynum + de - 1) // de
                if jump - numblock > 5:
                    numblock = jump - 1  # 1 is added later
        low2 = low1
        low1 = low
        if mynum == low or mynum == high:
            stop = True
        numblock += 1
    # the fill by 6 is done by the caller (it needs the previous @lownumber too)
    return low, nwarn


@_jit
def _fill_by_six_kernel(start, firstblock, window_map, base):
    x = start
    while x < firstblock:
        window_map[(x - base) >> 1] = 1
        x += 6


@_jit
def _first_free(window_map, start):
    """Index of the first zero from position start (search for the next prime)."""
    i = start
    while i < window_map.shape[0] and window_map[i]:
        i += 1
    return i


class Sieve:
    def __init__(self, sql_compatible=False, backend="auto", batch=4096):
        self.sql = sql_compatible
        self.a = set()          # table @a - NOT cleared between numbers (DELETE is commented out)
        self.lownumber = None   # @lownumber is not reset between calls
        self.warnings = []      # cases where SQL would fail / loop forever
        self.last_window = None # width of the window of the last call (@firstblock - @mynum)
        backend = backend.lower()
        if backend == "auto":
            backend = "numba" if NUMBA_OK else ("gpu" if GPU_OK else "numpy")
        if backend == "numba" and not NUMBA_OK:
            raise RuntimeError("Numba is not installed (pip install numba)")
        if backend == "gpu" and not GPU_OK:
            raise RuntimeError("GPU not available (CuPy or an NVIDIA card is missing)")
        if sql_compatible:
            backend = "python"  # NUMERIC rounding is only in the row-by-row version
        self.backend = backend
        self.xp = cp if backend == "gpu" else np
        self.batch = batch
        # backend "numba": @a as a byte map of odd numbers starting at self.base
        self.window_map = np.zeros(0, dtype=np.uint8)
        self.base = None
        self._warn = np.zeros((1000, 2), dtype=np.int64)

    # ---------------- backend "numba" ----------------
    def _procedure_numba(self, mynum):
        """The same procedure, compiled with Numba. @a is not cleared: the map is
        only shifted to start at mynum (smaller numbers are never queried again)."""
        numblock = first_block(mynum)
        firstblock = mynum + 2 * numblock + 2
        self.last_window = firstblock - mynum
        width = firstblock - mynum
        needed = width + 2                       # window + the same reserve above it (in odd numbers)
        if self.base is None:
            self.base = mynum
            self.window_map = np.zeros(needed, dtype=np.uint8)
        else:
            shift = (mynum - self.base) >> 1
            rest = self.window_map[shift:] if shift < self.window_map.shape[0] else self.window_map[:0]
            if rest.shape[0] < needed:
                rest = np.concatenate((rest, np.zeros(needed - rest.shape[0], dtype=np.uint8)))
            self.window_map = np.ascontiguousarray(rest)
            self.base = mynum
        low, nwarn = _procedure_kernel(mynum, numblock, firstblock, self.window_map,
                                       self.base, self._warn, 0)
        for k in range(min(nwarn, self._warn.shape[0])):
            code = "diff1 <= 0" if self._warn[k, 1] == 1 else "division by zero"
            self.warnings.append((mynum, int(self._warn[k, 0]), code))
        if low >= 0:
            self.lownumber = low
        if self.lownumber is not None and self.lownumber + 6 >= self.base:
            _fill_by_six_kernel(self.lownumber + 6, firstblock, self.window_map, self.base)

    def in_a(self, x):
        """Same as 'x in @a' (for the numba backend)."""
        if self.backend != "numba":
            return x in self.a
        i = (x - self.base) >> 1
        return 0 <= i < self.window_map.shape[0] and bool(self.window_map[i])

    def next_candidate(self, mynum):
        """First odd number above mynum that is not in @a (outer SQL loop)."""
        if self.backend == "numba" and self.base == mynum:
            i = _first_free(self.window_map, 1)
            if i < self.window_map.shape[0]:
                return self.base + 2 * i
        count_m = 0
        while True:
            count_m += 2
            if not self.in_a(mynum + count_m):
                return mynum + count_m

    # ---------------- original row-by-row version ----------------
    def _floor_div(self, a, b):     # FLOOR(@diffnum), @diffnum NUMERIC(37,1)
        if self.sql:
            return int(_sql_div(a, b, 1).to_integral_value(ROUND_FLOOR))
        return a // b

    def _ceil_div(self, a, b):      # CEILING(@mynum / (ABS(@diff1) - 4))
        if self.sql:
            return int(_sql_div(a, b, 6).to_integral_value(ROUND_CEILING))
        return -(-a // b)

    def procedure(self, mynum):
        if self.backend == "numba" and mynum < INT64_LIMIT:
            return self._procedure_numba(mynum)
        if self.backend not in ("python", "numba") and mynum < INT64_LIMIT:
            return self._procedure_batch(mynum)
        return self._procedure_python(mynum)

    def _procedure_python(self, mynum):
        """Body between --start of procedure and --end of procedure, row by row."""
        attention = 0
        low1 = low2 = 0
        stop = False

        numblock = first_block(mynum)
        maxblock = (4 * numblock - 1) * (2 * numblock - 1)
        block = 4 * numblock - 2
        firstblock = mynum + 2 * numblock + 2
        self.last_window = firstblock - mynum

        while 2 * numblock <= mynum and not stop and attention < 2:
            difference = maxblock - mynum
            diffnumlow = self._floor_div(difference, block)
            diffnumhigh = diffnumlow + 1
            lownumber = maxblock - diffnumlow * block
            highnumber = maxblock - diffnumhigh * block
            self.lownumber = lownumber

            if highnumber == 2 * numblock - 1:
                attention += 1

            if highnumber <= firstblock:
                self.a.add(lownumber)

            if low1 - low2 == lownumber - low1:
                diff1 = lownumber - low1
                if diff1 > 0:
                    plustoblock = lownumber + diff1
                    while plustoblock < firstblock:
                        self.a.add(plustoblock)
                        plustoblock += diff1
                else:
                    self.warnings.append((mynum, numblock, "diff1 <= 0"))

                divisor = abs(diff1) - 4
                if divisor == 0:
                    self.warnings.append((mynum, numblock, "division by zero"))
                else:
                    jump = self._ceil_div(mynum, divisor)
                    if jump - numblock > 5:
                        numblock = jump - 1   # 1 is added later

            low2 = low1
            low1 = lownumber

            if mynum in (lownumber, highnumber):
                stop = True

            numblock += 1
            maxblock = (4 * numblock - 1) * (2 * numblock - 1)
            block = 4 * numblock - 2

        self._fill_by_six(firstblock)

    def _fill_by_six(self, firstblock):
        # fill multiples by 6 (NULL + 6 in SQL => nothing is inserted)
        if self.lownumber is not None:
            plustoblock = self.lownumber + 6
            if plustoblock < firstblock:
                self.a.update(range(plustoblock, firstblock, 6))

    def _scalar(self, mynum, numblock, firstblock, low1, low2, attention, stop, limit):
        """Row-by-row version for stretches where jumps come every few blocks
        (a batch does not pay off there). Same logic as _procedure_python."""
        nmax = mynum // 2
        add = self.a.add
        it = 0
        while numblock <= nmax and not stop and attention < 2 and it < limit:
            it += 1
            d = 2 * numblock - 1
            q = -(-mynum // d)
            if q % 2 == 0:
                q += 1
            low = d * q
            high = low - 2 * d
            self.lownumber = low
            if high == d:
                attention += 1
            if high <= firstblock:
                add(low)
            if low1 - low2 == low - low1:
                diff1 = low - low1
                if diff1 > 0:
                    if low + diff1 < firstblock:
                        self.a.update(range(low + diff1, firstblock, diff1))
                else:
                    self.warnings.append((mynum, numblock, "diff1 <= 0"))
                de = abs(diff1) - 4
                if de == 0:
                    self.warnings.append((mynum, numblock, "division by zero"))
                else:
                    jump = -(-mynum // de)
                    if jump - numblock > 5:
                        numblock = jump - 1
            low2 = low1
            low1 = low
            if mynum == low or mynum == high:
                stop = True
            numblock += 1
        return numblock, low1, low2, attention, stop

    # ---------------- batched version (GPU / NumPy) ----------------
    def _procedure_batch(self, mynum):
        """Same logic, but hundreds to thousands of blocks at once.

        Uses the fact that lownumber = smallest odd multiple of d = 2n-1 that is
        >= mynum, and highnumber = lownumber - 2d (what SQL computes through
        maxblock and block, just without huge intermediate values). A batch is
        processed as vectors up to the first 'hard' event (jump in numblock,
        @attention, @stop); that event is evaluated exactly as in SQL and the
        next batch continues."""
        xp = self.xp
        numblock = first_block(mynum)
        firstblock = mynum + 2 * numblock + 2
        self.last_window = firstblock - mynum
        nmax = mynum // 2
        low1 = low2 = 0
        attention = 0
        stop = False
        K = self.batch

        while numblock <= nmax and not stop and attention < 2:
            cnt = min(K, nmax - numblock + 1)
            # --- heavy arithmetic (on the GPU if available) ---
            ns_x = xp.arange(numblock, numblock + cnt, dtype=xp.int64)
            d_x = 2 * ns_x - 1
            q_x = (mynum + d_x - 1) // d_x
            q_x += (q_x % 2 == 0)
            low_x = d_x * q_x
            if xp is np:
                ns, d, low = ns_x, d_x, low_x
            else:
                ns, d, low = (cp.asnumpy(v) for v in (ns_x, d_x, low_x))
            high = low - 2 * d

            # --- events ---
            ext = np.concatenate((np.array([low2, low1], dtype=np.int64), low))
            prev1, prev2 = ext[1:-1], ext[:-2]
            prog = (prev1 - prev2) == (low - prev1)
            diff1 = low - prev1
            de = np.abs(diff1) - 4
            de_safe = np.where(de == 0, 1, de)
            jump_to = -((-mynum) // de_safe)
            jump = prog & (de != 0) & (jump_to - ns > 5)
            att = high == d
            stp = (low == mynum) | (high == mynum)
            hard = jump | att | stp
            idx = np.flatnonzero(hard)
            m = int(idx[0]) + 1 if idx.size else cnt      # process [0, m)

            # --- insert into @a ---
            sel = high[:m] <= firstblock
            self.a.update(low[:m][sel].tolist())

            pm = prog[:m]
            if pm.any():
                p_i = np.flatnonzero(pm)
                dd = diff1[p_i]
                for i in p_i[dd <= 0]:
                    self.warnings.append((mynum, int(ns[i]), "diff1 <= 0"))
                for i in p_i[de[p_i] == 0]:
                    self.warnings.append((mynum, int(ns[i]), "division by zero"))
                ok = dd > 0
                start = low[p_i][ok] + dd[ok]
                step = dd[ok]
                count = np.maximum(0, (firstblock - start + step - 1) // step)
                total = int(count.sum())
                if total:
                    first = np.repeat(start, count)
                    st = np.repeat(step, count)
                    offset = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
                    self.a.update((first + offset * st).tolist())

            # --- state after the last processed position ---
            last = m - 1
            self.lownumber = int(low[last])
            low2 = int(low[last - 1]) if last >= 1 else low1
            low1 = int(low[last])
            if idx.size:
                if att[last]:
                    attention += 1
                if stp[last]:
                    stop = True
                if jump[last]:
                    numblock = int(jump_to[last])     # jump - 1, then + 1
                else:
                    numblock = int(ns[last]) + 1
                if m < 64 and not stop and attention < 2:
                    # dense jumps: a batch does not pay off, go row by row for a while
                    numblock, low1, low2, attention, stop = self._scalar(
                        mynum, numblock, firstblock, low1, low2, attention, stop, 50000)
                    K = self.batch
                else:
                    K = min(1 << 20, K * 2)
            else:
                numblock = int(ns[last]) + 1
                K = min(1 << 20, K * 2)

        self._fill_by_six(firstblock)


def search(start=11, count_p=2000, sql_compatible=False, backend="auto",
           verbose=True, mr=True):
    """Outer loop: returns the list @a_m (primes found) and the sieve.

    Each iteration finds the next prime and the next iteration starts from it.
    mr=True  - START is moved to the nearest prime below it if needed and
               every number found is verified with Miller-Rabin.
    mr=False - no Miller-Rabin; START must be prime.
    The sieve keeps s.history = [(previous prime, prime found, gap, window), ...]."""
    s = Sieve(sql_compatible, backend)
    s.history = []
    if mr and not is_prime(start):
        # the sieve must continue from a prime, otherwise @a fills with errors
        mynum_m = previous_prime(start)
    else:
        mynum_m = start
    a_m = []
    if verbose:
        print(f"backend: {s.backend}, Miller-Rabin: {'yes' if mr else 'no'}, "
              f"starting from {mynum_m}")
        print(f"{'#':>6}  {'prime':>22}  {'gap':>6}  {'window':>14}   time")
    t0 = time.time()
    for i in range(count_p):
        previous = mynum_m
        s.procedure(mynum_m)
        mynum_m = s.next_candidate(mynum_m)
        a_m.append(mynum_m)
        s.history.append((previous, mynum_m, mynum_m - previous, s.last_window))
        if verbose:
            check = "" if not mr or is_prime(mynum_m) else "   !!! NOT PRIME"
            print(f"{i + 1:>5}.  {mynum_m:>22}  {mynum_m - previous:>6}  "
                  f"{s.last_window:>14,}   ({time.time() - t0:.1f} s){check}")
    return a_m, s


def gap_window_summary(history):
    """Smallest and largest gap and window found in one run."""
    if not history:
        return
    g_min = min(history, key=lambda r: r[2])
    g_max = max(history, key=lambda r: r[2])
    w_min = min(history, key=lambda r: r[3])
    w_max = max(history, key=lambda r: r[3])
    avg = sum(r[2] for r in history) / len(history)
    ln_p = math.log(history[-1][1])
    print("\nGap and window summary")
    print(f"  smallest gap : {g_min[2]:>6}   between {g_min[0]} and {g_min[1]}")
    print(f"  largest gap  : {g_max[2]:>6}   between {g_max[0]} and {g_max[1]}"
          f"   (gap / (ln p)^2 = {g_max[2] / math.log(g_max[0]) ** 2:.4f})")
    print(f"  average gap  : {avg:>9.2f}   (ln p = {ln_p:.2f})")
    print(f"  smallest window: {w_min[3]:>14,}   from {w_min[0]}")
    print(f"  largest window : {w_max[3]:>14,}   from {w_max[0]}")
    print(f"  largest gap uses {100 * g_max[2] / g_max[3]:.5f} % of its window")


def harvest_window(p, width=None):
    """ONE pass of the blocks from the prime p (fresh @a, numba backend).
    Returns (primes, width): all primes in the window (p, p + width), in order.
    width=None -> the original window 2*n0 + 2 (= @firstblock - @mynum).
    The procedure itself is unchanged; only the result is read differently:
    instead of the first unmarked number we take ALL unmarked numbers."""
    if not NUMBA_OK or p >= INT64_LIMIT:
        raise RuntimeError("window mode needs Numba and p < 3e18")
    n0 = first_block(p)
    if width is None:
        width = 2 * n0 + 2
    firstblock = p + width
    window_map = np.zeros(width // 2 + 2, dtype=np.uint8)
    warn = np.zeros((10, 2), dtype=np.int64)
    low, _ = _procedure_kernel(p, n0, firstblock, window_map, p, warn, 0)
    if low >= 0:
        _fill_by_six_kernel(low + 6, firstblock, window_map, p)
    slots = (firstblock - p + 1) // 2                  # odd numbers p + 2i < @firstblock
    idx = np.flatnonzero(window_map[1:slots] == 0) + 1
    return p + 2 * idx.astype(np.int64), width


def _default_folder():
    """Folder of this script (or the current folder, e.g. in an interactive console)."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def _last_number_in_file(path):
    """Last integer in a text file (reads only the end of the file)."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 4096))
        lines = [ln for ln in f.read().split(b"\n") if ln.strip()]
    return int(lines[-1].split()[0]) if lines else None


def search_windows(start=11, windows=10, mr_check="sample", output_file="auto",
                   resume=True, verbose=True):
    """Window mode: each pass returns all primes of its window,
    the next pass starts from the largest prime found.

    mr_check:    "none"   - pure sieve (START must be prime),
                 "sample" - START may be composite; Miller-Rabin on the first/last
                            500 and 500 random primes of every window,
                 "all"    - Miller-Rabin on every prime (slower than the sieve itself).
    output_file: "auto" -> primes_<start>.txt next to this script, a path, or None.
                 One prime per line; the table and summary go to <name>_summary.txt.
    resume:      if the file already exists, continue after its last prime and append.
    Returns (history, total_count, primes_path, summary_path)."""
    p = start
    if output_file == "auto":
        output_file = os.path.join(_default_folder(), f"primes_{start}.txt")
    primes_path = os.path.abspath(output_file) if output_file else None
    summary_path = primes_path[:-4] + "_summary.txt" if primes_path else None

    mode = "w"
    if primes_path and resume and os.path.exists(primes_path) and os.path.getsize(primes_path) > 0:
        p = _last_number_in_file(primes_path)
        mode = "a"
    elif mr_check != "none" and not is_prime(p):
        p = previous_prime(p)

    out = open(primes_path, mode) if primes_path else None
    log = open(summary_path, mode) if summary_path else None

    def say(text=""):
        if verbose:
            print(text)
        if log:
            log.write(text + "\n")

    rng = np.random.default_rng(1)
    history, total, t0 = [], 0, time.time()
    if primes_path:
        print(f"primes file : {primes_path}" + ("   (continuing, appending)" if mode == "a" else ""))
        print(f"summary file: {summary_path}")
    say(f"\nwindow mode, Miller-Rabin check: {mr_check}, starting from {p}")
    say(f"{'#':>4}  {'from prime':>22}  {'window':>13}  {'primes':>9}  {'largest prime':>22}  "
        f"{'min gap':>7}  {'max gap':>7}  {'MR':>4}   time")
    for w in range(1, windows + 1):
        t = time.time()
        primes, width = harvest_window(p)
        grow = width
        while primes.size == 0:                        # gap longer than the window (only 113, 1327)
            grow *= 2
            primes, _ = harvest_window(p, grow)
        gaps = np.diff(np.concatenate(([p], primes)))
        i_max, i_min = int(np.argmax(gaps)), int(np.argmin(gaps))
        mr = "-"
        if mr_check != "none":
            if mr_check == "all" or primes.size <= 1500:
                test = primes
            else:
                test = np.concatenate((primes[:500], primes[-500:],
                                       rng.choice(primes[500:-500], 500, replace=False)))
            mr = "OK" if all(is_prime(int(x)) for x in test) else "FAIL"
        if out:
            out.write("\n".join(map(str, primes.tolist())) + "\n")
            out.flush()
        prev_p = int(primes[i_max - 1]) if i_max else p
        history.append(dict(start=p, width=width, count=int(primes.size), last=int(primes[-1]),
                            gap_min=int(gaps[i_min]), gap_max=int(gaps[i_max]),
                            gap_max_from=prev_p, mr=mr, seconds=time.time() - t))
        total += int(primes.size)
        say(f"{w:>4}  {p:>22}  {width:>13,}  {primes.size:>9,}  {int(primes[-1]):>22}  "
            f"{int(gaps[i_min]):>7}  {int(gaps[i_max]):>7}  {mr:>4}   ({time.time() - t0:.1f} s)")
        p = int(primes[-1])                            # continue from the largest prime
    window_summary(history, total, say)
    if out:
        out.close()
        say(f"\n{total:,} primes written to {primes_path}")
    if log:
        log.close()
    return history, total, primes_path, summary_path


def window_summary(history, total, say=print):
    """Smallest and largest gap and window over all windows."""
    g_max = max(history, key=lambda r: r["gap_max"])
    g_min = min(history, key=lambda r: r["gap_min"])
    w_min = min(history, key=lambda r: r["width"])
    w_max = max(history, key=lambda r: r["width"])
    secs = sum(r["seconds"] for r in history)
    say("\nGap and window summary")
    say(f"  primes found   : {total:,} in {len(history)} windows, {secs:.1f} s "
        f"({1e6 * secs / max(total, 1):.2f} µs per prime)")
    say(f"  from {history[0]['start']} to {history[-1]['last']}")
    say(f"  smallest gap   : {g_min['gap_min']:>6}")
    say(f"  largest gap    : {g_max['gap_max']:>6}   after {g_max['gap_max_from']}"
        f"   (gap / (ln p)^2 = {g_max['gap_max'] / math.log(g_max['gap_max_from']) ** 2:.4f})")
    say(f"  smallest window: {w_min['width']:>14,}   from {w_min['start']}")
    say(f"  largest window : {w_max['width']:>14,}   from {w_max['start']}")


def _numbers(path):
    """Integers from a text file: the first number on every line (other text is skipped)."""
    with open(path) as f:
        for line in f:
            tok = line.replace(",", " ").replace(";", " ").split()
            for t in tok:
                if t.isdigit():
                    yield int(t)
                    break


def check_against_list(found_path, reference_path, show=20):
    """Compare the primes found with your own list of primes (both sorted, one per line).
    Only the range covered by the found file is compared (reference numbers outside
    it are ignored). Prints the path of both files and the differences."""
    found = _numbers(found_path)
    first = next(found)
    ref = (x for x in _numbers(reference_path) if x >= first)
    extra, missing = [], []                    # extra = found but not in list, missing = in list but not found
    a, b, last, n = first, next(ref, None), first, 0
    while a is not None:
        n += 1
        last = a
        while b is not None and b < a:
            missing.append(b); b = next(ref, None)
        if b == a:
            b = next(ref, None)
        else:
            extra.append(a)
        a = next(found, None)
    while b is not None and b <= last:
        missing.append(b); b = next(ref, None)
    print(f"found file     : {os.path.abspath(found_path)}")
    print(f"reference list : {os.path.abspath(reference_path)}")
    print(f"compared range : {first} .. {last}  ({n:,} primes found)")
    print(f"found but NOT in your list : {len(extra)}   {extra[:show]}")
    print(f"in your list but NOT found : {len(missing)}   {missing[:show]}")
    if not extra and not missing:
        print("RESULT: identical")
    return extra, missing


def compare(start, count_p, backend="auto"):
    """Check that the fast version gives the same as the original row-by-row one:
    same primes, same warnings and same content of @a in the last window."""
    a1, s1 = search(start, count_p, backend="python", verbose=False)
    a2, s2 = search(start, count_p, backend=backend, verbose=False)
    if s2.backend == "numba":
        n = a1[-2] if len(a1) > 1 else start
        f = n + 2 * first_block(n) + 2
        same_a = all(s2.in_a(x) == (x in s1.a) for x in range(n + 2, f, 2))
    else:
        same_a = s1.a == s2.a
    return a1 == a2 and same_a and s1.warnings == s2.warnings and s1.history == s2.history


if __name__ == "__main__":
    MODE = "window"      # "window" = all primes of the window per pass, "prime" = one prime per pass
    START = 1_693_182_318_746_371
    BACKEND = "auto"     # auto = numba if available; otherwise "gpu" / "numpy" / "python" (prime mode)

    # --- window mode ---
    WINDOWS = 10          # number of passes (windows)
    MR_CHECK = "sample"  # "none" / "sample" / "all"
    OUTPUT_FILE = "auto" # "auto" = primes_<START>.txt next to this script; or a path; or None
    RESUME = True        # True = if the file exists, continue after its last prime
    REFERENCE = None     # e.g. r"C:\data\primes.txt" = your list, compared at the end

    # --- prime mode ---
    COUNT_P = 200
    MR = True            # False = no Miller-Rabin (START must then be prime)

    if MODE == "window":
        history, total, primes_path, summary_path = search_windows(
            START, WINDOWS, MR_CHECK, OUTPUT_FILE, RESUME)
        if REFERENCE and primes_path:
            print()
            check_against_list(primes_path, REFERENCE)
    else:
        a_m, s = search(start=START, count_p=COUNT_P, backend=BACKEND, mr=MR)
        last = a_m[-1]

        print(f"\nFirst {COUNT_P} numbers found above {START}:")
        print(a_m)

        gap_window_summary(s.history)

        if MR:
            # 1st SELECT: numbers found that are NOT prime
            wrong = [x for x in a_m if not is_prime(x)]
            print(f"\nNumbers found that are NOT prime ({len(wrong)}):")
            print(wrong)

            # 2nd SELECT: primes in the range (START, last found] that were missed
            found = set(a_m)
            missed = [p for p in range(START + 1, last + 1)
                      if p not in found and is_prime(p)]
            print(f"\nPrimes between {START} and {last} that were not found ({len(missed)}):")
            print(missed)

        if s.warnings:
            print(f"\nWarnings (cases where SQL would fail / loop forever): {len(s.warnings)}")
            print(s.warnings[:10])