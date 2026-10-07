"""Standard-library integer factorization (Python 3.8+).

    from solver import factor_semiprime
    p, q = factor_semiprime(n)

The input must be a semiprime; the result is an ordered pair of its prime
factors, including multiplicity.  For other composite inputs the two factors
need not be prime.  No factors, input-specific tables, external programs,
native extensions, or network services are used.

There is NO two-second runtime guarantee. Balanced large semiprimes are hard,
and the optional ``timeout`` is a CPU-time budget, not a promise of success.
The main algorithm is a self-initializing quadratic sieve with single/double
large-cofactor matching, bytearray sieving, batch smoothness tests, and
bit-packed Gaussian elimination. Parameters depend on input size, not on a
test corpus. The algorithm is implemented here; FLINT/PARI bindings are not
used, since their compiled dependencies are outside this module's contract.
"""

from bisect import bisect_left
from math import exp, gcd, isfinite, isqrt, log, log2
from operator import index
from random import Random
import re
from time import process_time


def _primes(bound):
    flags = bytearray(b"\x01") * (bound + 1)
    flags[:2] = b"\x00\x00"
    for p in range(2, isqrt(bound) + 1):
        if flags[p]:
            flags[p * p::p] = b"\x00" * ((bound - p * p) // p + 1)
    return [i for i in range(2, bound + 1) if flags[i]]


def _prime64(n):
    """Deterministic Miller--Rabin, used only below 2**64."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = (d & -d).bit_length() - 1
    d >>= s
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        a %= n
        if not a:
            continue
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


def _sqrt_mod(n, p):
    """Tonelli--Shanks; p is prime and n is a nonzero quadratic residue."""
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    q = p - 1
    s = (q & -q).bit_length() - 1
    q >>= s
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    c, t, r = pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while t != 1:
        i, tt = 0, t
        while tt != 1:
            tt = tt * tt % p
            i += 1
        b = pow(c, 1 << (s - i - 1), p)
        r = r * b % p
        c = b * b % p
        t = t * c % p
        s = i
    return r


def _rho(n, rng, steps, check):
    """A bounded Brent/Pollard rho attempt, with batched GCDs."""
    y, c = rng.randrange(1, n), rng.randrange(1, n)
    r, used = 1, 0
    while used < steps:
        check()
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        used += r
        for start in range(0, r, 64):
            previous, product = y, 1
            for _ in range(min(64, r - start)):
                y = (y * y + c) % n
                product = product * (x - y) % n
            used += min(64, r - start)
            g = gcd(product, n)
            if g == n:
                y = previous
                for _ in range(64):
                    y = (y * y + c) % n
                    g = gcd(x - y, n)
                    if g > 1:
                        return g if g < n else None
                return None
            if g > 1:
                return g
        r *= 2
    return None


def _pm1(n, primes, bound, check):
    """Small Pollard p-1 attempt; checkpoint so oversmoothing is recoverable."""
    a, old, powers = 2, 2, []
    for p in primes:
        if p > bound:
            break
        power = p
        while power * p <= bound:
            power *= p
        a = pow(a, power, n)
        powers.append(power)
        if len(powers) == 32:
            check()
            g = gcd(a - 1, n)
            if g == n:
                for power in powers:
                    old = pow(old, power, n)
                    g = gcd(old - 1, n)
                    if g > 1:
                        return g if g < n else None
            if 1 < g < n:
                return g
            old, powers = a, []
    g = gcd(a - 1, n)
    if g == n:
        for power in powers:
            old = pow(old, power, n)
            g = gcd(old - 1, n)
            if g > 1:
                return g if g < n else None
    return g if 1 < g < n else None


def _multiplier(n, primes):
    """Favor quadratic residues at small primes, accounting for growth in k*n."""
    best, best_score = 1, float("-inf")
    for k in range(1, 74, 2):
        if any(k % (p * p) == 0 for p in (3, 5, 7)):
            continue
        kn = k * n
        score = -0.5 * log(k)
        residue = kn % 8
        score += (2 if residue == 1 else 1 if residue == 5 else .5) * log(2)
        for p in primes[1:40]:
            r = kn % p
            if r == 0:
                score += log(p) / p
            elif pow(r, (p - 1) // 2, p) == 1:
                score += 2 * log(p) / (p - 1)
        if score > best_score:
            best, best_score = k, score
    return best


def _product_tree(values):
    tree = [values]
    while len(values) > 1:
        values = [values[i] * values[i + 1] if i + 1 < len(values)
                  else values[i] for i in range(0, len(values), 2)]
        tree.append(values)
    return tree


def _batch_gcd(product, values):
    """Find gcd(product, v) for a batch using a product/remainder tree."""
    tree = _product_tree(values)
    remainders = [product % tree[-1][0]]
    for level in reversed(tree[:-1]):
        remainders = [remainders[i // 2] % v for i, v in enumerate(level)]
    return [gcd(v, r) for v, r in zip(values, remainders)]


def _factor_over_base(value, tree):
    """Yield (factor-base index, exponent) for an already smooth integer."""
    stack = [(len(tree) - 1, 0, value)]
    while stack:
        level, i, v = stack.pop()
        if v == 1:
            continue
        if not level:
            p, exponent = tree[0][i], 0
            while v % p == 0:
                v //= p
                exponent += 1
            yield i, exponent
            continue
        child = i * 2
        # A product contains each prime once; recover repeated powers too.
        left = gcd(v, tree[level - 1][child])
        part = left
        v //= left
        while left != 1:
            left = gcd(v, left)
            part *= left
            v //= left
        if part > 1:
            stack.append((level - 1, child, part))
        if v > 1:
            stack.append((level - 1, child + 1, v))


def _qs(n, rng, check, progress=None):
    ln = log(n)
    # Bound memory for very large inputs; this is not a practical RSA breaker.
    bound = max(300, int(exp(min(log(2_000_000), .42 * (ln * log(ln)) ** .5))))
    primes = _primes(bound)
    check()
    k = _multiplier(n, primes)
    g = gcd(k, n)
    if 1 < g < n:
        return g
    kn = k * n
    fb, roots = [2], [kn & 1]
    for p in primes[1:]:
        r = kn % p
        if r == 0:
            if n % p == 0:
                return p
            fb.append(p)
            roots.append(0)
        elif pow(r, (p - 1) // 2, p) == 1:
            fb.append(p)
            roots.append(_sqrt_mod(r, p))
    check()
    tree = _product_tree(fb)
    base_product = tree[-1][0]
    size = len(fb)
    half = 1 << max(12, min(16, (bound // 2).bit_length()))
    width = 2 * half
    target = isqrt(2 * kn) // half
    # Work with Q(x) = ((A*x+B)**2-k*n)/A, keeping Q small.
    count = max(2, int(log(target) / log(1500)) + 1)
    center = exp(log(target) / count)
    # Small inputs have a smaller factor base: use more factors of A rather
    # than asking for primes outside that base (which can exhaust the pool).
    while center > fb[-1] * .65:
        count += 1
        center = exp(log(target) / count)
    lo = max(1, bisect_left(fb, center * .45))
    hi = min(size, max(lo + count + 5, bisect_left(fb, center * 2.2)))
    pool = [i for i in range(lo, hi) if roots[i] != 0]
    if len(pool) < count:
        pool = [i for i in range(1, size) if roots[i] != 0]
        count = min(count, len(pool))
    large_bound = fb[-1] * 128
    double_bound = fb[-1] ** 2 * 16 if n.bit_length() > 120 else large_bound
    # A byte holds a scaled logarithm. Saturation avoids wraparound.
    scale = min(2.0, 220.0 / (log2(kn) / 2 + log2(half) + 8))
    weights = [max(1, round(log2(p) * scale)) for p in fb]
    translations = {w: bytes(min(255, x + w) for x in range(256))
                    for w in set(weights)}
    patterns = {}
    partials, pivots, relations, used_a = {}, {}, [], set()
    parents, paths, ranks, edges = {}, {}, {}, []
    polys, candidates, matched = 0, 0, 0

    def find(vertex):
        if vertex not in parents:
            parents[vertex], paths[vertex], ranks[vertex] = vertex, 0, 0
            return vertex, 0
        v, path = vertex, 0
        while parents[v] != v:
            path ^= paths[v]
            v = parents[v]
        root, total = v, path
        v = vertex
        while parents[v] != v:
            parent, step = parents[v], paths[v]
            parents[v], paths[v] = root, path
            path ^= step
            v = parent
        return root, total

    def decompose(raw):
        """Expand a partial relation only when it contributes to a cycle."""
        if raw[4] is not None:
            return raw[4]
        u, smooth, amask, negative, _ = raw
        mask, square = amask | int(negative), 1
        for i, exponent in _factor_over_base(smooth, tree):
            bit = 1 << (i + 1)
            exponent += bool(amask & bit)
            mask &= ~bit
            if exponent & 1:
                mask |= bit
            if exponent > 1:
                square = square * pow(fb[i], exponent // 2, n) % n
        raw[4] = u, square, mask
        return raw[4]

    def cycle_relation(raw, left, right):
        """Eliminate large cofactors with a spanning forest and cycle basis."""
        lroot, lpath = find(left)
        rroot, rpath = find(right)
        path = lpath ^ rpath ^ (1 << len(edges))
        edges.append((raw, left, right))
        if lroot != rroot:
            if ranks[lroot] > ranks[rroot]:
                lroot, rroot = rroot, lroot
            parents[lroot], paths[lroot] = rroot, path
            if ranks[lroot] == ranks[rroot]:
                ranks[rroot] += 1
            return None
        u, square, mask, powers = 1, 1, 0, {}
        while path:
            bit = path & -path
            eraw, ep, eq = edges[bit.bit_length() - 1]
            eu, es, em = decompose(eraw)
            u, square = u * eu % n, square * es % n
            overlap = (mask & em) >> 1
            while overlap:
                low = overlap & -overlap
                square = square * fb[low.bit_length() - 1] % n
                overlap ^= low
            mask ^= em
            powers[ep] = powers.get(ep, 0) + 1
            powers[eq] = powers.get(eq, 0) + 1
            path ^= bit
        edges.pop()  # This edge closes a cycle; the spanning forest is unchanged.
        for prime, exponent in powers.items():
            square = square * pow(prime, exponent // 2, n) % n
        return u, square, mask

    def add_relation(u, square, mask):
        combination = 1 << len(relations)
        relations.append((u, square, mask))
        row = mask
        while row:
            lead = row.bit_length() - 1
            if lead not in pivots:
                pivots[lead] = row, combination
                return None
            other, selected = pivots[lead]
            row ^= other
            combination ^= selected
        # A dependency gives X**2 == Y**2 (mod n).
        x, y = 1, 1
        exponents = [0] * size
        while combination:
            bit = combination & -combination
            u, square, parity = relations[bit.bit_length() - 1]
            x, y = x * u % n, y * square % n
            parity >>= 1  # sign is already even by elimination
            while parity:
                low = parity & -parity
                exponents[low.bit_length() - 1] += 1
                parity ^= low
            combination ^= bit
        for p, exponent in zip(fb, exponents):
            if exponent:
                y = y * pow(p, exponent // 2, n) % n
        for difference in (x - y, x + y):
            g = gcd(difference, n)
            if 1 < g < n:
                return g
        return None

    while True:
        check()
        # Choose A near sqrt(2*k*n)/M without embedding size-specific tables.
        best = None
        for _ in range(20):
            indices = rng.sample(pool, count - 1)
            a = 1
            for i in indices:
                a *= fb[i]
            pos = bisect_left(fb, target // a)
            for last in range(max(1, pos - 2), min(size, pos + 3)):
                if last in indices or roots[last] == 0:
                    continue
                aa = a * fb[last]
                distance = abs(log(aa / target))
                if aa not in used_a and (best is None or distance < best[0]):
                    best = distance, aa, indices + [last]
        if best is None:
            # Broaden polynomial choices if a small pool has been exhausted.
            pool = [i for i in range(1, size) if roots[i] != 0]
            continue
        _, a, indices = best
        used_a.add(a)
        amask = sum(1 << (i + 1) for i in indices)
        pieces = []
        for i in indices:
            p = fb[i]
            aa = a // p
            gamma = roots[i] * pow(aa, -1, p) % p
            gamma = min(gamma, p - gamma)
            pieces.append(aa * gamma)
        b = sum(pieces)
        # Unreduced B permits cheap Gray-code root updates below. B is still
        # small compared with sqrt(k*n), since A is about sqrt(k*n)/M.
        inverse = [pow(a, -1, p) if a % p else 0 for p in fb]
        r1 = [(inv * (r - b) + half) % p if inv else 0
              for p, r, inv in zip(fb, roots, inverse)]
        r2 = [(inv * (-r - b) + half) % p if inv else 0
              for p, r, inv in zip(fb, roots, inverse)]
        shifts = [[2 * piece * inv % p for p, inv in zip(fb, inverse)]
                  for piece in pieces[:-1]]
        for variant in range(1 << (len(pieces) - 1)):
            check()
            if variant:
                v = (variant & -variant).bit_length() - 1
                sign = 1 if (variant >> (v + 1)) & 1 else -1
                b += sign * 2 * pieces[v]
                changes = shifts[v]
                r1 = [(r - sign * d) % p for r, d, p in zip(r1, changes, fb)]
                r2 = [(r - sign * d) % p for r, d, p in zip(r2, changes, fb)]
            c = (b * b - kn) // a
            sieve = bytearray(width)
            for i, (p, weight) in enumerate(zip(fb, weights)):
                if inverse[i]:
                    first, second = r1[i], r2[i]
                elif p == 2:
                    first = c & 1
                    second = first
                else:
                    first = (half - c * pow(2 * b, -1, p)) % p
                    second = first
                table = translations[weight]
                sieve[first::p] = sieve[first::p].translate(table)
                if first != second:
                    sieve[second::p] = sieve[second::p].translate(table)
                # Powers of small primes matter disproportionately to scoring.
                if p < 20:
                    modulus, current = p, {first, second}
                    while modulus * p <= 256:
                        extended = set()
                        for root in current:
                            for j in range(p):
                                root2 = root + j * modulus
                                x = root2 - half
                                if ((a * x + 2 * b) * x + c) % (modulus * p) == 0:
                                    extended.add(root2)
                        modulus *= p
                        for root in extended:
                            sieve[root::modulus] = sieve[root::modulus].translate(table)
                        current = extended
            # All candidates are verified exactly; scores are just a filter.
            max_value = max(abs((a * half + 2 * b) * half + c),
                            abs((a * half - 2 * b) * half + c), kn // a)
            threshold = max(1, min(254, int(scale *
                            (log2(max_value) - log2(double_bound) - 5))))
            pattern = patterns.get(threshold)
            if pattern is None:
                pattern = re.compile(b"[\\x%02x-\\xff]" % threshold)
                patterns[threshold] = pattern
            positions = [m.start() - half for m in pattern.finditer(sieve)]
            values = [abs((a * x + 2 * b) * x + c) for x in positions]
            # Q=0 would mean k*n is a square; extract a factor directly.
            if 0 in values:
                g = gcd(isqrt(kn), n)
                if 1 < g < n:
                    return g
                raise ArithmeticError("Degenerate sieve polynomial")
            polys += 1
            candidates += len(values)
            if not values:
                continue
            for x, value, common in zip(positions, values, _batch_gcd(base_product, values)):
                remainder = value
                while common != 1:
                    remainder //= common
                    common = gcd(remainder, common)
                if remainder > double_bound:
                    continue
                cofactor_root = isqrt(remainder)
                cofactor_square = cofactor_root * cofactor_root == remainder
                if remainder > large_bound and not cofactor_square:
                    # A base-2 probable prime can be skipped: missing a relation
                    # never compromises the correctness of returned factors.
                    if pow(2, remainder - 1, remainder) == 1:
                        continue
                    left = _rho(remainder, rng, 1024, check)
                    if left is None:
                        continue
                    right = remainder // left
                    if max(left, right) > large_bound:
                        continue
                else:
                    left, right = 1, remainder
                u = (a * x + b) % n
                negative = ((a * x + 2 * b) * x + c) < 0
                raw = [u, value // remainder, amask, negative, None]
                if cofactor_square:
                    u, square, mask = decompose(raw)
                    square = square * cofactor_root % n
                elif double_bound > large_bound:
                    combined = cycle_relation(raw, left, right)
                    if combined is None:
                        continue
                    u, square, mask = combined
                    matched += 1
                elif remainder != 1:
                    old = partials.get(remainder)
                    if old is None:
                        partials[remainder] = raw
                        continue
                    if raw[:4] == old[:4]:
                        continue
                    old_u, old_square, old_mask = decompose(old)
                    u, square, mask = decompose(raw)
                    u = u * old_u % n
                    square = square * old_square * remainder % n
                    overlap = (mask & old_mask) >> 1
                    while overlap:
                        bit = overlap & -overlap
                        square = square * fb[bit.bit_length() - 1] % n
                        overlap ^= bit
                    mask ^= old_mask
                    matched += 1
                g = add_relation(u, square, mask)
                if g:
                    return g
            if progress and polys % 16 == 0:
                progress({"polynomials": polys, "relations": len(relations),
                          "rank": len(pivots), "base": size,
                          "partials": len(partials) + len(edges), "matched": matched,
                          "candidates": candidates})


def factor_semiprime(n, *, timeout=None):
    """Return ``(p, q)`` with ``1 < p <= q`` and ``p*q == n``.

    ``n`` is an integer or a decimal integer string and is assumed semiprime.
    Squares are supported. Other composites may return composite factors.
    Prime inputs below 2**64 raise ValueError; larger primes are outside the
    contract and may run indefinitely. Set timeout to bound CPU time, raising
    TimeoutError when no factor has been found (with bounded check latency).
    A fresh deterministic local PRNG makes runs reproducible and independent.
    """
    started = process_time()
    if isinstance(n, bool):
        raise TypeError("n must be an integer or decimal integer string")
    if isinstance(n, str):
        n = int(n, 10)
    else:
        n = index(n)
    if n < 4:
        raise ValueError("n must be a semiprime >= 4")
    if timeout is not None:
        timeout = float(timeout)
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number")
    deadline = None if timeout is None else started + timeout

    def check():
        if deadline is not None and process_time() >= deadline:
            raise TimeoutError("CPU-time budget exhausted before finding factors")

    def finish(p):
        q, remainder = divmod(n, p)
        if remainder or min(p, q) <= 1:
            raise ArithmeticError("Invalid factorization")
        return (p, q) if p <= q else (q, p)

    check()
    if n % 2 == 0:
        return finish(2)
    root = isqrt(n)
    if root * root == n:
        return finish(root)
    small = _primes(2000)
    for p in small[1:]:
        if n % p == 0:
            if n == p:
                raise ValueError("n is prime, not semiprime")
            return finish(p)
    if n < 1 << 64 and _prime64(n):
        raise ValueError("n is prime, not semiprime")
    # Fermat is extremely effective for nearly equal factors, at any size.
    a = root + 1
    difference = a * a - n
    for _ in range(512):
        b = isqrt(difference)
        if b * b == difference:
            return finish(a - b)
        difference += 2 * a + 1
        a += 1
    check()
    rng = Random(n)
    factor = _rho(n, rng, 8192 if n.bit_length() > 80 else 65536, check)
    if factor:
        return finish(factor)
    factor = _pm1(n, small, 2000, check)
    if factor:
        return finish(factor)
    return finish(_qs(n, rng, check))


# Short names for callers that expect a solver/factor entry point.
solve = factor = factor_semiprime


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n", help="semiprime in decimal")
    parser.add_argument("--timeout", type=float, help="CPU-time budget in seconds")
    args = parser.parse_args()
    start = process_time()
    try:
        p, q = factor_semiprime(args.n, timeout=args.timeout)
    except (TypeError, ValueError, TimeoutError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps({"p": str(p), "q": str(q),
                      "cpu_seconds": process_time() - start}))
