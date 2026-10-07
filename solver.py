"""Single-file, standard-library semiprime factorization for Python 3.8+.

    from solver import factor_semiprime
    p, q = factor_semiprime(n, timeout=10)

The self-initializing quadratic sieve has an embedded x86-64 Linux fast path.
Its small machine-code kernels are assembled from literal instructions into
anonymous executable memory through ctypes/mmap.
No compiler, binary file, external package, subprocess, network, or stored
factors are required. Other platforms use the portable Python implementation.

Parameters depend on the input's size and quadratic residues, not a corpus.
Native trial division handles 256-bit polynomial values; larger values use
Python integers. There is no fixed runtime guarantee for arbitrary inputs.
The input is assumed semiprime; factors are verified and returned in order.
"""

from bisect import bisect_left
from math import exp, gcd, isfinite, isqrt, log, log2
from operator import index
from random import Random
import re
import struct
import array
import ctypes
import mmap
import platform
import sys
from time import process_time

_NATIVE_AVAILABLE = (sys.platform.startswith("linux") and
                     platform.machine().lower() in ("x86_64", "amd64") and
                     ctypes.sizeof(ctypes.c_void_p) == 8 and
                     array.array("I").itemsize == 4 and array.array("Q").itemsize == 8)


class _Code:
    def __init__(self):
        self.b = bytearray()
        self.labels = {}
        self.fixups = []

    def emit(self, s):
        self.b.extend(bytes.fromhex(s))

    def label(self, s):
        self.labels[s] = len(self.b)

    def jump(self, op, dest):
        self.emit(op)
        self.fixups.append((len(self.b), dest))
        self.b.extend(b'\0' * 4)

    def function(self, nargs):
        for pos, dest in self.fixups:
            self.b[pos:pos+4] = (self.labels[dest]-pos-4).to_bytes(4, 'little', signed=True)
        mem = mmap.mmap(-1, len(self.b), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        mem.write(self.b)
        fun = ctypes.CFUNCTYPE(ctypes.c_uint64, *([ctypes.c_void_p] * nargs))(
            ctypes.addressof(ctypes.c_char.from_buffer(mem)))
        fun.memory = mem
        return fun


def _saturated_sieve_kernel():
    # sieve(uint8_t *s, uint32_t (*pr)[4], size_t n, size_t length)
    c = _Code()
    c.emit('48 85 d2')
    c.jump('0f 84', 'done')
    c.label('prime')
    c.emit('8b 06 44 8b 46 04 44 8b 4e 08 44 8b 56 0c')
    for name in ('one', 'two'):
        # r8d / r9d are the two sieve offsets.
        c.emit('41 39 c8' if name == 'one' else '41 39 c9')
        c.jump('0f 83', name+'end')
        c.label(name)
        c.emit('46 00 14 07 73 05 42 c6 04 07 ff 41 01 c0 41 39 c8' if name == 'one'
               else '46 00 14 0f 73 05 42 c6 04 0f ff 41 01 c1 41 39 c9')
        c.jump('0f 82', name)
        c.label(name+'end')
    c.emit('48 83 c6 10 48 ff ca')
    c.jump('0f 85', 'prime')
    c.label('done')
    c.emit('c3')
    return c.function(4)



def _roots_kernel(add):
    # update(uint32_t (*entries)[4], uint32_t *shifts, size_t count)
    c = _Code()
    c.emit('48 85 d2')
    c.jump('0f 84', 'end')
    c.label('loop')
    c.emit('8b 0f 44 8b 47 04 44 8b 4f 08 8b 06')
    for reg, suffix in ((8, 'first'), (9, 'second')):
        if reg == 9:
            c.emit('41 83 f9 ff')
            c.jump('0f 84', suffix)
        if add:
            c.emit('41 01 c0 41 39 c8' if reg == 8 else '41 01 c1 41 39 c9')
            c.emit('45 89 c3 41 29 cb 45 0f 43 c3' if reg == 8 else '45 89 cb 41 29 cb 45 0f 43 cb')
        else:
            c.emit('41 29 c0' if reg == 8 else '41 29 c1')
            c.emit('45 89 c3 45 8d 1c 0b 45 0f 42 c3' if reg == 8 else '45 89 cb 45 8d 1c 0b 45 0f 42 cb')
        c.label(suffix)
    c.emit('44 89 47 04 44 89 4f 08 48 83 c7 10 48 83 c6 04 48 ff ca')
    c.jump('0f 85', 'loop')
    c.label('end')
    c.emit('c3')
    return c.function(3)


def _gather_kernel():
    # gather(sieve, offsets, length, threshold); length is a multiple of 16.
    c = _Code()
    c.emit('66 0f 6e c9 66 0f 60 c9 66 0f 61 c9 66 0f 70 c9 00')
    c.emit('45 31 c9 45 31 d2')
    c.label('block')
    c.emit('f3 42 0f 6f 04 0f 66 0f da c1 66 0f 74 c1 66 0f d7 c0 85 c0')
    c.jump('0f 84', 'next')
    c.label('hit')
    c.emit('0f bc c8 45 8d 1c 09 46 89 1c 96 41 ff c2')
    c.emit('8d 48 ff 21 c8')
    c.jump('0f 85', 'hit')
    c.label('next')
    c.emit('41 83 c1 10 41 39 d1')
    c.jump('0f 82', 'block')
    c.emit('44 89 d0 c3')
    return c.function(4)

def _whole_sieve_kernel():
    # Four independent byte stores per loop; caller proves scores fit in a byte.
    c = _Code()
    c.emit('53 55 41 54 48 85 d2')
    c.jump('0f 84', 'done')
    c.label('prime')
    c.emit('8b 06 44 8b 46 04 44 8b 4e 08 44 8b 56 0c')
    c.emit('8d 1c 40 8d 2c 85 00 00 00 00 41 89 cc 45 31 db 41 29 dc 45 0f 42 e3')
    for name, index in (('one',8), ('two',9)):
        c.emit('45 39 e0' if index==8 else '45 39 e1')
        c.jump('0f 83', name+'tail')
        c.label(name+'four')
        c.emit('4e 8d 1c 07' if index==8 else '4e 8d 1c 0f')
        c.emit('45 00 13 45 00 14 03 45 00 14 43 45 00 14 1b')
        c.emit('41 01 e8 45 39 e0' if index==8 else '41 01 e9 45 39 e1')
        c.jump('0f 82', name+'four')
        c.label(name+'tail')
        c.emit('41 39 c8' if index==8 else '41 39 c9')
        c.jump('0f 83', name+'end')
        c.label(name+'single')
        c.emit('46 00 14 07 41 01 c0 41 39 c8' if index==8 else '46 00 14 0f 41 01 c1 41 39 c9')
        c.jump('0f 82', name+'single')
        c.label(name+'end')
    c.emit('48 83 c6 10 48 ff ca')
    c.jump('0f 85', 'prime')
    c.label('done')
    c.emit('41 5c 5d 5b c3')
    return c.function(4)


def _blocked_sieve_kernel():
    # Paired roots may exchange order in this private block-work buffer.
    c = _Code()
    c.emit('53 55 41 54 48 85 d2')
    c.jump('0f 84','done')
    c.label('prime')
    c.emit('8b 06 44 8b 46 04 44 8b 4e 08 44 8b 56 0c')
    # Branchlessly put the smaller root first. The sentinel remains second.
    c.emit('45 89 c3 45 39 c8 45 0f 47 c1 45 0f 47 cb')
    c.emit('8d 1c 40 8d 2c 85 00 00 00 00 41 89 cc 45 31 db 41 29 dc 45 0f 42 e3')
    c.emit('41 83 f9 ff')
    c.jump('0f 84','single_root')
    c.emit('45 39 e1')
    c.jump('0f 83','pair_tail')
    c.label('pair_four')
    c.emit('4e 8d 1c 07 45 00 13 45 00 14 03 45 00 14 43 45 00 14 1b')
    c.emit('4e 8d 1c 0f 45 00 13 45 00 14 03 45 00 14 43 45 00 14 1b')
    c.emit('41 01 e8 41 01 e9 45 39 e1')
    c.jump('0f 82','pair_four')
    c.label('pair_tail')
    c.emit('41 39 c9')
    c.jump('0f 83','extra_first')
    c.label('pair_scalar')
    c.emit('46 00 14 07 46 00 14 0f 41 01 c0 41 01 c1 41 39 c9')
    c.jump('0f 82','pair_scalar')
    c.label('extra_first')
    c.emit('41 39 c8')
    c.jump('0f 83','store')
    c.emit('46 00 14 07 41 01 c0')
    c.jump('e9','store')
    c.label('single_root')
    c.emit('45 39 e0')
    c.jump('0f 83','single_tail')
    c.label('single_four')
    c.emit('4e 8d 1c 07 45 00 13 45 00 14 03 45 00 14 43 45 00 14 1b')
    c.emit('41 01 e8 45 39 e0')
    c.jump('0f 82','single_four')
    c.label('single_tail')
    c.emit('41 39 c8')
    c.jump('0f 83','store')
    c.label('single_scalar')
    c.emit('46 00 14 07 41 01 c0 41 39 c8')
    c.jump('0f 82','single_scalar')
    c.label('store')
    c.emit('41 29 c8 44 89 46 04 41 83 f9 ff')
    c.jump('0f 84','stored')
    c.emit('41 29 c9 44 89 4e 08')
    c.label('stored')
    c.emit('48 83 c6 10 48 ff ca')
    c.jump('0f 85','prime')
    c.label('done')
    c.emit('41 5c 5d 5b c3')
    return c.function(4)


def _trial_kernel():
    # factor(values[lo,hi], params[][6], count_odd_primes, output,
    #        prime_index, bound). Each parameter row contains the inverse mod
    # 2**128, maximum 128-bit quotient, maximum 64-bit quotient, and p*p.
    # For odd p, p divides v iff (v*inverse mod 2**bits) <= (2**bits-1)//p.
    # Early p*p termination assumes Q: any odd prime below the base bound
    # dividing Q occurs in the factor base. The caller handles a final base
    # prime. At most 128 output pairs are possible for a positive 128-bit v.
    c = _Code()
    c.emit('53 55 41 54 41 55 41 56 41 57 51')
    c.emit('49 89 f4 49 89 d5 4c 8b 37 4c 8b 7f 08 48 89 cb 31 ed 4d 89 c3 4d 89 ca')
    c.label('prime')
    c.emit('4d 85 ff')
    c.jump('0f 85','start_divide')
    c.emit('4d 3b 74 24 28')
    c.jump('0f 82','done')
    c.label('start_divide')
    c.emit('31 c9')
    c.label('divide')
    c.emit('4d 85 ff')
    c.jump('0f 84','low')
    c.emit('4c 89 f0 49 f7 24 24 49 89 c0 49 89 d1')
    c.emit('4c 89 f8 49 0f af 04 24 49 01 c1')
    c.emit('4c 89 f0 49 0f af 44 24 08 49 01 c1')
    c.emit('4d 3b 4c 24 18')
    c.jump('0f 87','divided')
    c.jump('0f 82','quotient')
    c.emit('4d 3b 44 24 10')
    c.jump('0f 87','divided')
    c.label('quotient')
    c.emit('4d 89 c6 4d 89 cf ff c1')
    c.jump('e9','divide')
    c.label('low')
    c.emit('4c 89 f0 49 0f af 04 24 49 3b 44 24 20')
    c.jump('0f 87','divided')
    c.emit('49 89 c6 ff c1')
    c.jump('e9','divide')
    c.label('divided')
    c.emit('85 c9')
    c.jump('0f 84','advance')
    c.emit('89 2b 89 4b 04 48 83 c3 08')
    c.label('next')
    c.emit('4d 85 ff')
    c.jump('0f 85','advance')
    c.emit('49 83 fe 01')
    c.jump('0f 84','done')
    c.emit('4d 39 d6')
    c.jump('0f 87','advance')
    c.emit('43 8b 04 b3 85 c0')
    c.jump('0f 84','advance')
    c.emit('ff c8 89 03 c7 43 04 01 00 00 00 48 83 c3 08 41 be 01 00 00 00')
    c.jump('e9','done')
    c.label('advance')
    c.emit('49 83 c4 30 ff c5 44 39 ed')
    c.jump('0f 82','prime')
    c.label('done')
    c.emit('4c 89 37 4c 89 7f 08 48 89 d8 48 2b 04 24 48 c1 e8 03 48 83 c4 08')
    c.emit('41 5f 41 5e 41 5d 41 5c 5d 5b c3')
    return c.function(6)


def _wide_trial_kernel():
    # strip(value[4], odd_primes, count, output[][2]) -> output pair count.
    # Four little-endian limbs hold a positive odd value. Unsigned long
    # division tests each prime and removes its full power, then stops as
    # soon as the remainder fits in 128 bits. The existing inverse-based
    # kernel can finish it, scanning the same base: every emitted prime is
    # already exhausted, so the two outputs cannot contain duplicate primes.
    # Output indices are zero-based in odd_primes, just like _trial_kernel.
    # At most 256 output pairs are needed for a positive 256-bit value.
    c = _Code()
    # Preserve callee-saved registers, output origin, and prime count.
    c.emit('53 55 41 54 41 55 41 56 41 57 51 52')
    c.emit('4c 8b 27 4c 8b 6f 08 4c 8b 77 10 4c 8b 7f 18')
    c.emit('48 89 cb 45 31 c0')  # rbx=output, r8=prime index
    c.emit('4c 89 f0 4c 09 f8')  # upper two limbs already zero?
    c.jump('0f 84', 'done')
    c.emit('4c 3b 04 24')
    c.jump('0f 83', 'done')
    c.label('prime')
    c.emit('42 8b 0c 86 31 ed')  # ecx=prime, ebp=exponent
    c.label('divide')
    # Divide from the most significant limb down. Each remainder is < p,
    # so every following 128-by-64 division has a quotient fitting 64 bits.
    # Skip zero leading quotients, especially just above the 128-bit boundary.
    c.emit('31 d2 45 31 c9 45 31 d2 4d 85 ff')
    c.jump('0f 85', 'top')
    c.emit('49 39 ce')  # third limb >= prime?
    c.jump('0f 83', 'third')
    c.emit('4c 89 f2')  # third limb itself is the initial remainder
    c.jump('e9', 'lower')
    c.label('top')
    c.emit('4c 89 f8 48 f7 f1 49 89 c1')
    c.label('third')
    c.emit('4c 89 f0 48 f7 f1 49 89 c2')
    c.label('lower')
    c.emit('4c 89 e8 48 f7 f1 49 89 c3')
    c.emit('4c 89 e0 48 f7 f1 48 85 d2')
    c.jump('0f 85', 'divided')
    c.emit('49 89 c4 4d 89 dd 4d 89 d6 4d 89 cf ff c5')
    c.jump('e9', 'divide')
    c.label('divided')
    c.emit('85 ed')
    c.jump('0f 84', 'advance')
    c.emit('44 89 03 89 6b 04 48 83 c3 08')
    c.label('advance')
    c.emit('49 ff c0 4c 89 f0 4c 09 f8')
    c.jump('0f 84', 'done')
    c.emit('4c 3b 04 24')
    c.jump('0f 82', 'prime')
    c.label('done')
    c.emit('4c 89 27 4c 89 6f 08 4c 89 77 10 4c 89 7f 18')
    c.emit('48 89 d8 48 2b 44 24 08 48 c1 e8 03 48 83 c4 10')
    c.emit('41 5f 41 5e 41 5d 41 5c 5d 5b c3')
    return c.function(4)



def _matrix_kernel():
    # eliminate(row[2*words], basis[columns][2*words], occupied, words)
    # First half of row is parity, second half its relation-history bitset.
    c = _Code()
    c.emit('53 55 41 54 41 55 41 56 41 57')
    c.emit('48 89 fb 49 89 f5 49 89 d4 49 89 ce 49 89 cf 49 ff cf 48 89 cd 48 c1 e5 04')
    c.label('lead')
    c.emit('4a 8b 04 fb 48 85 c0')
    c.jump('0f 85','pivot')
    c.emit('49 ff cf')
    c.jump('0f 89','lead')
    c.emit('b8 01 00 00 00')
    c.jump('e9','done')
    c.label('pivot')
    c.emit('4c 0f bd c0 4d 89 f9 49 c1 e1 06 4d 01 c8')
    c.emit('4d 89 c2 4c 0f af d5 4d 01 ea 45 31 db')
    c.emit('43 80 3c 04 00')
    c.jump('0f 84','insert')
    c.label('xor')
    c.emit('f3 42 0f 6f 04 1b f3 43 0f 6f 0c 1a 66 0f ef c1 f3 42 0f 7f 04 1b')
    c.emit('49 83 c3 10 49 39 eb')
    c.jump('0f 82','xor')
    c.jump('e9','lead')
    c.label('insert')
    c.emit('43 c6 04 04 01')
    c.label('copy')
    c.emit('f3 42 0f 6f 04 1b f3 43 0f 7f 04 1a 49 83 c3 10 49 39 eb')
    c.jump('0f 82','copy')
    c.emit('31 c0')
    c.label('done')
    c.emit('41 5f 41 5e 41 5d 41 5c 5d 5b c3')
    return c.function(4)



def _grow_matrix_basis(basis, flags, old_words, new_words):
    """Grow history capacity without changing stored parity or dependencies."""
    expanded = (ctypes.c_uint64 * (len(flags) * 2 * new_words))()
    old_addr, new_addr = ctypes.addressof(basis), ctypes.addressof(expanded)
    for column, occupied in enumerate(flags):
        if occupied:
            source = old_addr + column * 16 * old_words
            target = new_addr + column * 16 * new_words
            ctypes.memmove(target, source, old_words * 8)
            ctypes.memmove(target + new_words * 8, source + old_words * 8, old_words * 8)
    return expanded


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
    # Compute per-prime constants once; no input factors or persistent table.
    scored = [(p, n % p, (p-1)//2, log(p)/p, 2*log(p)/(p-1))
              for p in primes[1:320]]
    for k in range(1, 512, 2):
        if any(k % (p * p) == 0 for p in (3, 5, 7, 11, 13, 17, 19)):
            continue
        kn = k * n
        if kn % 4 != 1:
            continue
        score = -0.5 * log(k)
        residue = kn % 8
        score += (2 if residue == 1 else 0) * log(2)
        for p, n_mod_p, exponent, zero_weight, residue_weight in scored:
            r = k * n_mod_p % p
            if r == 0:
                score += zero_weight
            elif pow(r, exponent, p) == 1:
                score += residue_weight
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


def _qs_native(n, rng, check, progress=None):
    ln = log(n)
    # Bound memory for very large inputs; this is not a practical RSA breaker.
    bound = max(300, int(exp(min(log(2_000_000), 0.45 * (ln * log(ln)) ** .5))))
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
    # Cache-sized blocks and Gray-code polynomials amortize setup costs.
    half = 1 << max(13, min(17, (bound // 2).bit_length() + 2))
    width = 2 * half
    native_saturated = _saturated_sieve_kernel()
    native_sieve = _blocked_sieve_kernel()
    native_whole = _whole_sieve_kernel()
    block_stop = bisect_left(fb, 8192)
    native_gather = _gather_kernel()
    native_decompose = _trial_kernel()
    small_values = (ctypes.c_uint64*2)()
    small_output = (ctypes.c_uint32*512)()
    # Initialize the wider path only when an actual candidate needs it.
    native_wide = None
    prime_indices = (ctypes.c_uint32 * (bound+1))()
    for i,p in enumerate(fb): prime_indices[p]=i
    inverse_parameters = array.array('Q')
    for p in fb[1:]:
        inv=pow(p,-1,1<<128); limit=((1<<128)-1)//p
        inverse_parameters.extend((inv & ((1<<64)-1),inv>>64,limit & ((1<<64)-1),limit>>64,((1<<64)-1)//p,p*p))
    roots_add, roots_sub = _roots_kernel(True), _roots_kernel(False)
    entries = (ctypes.c_uint32 * (4*size))()
    offsets = (ctypes.c_uint32 * width)()
    entries_addr = ctypes.addressof(entries)
    work_entries = (ctypes.c_uint32 * (4*size))()
    work_addr = ctypes.addressof(work_entries)
    target = isqrt(kn // 2) // half
    # Odd B gives Q(x)=A*x*x+B*x+(B*B-k*n)/(4*A).
    count = max(2, int(log(target) / log(500)) + 1)
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
    # A byte holds a scaled logarithm; the sieve checks its overflow bound.
    scale = min(2.0, 220.0 / (log2(kn) / 2 + log2(half) + 8))
    weights = [max(1, round(log2(p) * scale)) for p in fb]
    max_weight_ratio = max(w / log2(p) for p, w in zip(fb, weights))
    translations = {w: bytes(min(255, x + w) for x in range(256))
                    for w in set(weights)}
    template_stop = bisect_left(fb, 11)
    two_patterns = {}
    block_bytes = 16 * block_stop
    block_entries = work_addr + 16 * template_stop
    block_count = block_stop - template_stop
    whole_entries = entries_addr + 16 * block_stop
    whole_count = size - block_stop
    inverse_address = inverse_parameters.buffer_info()[0]
    matrix_words = (size + 320) // 64
    native_matrix = None
    if (size + 1) * matrix_words * 16 <= 64 * 1024 * 1024:
        native_matrix = _matrix_kernel()
        matrix_row = (ctypes.c_uint64 * (2 * matrix_words))()
        matrix_basis = (ctypes.c_uint64 * ((size + 1) * 2 * matrix_words))()
        matrix_flags = (ctypes.c_ubyte * (size + 1))()
    rank = 0
    partials, pivots, relations, used_a = {}, {}, [], set()
    polys, candidates, matched = 0, 0, 0

    def decompose(raw):
        """Expand a partial relation only when it contributes to a cycle."""
        if raw[4] is not None:
            return raw[4]
        u, packed, amask, negative, _, twos, last = raw
        mask, square = amask | int(negative), 2
        if isinstance(packed, bytes):
            fs = [(0, twos)] if twos else []
            if last:
                fs.append((last, 1))
            fs.extend((i+1, exponent) for i, exponent in struct.iter_unpack("<II", packed))
        else:
            fs = packed  # arbitrary-precision trial division supplied a list
        for i, exponent in fs:
            bit = 1 << (i + 1)
            if exponent & 1:
                mask ^= bit
            exponent += bool(amask & bit)
            if exponent > 1:
                square = square * pow(fb[i], exponent // 2, n) % n
        raw[4] = u, square, mask
        return raw[4]

    def add_relation(u, square, mask):
        nonlocal matrix_words, matrix_row, matrix_basis, rank
        combination = 1 << len(relations)
        relations.append((u, square, mask))
        if native_matrix is not None:
            ci = len(relations)-1
            if ci >= matrix_words * 64:
                new_words = matrix_words * 2
                matrix_basis = _grow_matrix_basis(matrix_basis, matrix_flags,
                                                  matrix_words, new_words)
                matrix_words = new_words
                matrix_row = (ctypes.c_uint64 * (2 * matrix_words))()
            ctypes.memset(matrix_row, 0, 16 * matrix_words)
            data = mask.to_bytes(8 * matrix_words, "little")
            ctypes.memmove(matrix_row, data, len(data))
            matrix_row[matrix_words + (ci >> 6)] = 1 << (ci & 63)
            if not native_matrix(matrix_row, matrix_basis, matrix_flags, matrix_words):
                rank += 1
                return None
            combination = int.from_bytes(ctypes.string_at(
                ctypes.addressof(matrix_row) + 8 * matrix_words, 8 * matrix_words), "little")
        else:
            row = mask
            while row:
                lead = row.bit_length()-1
                if lead not in pivots:
                    pivots[lead] = row, combination
                    rank += 1
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
        if not b & 1:
            b += a
        # Unreduced B permits cheap Gray-code root updates below. B is still
        # small compared with sqrt(k*n), since A is about sqrt(k*n)/M.
        inverse = [pow(2 * a, -1, p) if p != 2 and a % p else 0 for p in fb]
        r1 = [(inv * (r - b) + half) % p if inv else 0
              for p, r, inv in zip(fb, roots, inverse)]
        r2 = [(inv * (-r - b) + half) % p if inv else 0
              for p, r, inv in zip(fb, roots, inverse)]
        shifts = [[2 * piece * inv % p for p, inv in zip(fb, inverse)]
                  for piece in pieces[:-1]]
        for i,p in enumerate(fb):
            entries[4*i] = p
            entries[4*i+1] = r1[i]
            entries[4*i+2] = r2[i] if r2[i] != r1[i] else 0xffffffff
            entries[4*i+3] = weights[i]
        shift_arrays = [array.array('I', row) for row in shifts]
        b_inverses = [pow(b, -1, fb[i]) for i in indices]
        for variant in range(1 << (len(pieces) - 1)):
            check()
            if variant:
                v = (variant & -variant).bit_length() - 1
                sign = 1 if (variant >> (v + 1)) & 1 else -1
                b += sign * 2 * pieces[v]
                # Only this component changes B modulo its prime divisor of A.
                b_inverses[v] = fb[indices[v]] - b_inverses[v]
                (roots_sub if sign > 0 else roots_add)(entries, shift_arrays[v].buffer_info()[0], size)
            c = (b * b - kn) // (4 * a)
            for j, i in enumerate(indices):
                p = fb[i]
                entries[4*i+1] = (half - c * b_inverses[j]) % p
            # 5040 = 2**4 * 3**2 * 5 * 7. Build the frequent small-prime
            # contributions once on this short period, then copy it in C.
            period = 5040
            key = (a & 15, b & 15, c & 15)
            twos = two_patterns.get(key)
            if twos is None:
                residues = [(key[0]*j*j + key[1]*j + key[2]) & 15 for j in range(16)]
                twos = bytearray(weights[0] * ((v & -v).bit_length()-1 if v else 4)
                                 for v in residues)
                two_patterns[key] = twos
            template = twos * (period//16)
            for i in range(1,template_stop):
                p = fb[i]
                table = translations[weights[i]]
                if inverse[i]:
                    first, second = entries[4*i+1], entries[4*i+2]
                    if second == 0xffffffff:
                        second = first
                else:
                    first = entries[4*i+1]
                    second = first
                template[first::p] = template[first::p].translate(table)
                if first != second:
                    template[second::p] = template[second::p].translate(table)
            sieve = template * ((width + period - 1) // period)
            del sieve[width:]
            sieve_addr = ctypes.addressof(ctypes.c_char.from_buffer(sieve))
            max_value = max(abs((a * half + b) * half + c),
                            abs((a * half - b) * half + c), (kn + 4*a-1) // (4*a))
            # Each scored prime power actually divides Q. Thus the score is
            # at most log2(|Q|) * max(weight[p]/log2(p)). Otherwise saturate.
            if max_value.bit_length() * max_weight_ratio < 255:
                ctypes.memmove(work_addr, entries_addr, block_bytes)
                for offset in range(0, width, 32768):
                    native_sieve(sieve_addr+offset, block_entries, block_count,
                                 min(32768, width-offset))
                native_whole(sieve_addr, whole_entries, whole_count, width)
            else:
                native_saturated(sieve_addr, entries_addr+16*template_stop,
                                 size-template_stop, width)
            # Scores only filter candidates; division and dependencies are exact.
            threshold = max(1, min(254, int(scale *
                            (log2(max_value) - log2(large_bound) - 3))))
            nc = native_gather(sieve_addr, offsets, width, threshold)
            positions = [x-half for x in offsets[:nc]]
            values = [abs((a * x + b) * x + c) for x in positions]
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
            for x,value in zip(positions,values):
                native_value = value.bit_length() <= 128
                if native_value:
                    e = (value & -value).bit_length()-1
                    odd = value >> e
                    small_values[0], small_values[1] = odd & 0xffffffffffffffff, odd >> 64
                    nh = native_decompose(small_values, inverse_address,
                                          size-1, small_output, prime_indices, bound)
                    remainder = small_values[0] + (small_values[1] << 64)
                    last_index = prime_indices[remainder] if 1 < remainder <= bound else 0
                    if last_index:
                        remainder = 1
                elif value.bit_length() <= 256:
                    native_value = True
                    e = (value & -value).bit_length()-1
                    odd = value >> e
                    if native_wide is None:
                        native_wide = _wide_trial_kernel()
                        wide_values = (ctypes.c_uint64*4)()
                        wide_primes = array.array('I', fb[1:])
                        wide_address = wide_primes.buffer_info()[0]
                        output_address = ctypes.addressof(small_output)
                    for limb in range(4):
                        wide_values[limb] = (odd >> (64*limb)) & 0xffffffffffffffff
                    nh = native_wide(wide_values, wide_address, size-1, small_output)
                    # If all base primes have been tried and the value
                    # is still wide, it exceeds the allowed cofactor.
                    if wide_values[2] or wide_values[3]:
                        continue
                    small_values[0], small_values[1] = wide_values[0], wide_values[1]
                    nh += native_decompose(small_values, inverse_address,
                                           size-1, output_address + nh*8,
                                           prime_indices, bound)
                    remainder = small_values[0] + (small_values[1] << 64)
                    last_index = prime_indices[remainder] if 1 < remainder <= bound else 0
                    if last_index:
                        remainder = 1
                else:
                    remainder, common = value, gcd(base_product, value)
                    while common != 1:
                        remainder //= common
                        common = gcd(remainder, common)
                if remainder > large_bound:
                    continue
                cofactor_root = isqrt(remainder)
                cofactor_square = cofactor_root * cofactor_root == remainder
                u = (2 * a * x + b) % n
                negative = ((a * x + b) * x + c) < 0
                if native_value:
                    # Most partial relations are never paired; retain compact
                    # native exponents until the relation is actually used.
                    packed = ctypes.string_at(small_output, nh * 8)
                    raw = [u, packed, amask, negative, None, e, last_index]
                else:
                    fs = list(_factor_over_base(value // remainder, tree))
                    raw = [u, fs, amask, negative, None, 0, 0]
                if cofactor_square:
                    u, square, mask = decompose(raw)
                    square = square * cofactor_root % n
                elif remainder != 1:
                    old = partials.get(remainder)
                    if old is None:
                        partials[remainder] = raw
                        continue
                    if raw[:4] == old[:4] and raw[5:] == old[5:]:
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
                          "rank": rank, "base": size,
                          "partials": len(partials), "matched": matched,
                          "candidates": candidates})


def _qs_python(n, rng, check, progress=None):
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
    # Long C-level bytearray operations amortize Python's per-prime overhead.
    half = 1 << max(12, min(20, (bound * 8).bit_length()))
    width = 2 * half
    target = isqrt(2 * kn) // half
    # Odd B gives Q(x)=A*x*x+B*x+(B*B-k*n)/(4*A).
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
    # A byte holds a scaled logarithm; the sieve checks its overflow bound.
    scale = min(2.0, 220.0 / (log2(kn) / 2 + log2(half) + 8))
    weights = [max(1, round(log2(p) * scale)) for p in fb]
    translations = {w: bytes(min(255, x + w) for x in range(256))
                    for w in set(weights)}
    template_stop = bisect_left(fb, 11)
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
        regular = [(i, p, translations[weights[i]]) for i, p in enumerate(fb)
                   if inverse[i] and p >= 20 and roots[i]]
        special = [(i, p, weights[i]) for i, p in enumerate(fb)
                   if p >= 11 and (not inverse[i] or p < 20 or not roots[i])]
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
            # 5040 = 2**4 * 3**2 * 5 * 7. Build the frequent small-prime
            # contributions once on this short period, then copy it in C.
            period = 5040
            template = bytearray(period)
            deferred = []
            for i in range(template_stop):
                p = fb[i]
                table = translations[weights[i]]
                if inverse[i]:
                    first, second = r1[i], r2[i]
                else:
                    first = (half - c * pow(2 * b, -1, p)) % p
                    second = first
                template[first::p] = template[first::p].translate(table)
                if first != second:
                    template[second::p] = template[second::p].translate(table)
                modulus, current = p, {first, second}
                while modulus * p <= 256 and current:
                    extended = set()
                    for root in current:
                        for j in range(p):
                            root2 = root + j * modulus
                            x = root2 - half
                            if ((a * x + 2 * b) * x + c) % (modulus * p) == 0:
                                extended.add(root2)
                    modulus *= p
                    if period % modulus == 0:
                        for root in extended:
                            template[root::modulus] = template[root::modulus].translate(table)
                    else:
                        for root in extended:
                            deferred.append((root, modulus, table))
                    current = extended
            sieve = template * ((width + period - 1) // period)
            del sieve[width:]
            for root, modulus, table in deferred:
                sieve[root::modulus] = sieve[root::modulus].translate(table)
            for i, p, table in regular:
                first, second = r1[i], r2[i]
                sieve[first::p] = sieve[first::p].translate(table)
                sieve[second::p] = sieve[second::p].translate(table)
            for i, p, weight in special:
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
                    while modulus * p <= 256 and current:
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


def _qs(n, rng, check, progress=None):
    """Use the embedded System V AMD64 kernels when the platform supports them."""
    if _NATIVE_AVAILABLE:
        try:
            return _qs_native(n, rng, check, progress)
        except (OSError, PermissionError):
            # Some hosts prohibit executable anonymous mappings.
            check()
    return _qs_python(n, rng, check, progress)


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
