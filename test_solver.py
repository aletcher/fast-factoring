"""Correctness tests independent of the supplied benchmark answers."""

from math import gcd, prod
from random import Random
from time import process_time
import unittest

import solver


class SolverTests(unittest.TestCase):
    def test_all_small_semiprimes(self):
        primes = solver._primes(100)
        for i, p in enumerate(primes):
            for q in primes[i:]:
                with self.subTest(p=p, q=q):
                    self.assertEqual(solver.factor_semiprime(p * q), (p, q))

    def test_squares_and_large_unequal_factors(self):
        # Known Mersenne primes create independent inputs at several sizes.
        p, q = (1 << 31) - 1, (1 << 127) - 1
        for a, b in ((p, p), (q, q), (3, q), (2003, q)):
            with self.subTest(bits=(a * b).bit_length()):
                self.assertEqual(solver.factor_semiprime(a * b), (a, b))

    def test_fresh_random_semiprimes(self):
        rng = Random(20261007)
        for bits in (12, 18, 24, 32, 40):
            for _ in range(3):
                factors = []
                for _ in range(2):
                    p = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
                    while not solver._prime64(p):
                        p += 2
                    factors.append(p)
                self.assertEqual(solver.factor_semiprime(prod(factors), timeout=10),
                                 tuple(sorted(factors)))

    def test_sieve_directly_including_double_large_cofactors(self):
        rng = Random(731)
        for bits in (30, 63):
            factors = []
            for _ in range(2):
                p = rng.getrandbits(bits) | (1 << (bits - 1)) | 1
                while not solver._prime64(p):
                    p += 2
                factors.append(p)
            n = prod(factors)
            start = process_time()

            def check():
                if process_time() - start > 20:
                    self.fail("Independent sieve correctness test exceeded 20 CPU seconds")

            self.assertIn(solver._qs(n, Random(n), check), factors)

    def test_batch_smoothness_with_prime_powers(self):
        primes = solver._primes(500)
        tree = solver._product_tree(primes)
        product = prod(primes)
        rng = Random(983)
        values, expected = [1], [{}]
        for _ in range(50):
            exponents = {}
            for _ in range(8):
                i = rng.randrange(len(primes))
                exponents[i] = exponents.get(i, 0) + rng.randrange(1, 6)
            values.append(prod(primes[i] ** e for i, e in exponents.items()))
            expected.append(exponents)
        self.assertEqual(solver._batch_gcd(product, values),
                         [gcd(product, value) for value in values])
        for value, exponents in zip(values, expected):
            self.assertEqual(dict(solver._factor_over_base(value, tree)), exponents)

    def test_decimal_string_and_aliases(self):
        self.assertEqual(solver.solve("10007000070049"), (10007, 1000000007))
        self.assertIs(solver.factor, solver.factor_semiprime)

    def test_invalid_inputs(self):
        for value in (-10, 0, 1, 2, 3, 97, 2003, (1 << 61) - 1):
            with self.subTest(n=value), self.assertRaises(ValueError):
                solver.factor_semiprime(value)
        for value in (True, False, 15.0, None, [], b"15"):
            with self.subTest(n=value), self.assertRaises(TypeError):
                solver.factor_semiprime(value)
        for budget in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=budget), self.assertRaises(ValueError):
                solver.factor_semiprime(15, timeout=budget)

    def test_timeout_is_failure_not_a_factorization(self):
        with self.assertRaises(TimeoutError):
            solver.factor_semiprime(((1 << 61) - 1) * ((1 << 127) - 1),
                                    timeout=1e-12)


if __name__ == "__main__":
    unittest.main()
