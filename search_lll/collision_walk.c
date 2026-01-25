// collision_walk.c
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// Return codes:
//  1 -> found distinguished point
//  0 -> abandoned (too many terms or max steps)
// -1 -> error

int collision_walk(
    const uint32_t *atom_indices,   // length n_atoms (pos -> global atom idx)
    uint32_t n_atoms,
    const uint64_t *rand_table,     // length n_atoms (pos -> rand value)
    uint64_t target_mask,
    uint32_t max_terms,
    uint64_t seed,
    uint32_t *touched,              // out: touched positions (pos)
    uint32_t *counts,               // out: counts for touched (counts per pos)
    uint32_t *exps,                 // scratch: length n_atoms (must be zeroed by caller)
    uint32_t *out_len,              // out
    uint64_t *out_state,            // out
    uint64_t max_steps              // ADDED: hard step limit
) {
    if (!atom_indices || !rand_table || !touched || !counts || !exps || !out_len || !out_state)
        return -1;
    if (n_atoms == 0 || max_terms == 0)
        return -1;

    uint64_t state = seed;
    uint32_t touched_len = 0;
    uint64_t steps = 0;

    while (1) {
        steps++;
        
        // HARD STEP LIMIT
        if (steps > max_steps) {
            // cleanup and abandon
            for (uint32_t i = 0; i < touched_len; i++) {
                exps[touched[i]] = 0;
            }
            *out_len = 0;
            return 0;  // timeout/abandon
        }

        uint32_t pos = (uint32_t)(state % n_atoms);

        if (exps[pos] == 0) {
            if (touched_len >= max_terms) {
                // cleanup and abandon
                for (uint32_t i = 0; i < touched_len; i++) {
                    exps[touched[i]] = 0;
                }
                *out_len = 0;
                return 0;
            }
            touched[touched_len++] = pos;
        }

        exps[pos]++;

        // FIXED: Use high bits for DP check + inject Weyl sequence to avoid short cycles
        state = mix64(state + rand_table[pos] + 0x9e3779b97f4a7c15ULL);

        // FIXED: Check high bits, not low bits
        if ((state >> (64 - __builtin_ctzll(target_mask + 1))) == 0) {
            // DP hit
            for (uint32_t i = 0; i < touched_len; i++) {
                uint32_t a = touched[i];
                counts[i] = exps[a];
                exps[a] = 0;  // reset for next walk
            }
            *out_len = touched_len;
            *out_state = state;
            return 1;
        }
    }
}


#ifdef __cplusplus
}
#endif
