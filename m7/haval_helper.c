// haval_helper.c - Core passes and finishing functions for HAVAL

#include <stddef.h>
#include <string.h>
#include "sph_haval.h"

#if !defined(PASSES)
#error "PASSES macro must be defined before including haval_helper.c"
#endif

// Context type
typedef sph_haval_context context_t;

// Internal block size for HAVAL: 1024 bits = 128 bytes
#define HAVAL_BLOCK_SIZE 128

// Forward declarations of generic core and finish functions
static void haval_update(context_t *sc, const void *data, size_t len);
static void haval_finish(context_t *sc, unsigned ub, unsigned n, void *dst);

// Transformation function wrapper selected by PASSES
static void haval_core(context_t *sc, const unsigned char *data) {
	DSTATE;
	RSTATE;

#if PASSES == 3
	// Prepare message words
#if SPH_LITTLE_FAST
	const unsigned char *load_ptr = data;
#define INMSG(i) sph_dec32le_aligned(load_ptr + 4*(i))
#elif !SPH_LITTLE_FAST
	sph_u32 msg[32];
	for (int i=0; i<32; i++)
		msg[i] = sph_dec32le_aligned(data + 4*i);
#define INMSG(i) msg[i]
#else
#error "Endianness setting incompatible in haval_helper"
#endif

	CORE3(INMSG);
#elif PASSES == 4
#if SPH_LITTLE_FAST
	const unsigned char *load_ptr = data;
#define INMSG(i) sph_dec32le_aligned(load_ptr + 4*(i))
#elif !SPH_LITTLE_FAST
	sph_u32 msg[32];
	for (int i=0; i<32; i++)
		msg[i] = sph_dec32le_aligned(data + 4*i);
#define INMSG(i) msg[i]
#else
#error "Endianness setting incompatible in haval_helper"
#endif

	CORE4(INMSG);
#elif PASSES == 5
#if SPH_LITTLE_FAST
	const unsigned char *load_ptr = data;
#define INMSG(i) sph_dec32le_aligned(load_ptr + 4*(i))
#elif !SPH_LITTLE_FAST
	sph_u32 msg[32];
	for (int i=0; i<32; i++)
		msg[i] = sph_dec32le_aligned(data + 4*i);
#define INMSG(i) msg[i]
#else
#error "Endianness setting incompatible in haval_helper"
#endif

	CORE5(INMSG);
#endif
}

// Public API functions invoked by main HAVAL implementation for update/finalize

void haval##PASSES(context_t *sc, const void *data, size_t len) {
	const unsigned char *buf = (const unsigned char *)data;
	size_t used = (sc->count & 0x7F);
	size_t fill = HAVAL_BLOCK_SIZE - used;

	sc->count += (unsigned)len;

	if (used && len >= fill) {
		memcpy(&sc->buf[used], buf, fill);
		haval_core(sc, sc->buf);
		buf += fill;
		len -= fill;
		used = 0;
	}

	while (len >= HAVAL_BLOCK_SIZE) {
		haval_core(sc, buf);
		buf += HAVAL_BLOCK_SIZE;
		len -= HAVAL_BLOCK_SIZE;
	}

	if (len > 0) {
		memcpy(&sc->buf[used], buf, len);
	}

	sc->count_low = (unsigned)(sc->count & 0xFFFFFFFF);
	sc->count_high = (unsigned)(sc->count >> 32);
}

void haval##PASSES##_close(context_t *sc, unsigned ub, unsigned n, void *dst) {
	unsigned char *buf = sc->buf;
	unsigned used = (unsigned)(sc->count & 0x7F);

	// Add padding
	buf[used ++] = 0x01;

	if (used > HAVAL_BLOCK_SIZE - 10) {
		memset(buf + used, 0, HAVAL_BLOCK_SIZE - used);
		haval_core(sc, buf);
		used = 0;
	}

	memset(buf + used, 0, HAVAL_BLOCK_SIZE - used - 10);

	// Add bit counts and fingerprint selector
	sph_enc64le(buf + HAVAL_BLOCK_SIZE - 10, sc->count << 3);

	buf[HAVAL_BLOCK_SIZE - 2] = (unsigned char)((sc->olen * 8) & 0xFF);
	buf[HAVAL_BLOCK_SIZE - 1] = (unsigned char)((PASSES << 3) | (sc->olen & 7));

	haval_core(sc, buf);
	haval_out(sc, dst);
}
