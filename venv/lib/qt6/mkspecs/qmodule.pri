QT_CPU_FEATURES.x86_64 = cx16 sse3
QT.global_private.enabled_features = reduce_exports x86intrin sse2 sse3 ssse3 sse4_1 sse4_2 avx f16c avx2 avx512f avx512er avx512cd avx512pf avx512dq avx512bw avx512vl avx512ifma avx512vbmi avx512vbmi2 aesni vaes rdrnd rdseed shani posix_fallocate alloca_h alloca system-zlib dbus gui network printsupport sql testlib widgets xml openssl dlopen relocatable largefile sse2 sse3 ssse3 sse4_1 sse4_2 avx f16c avx2 avx512f avx512er avx512cd avx512pf avx512dq avx512bw avx512vl avx512ifma avx512vbmi avx512vbmi2 aesni vaes rdrnd rdseed shani
QT.global_private.disabled_features = use_bfd_linker use_gold_linker use_lld_linker use_mold_linker android-style-assets gc_binaries developer-build private_tests debug elf_private_full_version no_direct_extern_access mips_dsp mips_dspr2 neon arm_crc32 arm_crypto alloca_malloc_h stack-protector-strong stdlib-libcpp dbus-linked libudev intelcet
CONFIG += largefile sse2 sse3 ssse3 sse4_1 sse4_2 avx f16c avx2 avx512f avx512er avx512cd avx512pf avx512dq avx512bw avx512vl avx512ifma avx512vbmi avx512vbmi2 aesni vaes rdrnd rdseed shani
PKG_CONFIG_EXECUTABLE = /croot/qtbase_1742934294744/_build_env/bin/pkg-config
QT_COORD_TYPE = double
QT_BUILD_PARTS = libs tools

QMAKE_LIBS_ZLIB = -lz
QMAKE_LIBS_ZSTD = -lzstd -lzstd
QMAKE_INCDIR_DBUS = /workspaces/ds-eda-01-iris-dataset/venv/include/dbus-1.0 /workspaces/ds-eda-01-iris-dataset/venv/lib/dbus-1.0/include
QMAKE_LIBS_DBUS = -ldbus-1
